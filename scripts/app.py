#!/usr/bin/env python3
"""
S-Parameter Web Dashboard
Flask 驱动的本地 Web 仪表盘：上传 .sNp → 交互式 Plotly 图表 → 导出
启动: python app.py  →  浏览器打开 http://localhost:5050
"""

import os
import sys
import io
import json
import tempfile
import traceback
import html
import re
from pathlib import Path

import numpy as np
import skrf as rf
import plotly
from flask import Flask, render_template, request, jsonify, send_file

# ── PyInstaller 兼容：定位资源目录 ──
def _base_dir():
    """返回应用根目录（兼容 PyInstaller 打包和源码运行）。"""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

_BASE = _base_dir()

# ── 导入本地工具库 ───────────────────────────────────────────
sys.path.insert(0, _BASE)
import s_params as sp
import nl_parser
import code_agent
import agent_plan
import network_inspector

app = Flask(__name__, template_folder=os.path.join(_BASE, "templates"))
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB 上传上限（支持大端口文件如 .s64p）

# 会话级存储：上传文件的解析结果
sessions = {}  # { session_id: { "networks": {name: Network}, "freq_unit": "ghz" } }
EAGER_LOAD_MAX_PORTS = 7
EAGER_LOAD_MAX_BYTES = 25 * 1024 * 1024
_MIXED_PARAM_RE = re.compile(
    r"(?<![A-Z0-9_])S(?:DD|DC|CD|CC)(?:\d+_\d+|[1-9][1-9])(?![A-Z0-9_])",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────
#  页面路由
# ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """主仪表盘页面"""
    return render_template("dashboard.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "libraries": {"scikit-rf": rf.__version__, "plotly": plotly.__version__}})


@app.route("/api/chat/mode")
def chat_mode():
    """返回当前 LLM 配置状态"""
    cfg = code_agent._get_llm_config()
    return jsonify({
        "llm_available": cfg is not None,
        "mode": "llm" if cfg else "unavailable",
        "llm_model": cfg.get("model", "") if cfg else "",
        "llm_base_url": cfg.get("base_url", "") if cfg else "",
    })


# ──────────────────────────────────────────────────────────────
#  🤖 代码生成 Agent API (强约束模式)
# ──────────────────────────────────────────────────────────────

@app.route("/api/agent", methods=["POST"])
def agent():
    """
    强约束代码生成 Agent。
    接收: { "text": "画 S21 dB 图", "session": "default" }
    LLM → 生成受限 Python 代码 → AST 校验 → 沙箱执行 → 返回图表
    """
    data = request.get_json()
    if data is None:
        return jsonify({"reply": "请求体为空或非 JSON", "results": []})

    try:
        session_id = data.get("session", "default")
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"reply": "请说点什么", "results": []})

        if not code_agent.is_available():
            return jsonify({
                "reply": "需要配置 LLM API Key。设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 环境变量。",
                "results": [],
                "mode": "unavailable",
            })

        # 获取当前会话中的文件路径（用绝对路径）
        ses = sessions.get(session_id, {}).get("networks", {})
        file_path = None
        if ses:
            last = list(ses.values())[-1]
            last_path = last.get("path", "")
            if last_path:
                file_path = os.path.abspath(last_path)

        # 如果文本中包含文件路径，优先使用（提取 .sNp 路径）
        import re as _re
        snp_match = _re.search(r"([\w./\\-]+\.s\dp)", text, _re.IGNORECASE)
        if snp_match:
            file_path = os.path.abspath(snp_match.group(1))

        # ── 构建网络字典（名称 → dict），供 Agent 执行时预加载 ──
        nets_dict = {}
        for net_name, net_info in ses.items():
            nets_dict[net_name] = _agent_network_payload(net_info)

        _apply_user_mixed_mode_confirmation(text, nets_dict, ses)
        mixed_block = _mixed_mode_confirmation_response(text, nets_dict)
        if mixed_block:
            return jsonify(mixed_block)

        # ── 先让 LLM 产出短 RF Plan；能确定性执行的请求不再生成代码 ──
        plan_result = code_agent.build_rf_plan(
            text,
            file_path=file_path,
            networks=nets_dict if nets_dict else None,
        )
        if plan_result.get("needs_confirmation"):
            reply = plan_result.get("reply", "需要确认后才能执行")
            return jsonify({
                "reply": reply,
                "results": [{"type": "error", "message": reply}],
                "mode": "agent",
                "needs_confirmation": True,
                "plan": plan_result.get("plan"),
                "planning_history": plan_result.get("history", []),
            })
        if not plan_result.get("ok"):
            return jsonify({
                "reply": f"❌ Agent 规划失败: {plan_result.get('error', '未知错误')}",
                "results": [],
                "mode": "agent",
                "plan": plan_result.get("plan"),
                "planning_history": plan_result.get("history", []),
            })

        plan = plan_result["plan"]
        planning_history = plan_result.get("history", [])
        deterministic = _try_execute_rf_plan(
            session_id,
            plan,
            nets_dict if nets_dict else {},
            planning_history=planning_history,
        )
        if deterministic is not None:
            return jsonify(deterministic)

        # ── 超出确定性执行范围时，再调用代码生成 Agent ──
        result = code_agent.generate_code(text, file_path=file_path,
                                           networks=nets_dict if nets_dict else None,
                                           rf_plan=plan,
                                           planning_history=planning_history)

        if result.get("needs_confirmation"):
            reply = result.get("reply", "需要确认后才能执行")
            return jsonify({
                "reply": reply,
                "results": [{"type": "error", "message": reply}],
                "mode": "agent",
                "needs_confirmation": True,
                "plan": result.get("plan"),
                "planning_history": result.get("planning_history", []),
            })

        if "error" in result:
            return jsonify({
                "reply": f"❌ {result['error']}",
                "results": [],
                "mode": "agent",
                "code": result.get("code", ""),
                "llm_raw": result.get("llm_raw", ""),
                "plan": result.get("plan"),
                "planning_history": result.get("planning_history", []),
            })

        code = result["code"]
        validated = result["validated"]
        validation_msg = result["validation_msg"]
        exec_r = result["exec_result"]

        results = []

        # 校验失败 → 显示代码 + 错误
        if not validated:
            results.append({
                "type": "error",
                "message": f"代码校验未通过: {validation_msg}",
            })
            return jsonify({
                "reply": f"❌ 生成的代码未通过安全检查: {validation_msg}",
                "results": results,
                "mode": "agent",
                "code": code,
                "validated": False,
            })

        # 执行失败
        if not exec_r.get("ok"):
            error_msg = exec_r.get("error", "未知错误")
            stderr = exec_r.get("stderr", "")
            results.append({
                "type": "error",
                "message": f"执行失败:\n{error_msg}",
            })
            if stderr:
                results.append({"type": "text", "message": f"stderr:\n{stderr}"})
            return jsonify({
                "reply": f"❌ 代码执行出错: {error_msg[:200]}",
                "results": results,
                "mode": "agent",
                "code": code,
                "exec_stdout": exec_r.get("stdout", ""),
                "exec_stderr": stderr,
                "retries": result.get("retries", 0),
                "plan": result.get("plan"),
                "planning_history": result.get("planning_history", []),
                "history": result.get("history", []),
            })

        # 执行成功
        reply_parts = ["✅ 代码生成并执行成功"]

        if exec_r.get("stdout"):
            results.append({"type": "text", "message": f"输出:\n{exec_r['stdout']}"})

        # 渲染图表
        if exec_r.get("figure_json"):
            fig_data = exec_r["figure_json"].get("data", [])
            fig_layout = exec_r["figure_json"].get("layout", {})
            results.append({
                "type": "chart",
                "chart": {"data": fig_data, "layout": fig_layout},
                "title": text[:60],
            })
            reply_parts.append("📊 图表已生成")

        return jsonify({
            "reply": "\n".join(reply_parts),
            "results": results,
            "mode": "agent",
            "code": code,
            "validated": True,
            "exec_stdout": exec_r.get("stdout", ""),
            "retries": result.get("retries", 0),
            "plan": result.get("plan"),
            "planning_history": result.get("planning_history", []),
            "history": result.get("history", []),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "reply": f"❌ 服务器内部错误: {str(e)[:300]}",
            "results": [],
            "mode": "agent",
        })


# ──────────────────────────────────────────────────────────────
#  🤖 自然语言对话 API (旧版，保留兼容)
# ──────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    自然语言对话端点。
    接收: { "text": "读 amp.s2p，画 S21 dB 图", "session": "default" }
    返回: { "reply": "...", "results": [{ type: "chart"|"text"|"error", ... }] }
    """
    data = request.get_json()
    session_id = data.get("session", "default")
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"reply": "请说点什么吧 😊", "results": [], "handled": False})

    # 列出已有文件
    available = list(sessions.get(session_id, {}).get("networks", {}).keys())

    # ── 解析自然语言：规则优先，LLM 作为兜底 ──
    ops = None
    parse_mode = "rule"

    try:
        ops = nl_parser.parse(text, available)
    except Exception as e:
        return jsonify({"reply": f"解析失败: {str(e)}", "results": [], "handled": False})

    if ops and _should_defer_to_agent(text, ops):
        ops = []

    if not ops:
        return jsonify({
            "reply": "不太确定你想做什么。试试这样说：\n"
                     "• \"读取 filter.s2p\"\n"
                     "• \"画 S11 和 S21 的 dB 图\"\n"
                     "• \"级联 A 和 B，画 S21 Smith 圆图\"\n"
                     "• \"导出 S21 为 CSV\"",
            "results": [],
            "handled": False,
        })

    # ── 初始化 session ──
    if session_id not in sessions:
        sessions[session_id] = {"networks": {}}

    # ── 执行操作 ──
    results = []
    last_ntwk_name = None

    for op in ops:
        try:
            result = _execute_op(op, session_id, last_ntwk_name)
            results.append(result)
            if result.get("ntwk_name"):
                last_ntwk_name = result["ntwk_name"]
        except Exception as e:
            traceback.print_exc()
            results.append({"type": "error", "message": f"执行 [{op.action}] 失败: {str(e)}"})

    # ── 生成回复 ──
    reply_lines = []
    for r in results:
        if r["type"] == "chart":
            reply_lines.append(f"📊 {r.get('title', '图表')} 已生成")
        elif r["type"] == "text":
            reply_lines.append(r["message"])
        elif r["type"] == "error":
            reply_lines.append(f"❌ {r['message']}")

    return jsonify({
        "reply": "\n".join(reply_lines) if reply_lines else "完成！",
        "results": results,
        "ops_debug": nl_parser.format_ops(ops),
        "parse_mode": parse_mode,
        "handled": True,
    })


def _should_defer_to_agent(text: str, ops: list) -> bool:
    """复杂 RF 意图不让规则路径默认展开成全部参数，交给代码 Agent。"""
    text = text or ""
    has_mixed_param = _MIXED_PARAM_RE.search(text) is not None
    has_mixed_words = any(word in text for word in ("差分", "共模", "混模"))
    if not (has_mixed_param or has_mixed_words):
        return False
    for op in ops:
        if op.action in ("plot", "export", "compare") and not op.params:
            return True
    return False


def _is_mixed_mode_request(text: str) -> bool:
    text = text or ""
    return _MIXED_PARAM_RE.search(text) is not None or any(word in text for word in ("差分", "共模", "混模"))


def _mixed_mode_confirmation_response(text: str, networks: dict):
    """Stop ambiguous mixed-mode requests before the Agent fabricates a P/N mapping."""
    if not _is_mixed_mode_request(text) or not networks:
        return None

    name = list(networks.keys())[-1]
    info = networks[name]
    mixed = info.get("mixed_mode") or {}
    if mixed.get("status") == "ready":
        return None

    port_names = info.get("port_names") or [f"Port{i + 1}" for i in range(info.get("nports", 0))]
    lines = [
        f"我不能安全确定 **{name}** 的差分 P/N 端口，所以没有生成权威混模图。",
    ]
    through = mixed.get("through_paths") or []
    if through:
        lines.append("当前只能识别到疑似单端传输路径: " + _format_port_pairs(through, port_names))
    alternatives = mixed.get("alternatives") or []
    if alternatives:
        lines.append("可能的差分端点候选:")
        for idx, alt in enumerate(alternatives[:3], start=1):
            pairs = [
                pair.get("ports", [])
                for pair in alt.get("differential_pairs", [])
                if len(pair.get("ports", [])) == 2
            ]
            lines.append(f"  {idx}. {_format_port_pairs(pairs, port_names)}")
    lines.append("请在 Touchstone 端口名中标出 _P/_N、+/-、DP/DN，或明确告诉我端口配对，例如: P/N 配对为 (1,3),(2,4)。")

    message = "\n".join(lines)
    return {
        "reply": message,
        "results": [{"type": "error", "message": message}],
        "mode": "agent",
        "needs_confirmation": True,
        "mixed_mode": mixed,
    }


def _apply_user_mixed_mode_confirmation(text: str, networks: dict, ses: dict):
    if not _is_mixed_mode_request(text) or not networks:
        return
    name = list(networks.keys())[-1]
    info = networks[name]
    pairs = _parse_user_mixed_mode_pairs(text, int(info.get("nports") or 0))
    if not pairs:
        return

    mixed = {
        "status": "ready",
        "source": "user",
        "pair_count": len(pairs),
        "differential_pairs": [
            {
                "ports": [p, n],
                "p": p,
                "n": n,
                "source": "user",
                "confidence": "high",
            }
            for p, n in pairs
        ],
        "se2gmm_order": [port for pair in pairs for port in pair],
        "through_paths": (info.get("mixed_mode") or {}).get("through_paths", []),
        "confidence": {
            "endpoint_pairing": "high",
            "polarity": "user",
        },
    }
    info["mixed_mode"] = mixed
    if name in ses:
        ses[name]["mixed_mode"] = mixed


def _parse_user_mixed_mode_pairs(text: str, nports: int):
    if nports < 2:
        return []
    matches = re.findall(r"\(\s*(\d+)\s*[,，/]\s*(\d+)\s*\)", text or "")
    if not matches:
        return []
    pairs = []
    used = set()
    for p_text, n_text in matches:
        p = int(p_text) - 1
        n = int(n_text) - 1
        if p < 0 or n < 0 or p >= nports or n >= nports or p == n:
            return []
        if p in used or n in used:
            return []
        pairs.append((p, n))
        used.update((p, n))
    if len(pairs) * 2 != nports:
        return []
    return pairs


def _format_port_pairs(pairs, port_names) -> str:
    formatted = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        a, b = int(pair[0]), int(pair[1])
        a_name = port_names[a] if 0 <= a < len(port_names) else f"Port{a + 1}"
        b_name = port_names[b] if 0 <= b < len(port_names) else f"Port{b + 1}"
        formatted.append(f"({a_name}/{a + 1}, {b_name}/{b + 1})")
    return ", ".join(formatted) if formatted else "无"


def _execute_op(op, session_id: str, last_ntwk_name: str = None) -> dict:
    """执行单个 SParamOp，返回结果 dict。"""
    ses = sessions.setdefault(session_id, {"networks": {}})

    # ── 自动补全 target：从 last_ntwk_name 或 session 中取最新网络 ──
    if not op.target and op.action in ("plot", "slice", "export", "info", "cascade"):
        if last_ntwk_name:
            op.target = last_ntwk_name
        elif ses["networks"]:
            op.target = list(ses["networks"].keys())[-1]

    # ── LOAD ──
    if op.action == "load":
        if not op.target:
            return {"type": "error", "message": "没有指定文件名"}
        # 如果 target 是目录，走批量加载
        if os.path.isdir(op.target):
            return _load_dir_to_session(op.target, session_id, ses)
        # 尝试在当前目录及子目录搜索文件
        import glob as globmod
        candidates = globmod.glob(f"**/{op.target}", recursive=True)
        if not candidates:
            candidates = globmod.glob(f"**/{os.path.basename(op.target)}", recursive=True)
        # 检查候选是否为目录
        if candidates and os.path.isdir(candidates[0]):
            return _load_dir_to_session(candidates[0], session_id, ses)
        if candidates:
            op.target = candidates[0]

        name = os.path.splitext(os.path.basename(op.target))[0]
        info = _register_path(session_id, op.target, name=name, dedup=True)
        ntwk = _get_network(session_id, info["name"]) if info.get("loaded") else None
        if ntwk is not None:
            info_str = sp.info(ntwk)
        else:
            info_str = (
                f"{info['nports']}端口, {info['f_min']/1e9:.3f}-{info['f_max']/1e9:.3f} GHz, "
                f"{info['npoints']}点 (metadata only, 按需加载)"
            )
        return {
            "type": "text",
            "message": f"✅ 已加载 **{info['name']}**\n{info_str}",
            "ntwk_name": info["name"],
        }

    # ── INFO ──
    if op.action == "info":
        name = op.target or last_ntwk_name
        ntwk = _get_network(session_id, name)
        if ntwk is None:
            return {"type": "error", "message": f"找不到网络 '{name}'，请先加载文件"}
        info_str = sp.info(ntwk)
        return {
            "type": "text",
            "message": f"📋 **{name}**\n{info_str}",
            "ntwk_name": name,
        }

    # ── CASCADE ──
    if op.action == "cascade":
        name_a = op.target or last_ntwk_name
        name_b = op.cascade_with
        if not name_b:
            return {"type": "error", "message": "级联需要两个网络，例如：级联 A.s2p 和 B.s2p"}
        ntwk_a = _get_network(session_id, name_a)
        ntwk_b = _get_network(session_id, name_b)
        if ntwk_a is None or ntwk_b is None:
            return {"type": "error", "message": f"找不到网络，已加载: {list(ses['networks'].keys())}"}

        ntwk_result = sp.cascade(ntwk_a, ntwk_b)
        result_name = op.result_name or f"{_basename(name_a)}+{_basename(name_b)}"
        tmp_path = tempfile.mktemp(suffix=f".s{ntwk_result.nports}p")
        ntwk_result.write_touchstone(tmp_path)
        _store_network(session_id, result_name, ntwk_result, tmp_path)
        return {
            "type": "text",
            "message": f"🔗 级联完成 → **{result_name}** ({ntwk_result.nports}端口)",
            "ntwk_name": result_name,
        }

    # ── SLICE ──
    if op.action == "slice":
        name = op.target or last_ntwk_name
        ntwk = _get_network(session_id, name)
        if ntwk is None:
            return {"type": "error", "message": f"找不到网络 '{name}'"}
        if not op.freq_range:
            return {"type": "error", "message": "请指定频率范围，例如：2-4GHz"}

        sliced = sp.slice_freq(ntwk, op.freq_range[0], op.freq_range[1])
        # 更新 session 中的网络
        sliced_name = name
        tmp_path = tempfile.mktemp(suffix=f".s{sliced.nports}p")
        sliced.write_touchstone(tmp_path)
        _store_network(session_id, sliced_name, sliced, tmp_path)
        return {
            "type": "text",
            "message": f"✂️ 已截取 {op.freq_range[0]/1e9:.2f}–{op.freq_range[1]/1e9:.2f} GHz ({len(sliced.f)} 点)",
            "ntwk_name": sliced_name,
        }

    # ── PLOT ──
    if op.action == "plot":
        name = op.target or last_ntwk_name
        ntwk = _get_network(session_id, name)
        if ntwk is None:
            return {"type": "error", "message": f"找不到网络 '{name}'，请先加载文件"}

        if op.freq_range:
            ntwk = sp.slice_freq(ntwk, op.freq_range[0], op.freq_range[1])

        params = op.params if op.params else None
        chart_type = op.chart_type or "db"

        # 双Y轴
        if getattr(op, "dual_axis", False) and chart_type in ("db", "mag"):
            # 拆分用户指定参数：反射参数(m==n)左轴，传输参数(m!=n)右轴
            parsed = sp._parse_params(ntwk, params) if params else []
            left_p = [(m,n) for m,n in parsed if m==n]
            right_p = [(m,n) for m,n in parsed if m!=n]
            if not left_p and not right_p:
                fig = sp.plot_s_db_dual(ntwk, title=op.title or f"{_basename(name)} Dual Y-Axis")
            else:
                fig = sp.plot_s_db_dual(ntwk, left_params=left_p or None, right_params=right_p or None,
                                        title=op.title or f"{_basename(name)} Dual Y-Axis")
        # VSWR 处理
        elif chart_type == "vswr":
            ports = sp._parse_vswr_params(ntwk, params or ["VSWR1"])
            fig = sp.plot_vswr(ntwk, ports, title=op.title or f"{_basename(name)} VSWR")
        else:
            fig = _dispatch_plot(ntwk, params, chart_type, op.title, name)

        chart_json = json.loads(json.dumps(
            {"data": fig.data, "layout": fig.layout},
            cls=plotly.utils.PlotlyJSONEncoder,
        ))
        return {
            "type": "chart",
            "chart": chart_json,
            "title": op.title or f"{_basename(name)} {','.join(params or ['S'])} ({chart_type})",
            "ntwk_name": name,
        }

    # ── EXPORT ──
    if op.action == "export":
        name = op.target or last_ntwk_name
        ntwk = _get_network(session_id, name)
        if ntwk is None:
            return {"type": "error", "message": f"找不到网络 '{name}'"}

        params = op.params if op.params else None
        fmt = op.export_format or "csv"

        if fmt == "csv":
            out_path = tempfile.mktemp(suffix=".csv")
            sp.save_csv(ntwk, params, out_path)
            return {
                "type": "text",
                "message": f"💾 已导出 CSV: `{out_path}`",
                "file": out_path,
            }
        elif fmt == "touchstone":
            out_path = tempfile.mktemp(suffix=f".s{ntwk.nports}p")
            sp.save_touchstone(ntwk, out_path)
            return {
                "type": "text",
                "message": f"💾 已导出 Touchstone: `{out_path}`",
                "file": out_path,
            }
        elif fmt == "html":
            fig = sp.plot_s_db(ntwk, params, title=op.title or f"{_basename(name)}")
            out_path = tempfile.mktemp(suffix=".html")
            fig.write_html(out_path, include_plotlyjs="cdn")
            return {
                "type": "text",
                "message": f"💾 已导出 HTML 报告: `{out_path}`",
                "file": out_path,
            }

    # ── LIST ──
    if op.action == "list":
        names = list(ses["networks"].keys())
        if not names:
            return {"type": "text", "message": "📭 尚未加载任何文件。拖拽 .sNp 文件上传或说\"读取 xxx.s2p\""}
        lines = ["📋 已加载的网络："]
        for n in names:
            info = ses["networks"][n]
            lines.append(f"  • **{n}** — {info['nports']}端口, "
                        f"{info['f_min']/1e9:.3f}–{info['f_max']/1e9:.3f} GHz, {info['npoints']}点")
        return {"type": "text", "message": "\n".join(lines)}

    # ── CASCADE_CHAIN ──
    if op.action == "cascade_chain":
        chain_names = op.chain if op.chain else [op.target, op.cascade_with]
        chain_names = [n for n in chain_names if n]
        if len(chain_names) < 2:
            return {"type": "error", "message": "链式级联需要至少 2 个网络，例如: LNA → BPF → AMP"}
        networks = []
        for name in chain_names:
            ntwk = _get_network(session_id, name)
            if ntwk is None:
                return {"type": "error", "message": f"找不到网络 '{name}'，已加载: {list(ses['networks'].keys())}"}
            networks.append(ntwk)
        try:
            result = sp.cascade_chain(networks)
        except ValueError as e:
            return {"type": "error", "message": str(e)}
        result_name = op.result_name or "_".join(chain_names)
        tmp_path = tempfile.mktemp(suffix=f".s{result.nports}p")
        result.write_touchstone(tmp_path)
        _store_network(session_id, result_name, result, tmp_path)
        return {
            "type": "text",
            "message": f"🔗 链式级联完成: {' → '.join(chain_names)} → **{result_name}** ({result.nports}端口)",
            "ntwk_name": result_name,
        }

    # ── LOAD_BATCH (glob 模式) ──
    if op.action == "load_batch":
        pattern = op.batch_pattern or op.target
        if not pattern:
            return {"type": "error", "message": "请提供文件匹配模式，例如 'data/*.s2p'"}
        # 如果 pattern 是目录，走文件夹扫描
        if os.path.isdir(pattern):
            return _load_dir_to_session(pattern, session_id, ses)
        import glob as globmod
        matches = globmod.glob(pattern, recursive=True)
        if not matches:
            matches = globmod.glob(f"**/{pattern}", recursive=True)
        if not matches:
            return {"type": "error", "message": f"未找到匹配 '{pattern}' 的文件"}
        loaded = []
        for path in matches:
            try:
                info = _register_path(session_id, path, name=os.path.splitext(os.path.basename(path))[0], dedup=True)
                loaded.append(info["name"])
            except Exception as e:
                pass
        return {
            "type": "text",
            "message": f"📂 批量加载完成: {len(loaded)} 个文件\n  " + ", ".join(loaded),
            "ntwk_name": loaded[-1] if loaded else None,
        }

    # ── COMPARE ──
    if op.action == "compare":
        compare_names = op.compare_networks if op.compare_networks else []
        if not compare_names:
            # 从 target 和其他字段推断
            compare_names = [n for n in [op.target, op.cascade_with] if n]
            if not compare_names:
                compare_names = list(ses["networks"].keys())[:10]
        if len(compare_names) < 2:
            return {"type": "error", "message": "对比至少需要 2 个网络"}
        networks = []
        names = []
        for name in compare_names:
            ntwk = _get_network(session_id, name)
            if ntwk is None:
                continue
            networks.append(ntwk)
            names.append(name)
        if len(networks) < 2:
            return {"type": "error", "message": f"至少需要 2 个已加载的网络，当前找到 {len(networks)} 个"}
        try:
            interpolated = sp.interpolate_to_common_freq(networks, npoints=max(len(n.f) for n in networks))
        except ValueError as e:
            return {"type": "error", "message": f"网络频率范围不兼容: {e}"}
        params = op.params if op.params else ["S21"]
        parsed_params = []
        for ps in params:
            _, m, n = sp.parse_network_param(ps, min(n.nports for n in networks), allowed_prefixes=("S",))
            parsed_params.append((m, n))
        if not parsed_params:
            parsed_params = [(1, 0)]
        ref_idx = 0
        if op.reference and op.reference in names:
            ref_idx = names.index(op.reference)
        chart_type = op.chart_type or "db"
        if chart_type == "smith":
            fig = sp.plot_multi_smith(interpolated, names=names, param=parsed_params[0],
                                      title=op.title or "Smith Comparison")
        elif len(parsed_params) == 1:
            fig = sp.plot_multi_db(interpolated, names=names, param=parsed_params[0],
                                   title=op.title or "Multi-File Comparison",
                                   show_diff=op.show_diff, reference_idx=ref_idx)
        else:
            fig = sp.plot_compare(interpolated, names=names, params=parsed_params,
                                  title=op.title or "Multi-Parameter Comparison",
                                  show_diff=op.show_diff, reference_idx=ref_idx)
        chart_json = json.loads(json.dumps(
            {"data": fig.data, "layout": fig.layout},
            cls=plotly.utils.PlotlyJSONEncoder,
        ))
        return {
            "type": "chart",
            "chart": chart_json,
            "title": op.title or f"{', '.join(names)} Comparison",
            "ntwk_name": names[0],
        }

    return {"type": "error", "message": f"不支持的操作: {op.action}"}


def _dispatch_plot(ntwk, params, chart_type: str, title: str, name: str):
    """根据 chart_type 分发到对应的画图函数。"""
    p = params if params else None
    basename = _basename(name)
    if chart_type == "deg":
        return sp.plot_s_deg(ntwk, p, title=title or f"{basename} Phase")
    elif chart_type == "smith":
        return sp.plot_s_smith(ntwk, p, title=title or f"{basename} Smith")
    elif chart_type == "groupdelay":
        return sp.plot_group_delay(ntwk, p, title=title or f"{basename} Group Delay")
    elif chart_type == "mag":
        # 复用 db 图但用线性幅度
        return sp.plot_s_db(ntwk, p, title=title or f"{basename} Magnitude")
    else:
        return sp.plot_s_db(ntwk, p, title=title or f"{basename} S-Parameters")


def _try_execute_rf_plan(session_id: str, plan: dict, networks_meta: dict, planning_history: list = None):
    """
    Execute the validated subset of RF Plan directly in Python.

    This path is intentionally narrow: trusted RF transforms and exact trace plotting.
    Unsupported steps return None so the code Agent can handle genuinely open-ended work.
    """
    output = plan.get("output") or {}
    if output.get("kind") != "plot":
        return None

    chart_type = (output.get("chart_type") or "db").lower()
    if chart_type not in {"db", "mag", "deg", "smith", "vswr", "zmag", "zreal", "zimag", "groupdelay"}:
        return None

    values = {}
    source_names = {}
    value_modes = {}

    try:
        for step in plan.get("steps") or []:
            sid = step.get("id")
            op = step.get("op")
            inputs = step.get("inputs") or []
            args = step.get("args") or {}

            if op == "select":
                name = inputs[0].split(":", 1)[1]
                ntwk = _get_network(session_id, name)
                if ntwk is None:
                    return _rf_plan_error(f"找不到网络 '{name}'，无法执行 RF Plan", plan, planning_history)
                values[sid] = ntwk
                source_names[sid] = name
                value_modes[sid] = "single_ended"
                continue

            if op not in {"slice_freq", "renormalize", "mixed_mode"}:
                return None

            ref = _single_step_input(inputs)
            if not ref or ref not in values:
                return _rf_plan_error(f"RF Plan step {sid} 输入无效", plan, planning_history)

            ntwk = values[ref]
            source_names[sid] = source_names.get(ref, "")
            value_modes[sid] = value_modes.get(ref, "single_ended")

            if op == "slice_freq":
                freq_range = _plan_freq_range(args)
                if not freq_range:
                    return _rf_plan_error(f"RF Plan step {sid} 缺少频率范围", plan, planning_history)
                values[sid] = sp.slice_freq(ntwk, freq_range[0], freq_range[1])
            elif op == "renormalize":
                z0 = _first_present(args, "z0", "z0_ohm", "reference_impedance", "impedance")
                if z0 is None:
                    return _rf_plan_error(f"RF Plan step {sid} 缺少 z0", plan, planning_history)
                values[sid] = sp.renormalize(ntwk, float(z0))
            elif op == "mixed_mode":
                name = source_names.get(ref, "")
                mixed = (networks_meta.get(name) or {}).get("mixed_mode") or {}
                if mixed.get("status") != "ready":
                    reply = f"网络 '{name}' 的差分 P/N 映射未确认，不能生成权威混模结果。"
                    return {
                        "reply": reply,
                        "results": [{"type": "error", "message": reply}],
                        "mode": "agent_plan",
                        "needs_confirmation": True,
                        "plan": plan,
                        "planning_history": planning_history or [],
                    }
                values[sid] = _convert_to_mixed_mode(ntwk, mixed)
                value_modes[sid] = "mixed_mode"
    except ValueError as e:
        return _rf_plan_error(str(e), plan, planning_history)

    traces = output.get("traces") or []
    if not traces:
        return _rf_plan_error("RF Plan 没有指定任何输出 trace", plan, planning_history)

    try:
        fig_data = []
        for trace in traces:
            source = trace.get("source")
            param = agent_plan.normalize_param(trace.get("param", ""))
            if source not in values or not param:
                return _rf_plan_error("RF Plan trace 引用无效", plan, planning_history)
            name = source_names.get(source, source)
            label = trace.get("label") or name
            if _is_mixed_param(param):
                if value_modes.get(source) != "mixed_mode":
                    mixed_source = _matching_mixed_source(source, source_names, value_modes)
                    if not mixed_source:
                        return _rf_plan_error(f"{param} 需要先执行 mixed_mode step", plan, planning_history)
                    source = mixed_source
                    name = source_names.get(source, source)
                    label = trace.get("label") or name
                fig_data.append(_make_mixed_trace(values[source], param, chart_type, label))
            else:
                fig_data.append(_make_plan_trace(values[source], param, chart_type, label))

        expected = agent_plan.expected_trace_count(plan)
        if expected is not None and len(fig_data) != expected:
            return _rf_plan_error(f"RF Plan 期望 {expected} 条曲线，实际生成 {len(fig_data)} 条", plan, planning_history)

        title = _rf_plan_title(plan, source_names)
        layout_type = "zmag" if chart_type in ("zreal", "zimag") else chart_type
        layout = _make_layout(layout_type, title, {})
        if chart_type == "zreal":
            layout["yaxis"]["title"] = "Real(Z) (ohm)"
            layout["yaxis"].pop("type", None)
        elif chart_type == "zimag":
            layout["yaxis"]["title"] = "Imag(Z) (ohm)"
            layout["yaxis"].pop("type", None)

        chart = json.loads(json.dumps(
            {"data": fig_data, "layout": layout},
            cls=plotly.utils.PlotlyJSONEncoder,
        ))
        return {
            "reply": "✅ RF Plan 已确定性执行\n📊 图表已生成",
            "results": [{"type": "chart", "chart": chart, "title": title}],
            "mode": "agent_plan",
            "deterministic": True,
            "plan": plan,
            "planning_history": planning_history or [],
        }
    except ValueError as e:
        return _rf_plan_error(str(e), plan, planning_history)


def _single_step_input(inputs):
    refs = [item for item in inputs if isinstance(item, str) and not item.startswith("network:")]
    return refs[0] if len(refs) == 1 else None


def _matching_mixed_source(source: str, source_names: dict, value_modes: dict):
    name = source_names.get(source)
    matches = [
        sid for sid, mode in value_modes.items()
        if mode == "mixed_mode" and source_names.get(sid) == name
    ]
    return matches[-1] if len(matches) == 1 else None


def _first_present(mapping: dict, *keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _plan_freq_range(args: dict):
    for key in ("freq_range", "frequency_range", "range"):
        if key not in args:
            continue
        value = args[key]
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return value[0], value[1]
        if isinstance(value, str):
            return value, None
    start = _first_present(args, "start", "start_freq", "f_start", "from")
    stop = _first_present(args, "stop", "stop_freq", "f_stop", "end", "to")
    if start is not None and stop is not None:
        return start, stop
    return None


def _rf_plan_error(message: str, plan: dict, planning_history: list = None):
    reply = f"❌ RF Plan 执行失败: {message}"
    return {
        "reply": reply,
        "results": [{"type": "error", "message": message}],
        "mode": "agent_plan",
        "deterministic": True,
        "plan": plan,
        "planning_history": planning_history or [],
    }


def _convert_to_mixed_mode(ntwk, mixed: dict):
    pair_count = int(mixed.get("pair_count") or 0)
    order = [int(i) for i in (mixed.get("se2gmm_order") or [])]
    if pair_count <= 0:
        raise ValueError("mixed_mode 缺少有效 pair_count")
    if len(order) != ntwk.nports:
        raise ValueError("mixed_mode se2gmm_order 与网络端口数不一致")
    if sorted(order) != list(range(ntwk.nports)):
        raise ValueError("mixed_mode se2gmm_order 不是完整端口排列")
    if pair_count * 2 != ntwk.nports:
        raise ValueError("mixed_mode pair_count 与网络端口数不一致")

    mm = ntwk.copy()
    if order != list(range(len(order))):
        mm.renumber(order, list(range(len(order))))
    mm.se2gmm(p=pair_count)
    return mm


def _is_mixed_param(param: str) -> bool:
    return str(param or "").upper().startswith(("SDD", "SDC", "SCD", "SCC"))


def _mixed_indices(param: str, pair_count: int):
    m = re.fullmatch(r"S(DD|DC|CD|CC)(\d+)_(\d+)", param.upper())
    if not m:
        raise ValueError(f"无法解析混模参数: {param}")
    block, row_text, col_text = m.groups()
    row = int(row_text) - 1
    col = int(col_text) - 1
    if row < 0 or col < 0 or row >= pair_count or col >= pair_count:
        raise ValueError(f"{param} 差分端口越界")
    row_offset = 0 if block[0] == "D" else pair_count
    col_offset = 0 if block[1] == "D" else pair_count
    return row + row_offset, col + col_offset


def _make_mixed_trace(ntwk, param: str, chart_type: str, label: str):
    row, col = _mixed_indices(param, ntwk.nports // 2)
    return _make_s_like_trace(ntwk, row, col, chart_type, label, param)


def _make_plan_trace(ntwk, param: str, chart_type: str, label: str):
    param = agent_plan.normalize_param(param)
    if param.startswith("VSWR"):
        port = sp.parse_vswr_param(param, ntwk.nports)
        return _make_vswr_trace(ntwk, port, label, param)

    prefix, row, col = sp.parse_network_param(param, ntwk.nports, allowed_prefixes=("S", "Z", "Y"))
    if prefix == "S":
        return _make_s_like_trace(ntwk, row, col, chart_type, label, param)
    if prefix == "Z":
        return _make_z_trace(ntwk, row, col, chart_type, label, param)
    raise ValueError("确定性执行暂不支持 Y 参数图表")


def _make_s_like_trace(ntwk, row: int, col: int, chart_type: str, label: str, param: str):
    if chart_type == "deg":
        return _make_deg_trace(ntwk, row, col, label, param)
    if chart_type == "smith":
        return _make_smith_trace(ntwk, row, col, label, param)
    if chart_type == "groupdelay":
        return _make_groupdelay_trace(ntwk, row, col, label, param)
    if chart_type == "mag":
        return _make_mag_trace(ntwk, row, col, label, param)
    if chart_type == "vswr":
        return _make_vswr_trace(ntwk, row, label, param)
    if chart_type in {"db", "zmag", "zreal", "zimag"}:
        return _make_db_trace(ntwk, row, col, label, param)
    raise ValueError(f"不支持的图表类型: {chart_type}")


def _make_z_trace(ntwk, row: int, col: int, chart_type: str, label: str, param: str):
    freq = ntwk.f / 1e9
    z = ntwk.z[:, row, col]
    if chart_type == "zreal":
        y = np.real(z)
        unit = "Re(Z)"
    elif chart_type == "zimag":
        y = np.imag(z)
        unit = "Im(Z)"
    else:
        y = np.abs(z)
        unit = "|Z|"
    return {
        "x": freq.tolist(),
        "y": np.asarray(y).tolist(),
        "type": "scatter",
        "mode": "lines",
        "name": f"{label} {param}",
        "hovertemplate": f"<b>{label} {param}</b><br>%{{x:.4f}} GHz<br>{unit}: %{{y:.4f}} Ω<extra></extra>",
    }


def _rf_plan_title(plan: dict, source_names: dict) -> str:
    params = [t.get("param", "") for t in (plan.get("output") or {}).get("traces") or []]
    names = list(dict.fromkeys(n for n in source_names.values() if n))
    prefix = ", ".join(names) if names else "RF Plan"
    return f"{prefix} {'/'.join(params)}"


def _basename(path_or_name: str) -> str:
    return os.path.splitext(os.path.basename(path_or_name))[0]


def _session(session_id: str) -> dict:
    return sessions.setdefault(session_id, {"networks": {}})


def _param_names(nports: int) -> list:
    return [f"S{m+1}_{n+1}" for m in range(nports) for n in range(nports)]


def _entry_response(name: str, entry: dict) -> dict:
    response = {
        "ok": True,
        "name": name,
        "nports": entry.get("nports", 0),
        "f_min": float(entry.get("f_min", 0.0)),
        "f_max": float(entry.get("f_max", 0.0)),
        "f_unit": entry.get("freq_unit", "ghz"),
        "npoints": entry.get("npoints", 0),
        "loaded": entry.get("_ntwk") is not None,
        "network_kind": entry.get("network_kind", "unknown"),
        "port_names": entry.get("port_names", []),
        "port_pairs": entry.get("port_pairs", []),
        "mixed_mode": entry.get("mixed_mode", {}),
        "quick_actions": entry.get("quick_actions", []),
    }
    params = entry.get("params", [])
    response["params"] = params if len(params) <= 200 else params[:200]
    response["params_truncated"] = len(params) > 200
    response["total_params"] = len(params)
    return response


def _agent_network_payload(entry: dict) -> dict:
    """Return JSON-safe network context for the code-generation Agent."""
    ntwk = entry.get("_ntwk")
    if ntwk is not None and (
        "network_kind" not in entry or "port_names" not in entry or "quick_actions" not in entry
    ):
        entry.update(_inspect_network(ntwk))

    params = list(entry.get("params") or [])
    if ntwk is not None and not params:
        params = sp.list_params(ntwk)
    path = entry.get("path") or ""
    f_min = entry.get("f_min")
    f_max = entry.get("f_max")
    npoints = entry.get("npoints")
    if ntwk is not None:
        if f_min is None and len(ntwk.f):
            f_min = ntwk.f[0]
        if f_max is None and len(ntwk.f):
            f_max = ntwk.f[-1]
        if npoints is None:
            npoints = len(ntwk.f)
    payload = {
        "path": os.path.abspath(path) if path else "",
        "nports": int(entry.get("nports") or getattr(ntwk, "nports", 0) or 0),
        "f_min": float(f_min or 0.0),
        "f_max": float(f_max or 0.0),
        "npoints": int(npoints or 0),
        "loaded": ntwk is not None,
        "network_kind": entry.get("network_kind", "unknown"),
        "port_names": list(entry.get("port_names") or []),
        "port_pairs": list(entry.get("port_pairs") or []),
        "mixed_mode": entry.get("mixed_mode") or {},
        "quick_actions": list(entry.get("quick_actions") or []),
        "params": params[:80],
        "total_params": len(params),
    }

    if ntwk is not None:
        try:
            z0 = np.asarray(ntwk.z0)
            if z0.size:
                payload["z0"] = float(np.real(z0.flat[0]))
        except Exception:
            pass

    return payload


def _entry_from_network(path: str, ntwk) -> dict:
    return {
        "path": path,
        "_ntwk": ntwk,
        "nports": ntwk.nports,
        "f_min": float(ntwk.f[0]),
        "f_max": float(ntwk.f[-1]),
        "npoints": len(ntwk.f),
        "params": sp.list_params(ntwk),
        **_inspect_network(ntwk),
    }


def _entry_from_header(path: str, header: dict) -> dict:
    nports = int(header["nports"])
    return {
        "path": path,
        "_ntwk": None,
        "nports": nports,
        "f_min": float(header["f_min"]),
        "f_max": float(header["f_max"]),
        "npoints": int(header["npoints"]),
        "freq_unit": header.get("freq_unit", "ghz"),
        "params": _param_names(nports),
        "network_kind": "unknown",
        "port_names": [f"Port{i + 1}" for i in range(nports)],
        "port_pairs": [],
        "mixed_mode": {
            "status": "unsupported",
            "source": "header",
            "pair_count": 0,
            "differential_pairs": [],
            "se2gmm_order": [],
            "through_paths": [],
            "confidence": {"endpoint_pairing": "none", "polarity": "unknown"},
            "message": "metadata-only network is not inspected yet",
        },
        "quick_actions": [],
    }


def _should_lazy_load(path: str, header: dict) -> bool:
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    return int(header.get("nports", 0)) >= 8 or size > EAGER_LOAD_MAX_BYTES


def _register_path(session_id: str, path: str, name: str = None, dedup: bool = False) -> dict:
    ses = _session(session_id)
    networks = ses["networks"]
    name = name or os.path.splitext(os.path.basename(path))[0]
    if dedup:
        name = _dedup_name(networks, name)

    header = _read_touchstone_header(path)
    if header and _should_lazy_load(path, header):
        entry = _entry_from_header(path, header)
    else:
        ntwk = rf.Network(path)
        entry = _entry_from_network(path, ntwk)
        ses["freq_unit"] = _guess_freq_unit(ntwk)

    networks[name] = entry
    return _entry_response(name, entry)


def _store_network(session_id: str, name: str, ntwk, path: str, dedup: bool = False) -> dict:
    ses = _session(session_id)
    if dedup:
        name = _dedup_name(ses["networks"], name)
    entry = _entry_from_network(path, ntwk)
    ses["networks"][name] = entry
    return _entry_response(name, entry)


# ──────────────────────────────────────────────────────────────
#  文件上传 API
# ──────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload():
    """上传 Touchstone 文件"""
    if "file" not in request.files:
        return jsonify({"error": "没有选择文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    # 保存临时文件
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        name = Path(file.filename).stem
        session_id = request.form.get("session", "default")
        return jsonify(_register_path(session_id, tmp_path, name=name))
    except Exception as e:
        os.unlink(tmp_path)
        traceback.print_exc()
        return jsonify({"error": f"解析失败: {str(e)}"}), 400


@app.route("/api/upload/batch", methods=["POST"])
def upload_batch():
    """批量上传多个 Touchstone 文件。"""
    if "files" not in request.files:
        return jsonify({"error": "没有选择文件"}), 400

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "文件列表为空"}), 400

    session_id = request.form.get("session", "default")
    if session_id not in sessions:
        sessions[session_id] = {"networks": {}}

    results = []
    for file in files:
        if file.filename == "":
            continue
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        try:
            results.append(_register_path(session_id, tmp_path, name=Path(file.filename).stem, dedup=True))
        except Exception as e:
            os.unlink(tmp_path)
            results.append({"ok": False, "name": file.filename, "error": str(e)})

    return jsonify({
        "ok": True,
        "loaded": [r for r in results if r.get("ok")],
        "failed": [r for r in results if not r.get("ok")],
        "total": len(results),
    })


@app.route("/api/load-local", methods=["POST"])
def load_local():
    """
    直接从本地路径加载文件（跳过上传，适合大文件）。
    支持单个文件路径和文件夹路径（自动扫描所有 .sNp）。
    接收 JSON: { "path": "/data/big.s64p", "session": "default" }
               { "path": "/data/folder",   "session": "default" }
    """
    data = request.get_json()
    local_path = data.get("path", "").strip()
    session_id = data.get("session", "default")

    if not local_path:
        return jsonify({"error": "请提供本地文件路径"}), 400
    if not os.path.exists(local_path):
        return jsonify({"error": f"路径不存在: {local_path}"}), 404

    if os.path.isdir(local_path):
        return _load_local_dir(local_path, session_id)

    return _load_local_file(local_path, session_id, data.get("name"))


def _load_local_dir(dir_path, session_id):
    """扫描文件夹下所有 .sNp 文件并加载。"""
    snp_exts = {f".s{p}p" for p in range(1, 65)} | {".ts"}
    files = sorted(
        f for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f)) and Path(f).suffix.lower() in snp_exts
    )
    if not files:
        return jsonify({"error": f"文件夹中未找到 .sNp 文件: {dir_path}"}), 404

    if session_id not in sessions:
        sessions[session_id] = {"networks": {}}

    results = []
    for fname in files:
        fpath = os.path.join(dir_path, fname)
        try:
            results.append(_register_path(session_id, fpath, name=os.path.splitext(fname)[0], dedup=True))
        except Exception as e:
            results.append({"ok": False, "name": fname, "error": str(e)})

    return jsonify({
        "ok": True,
        "is_dir": True,
        "loaded": [r for r in results if r.get("ok")],
        "failed": [r for r in results if not r.get("ok")],
        "total": len(results),
    })


def _load_local_file(local_path, session_id, name_override=None):
    """加载单个本地 .sNp 文件。"""
    try:
        return jsonify(_register_path(session_id, local_path, name=name_override))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _dedup_name(networks, name, start=2):
    """确保名称在 networks 中不重复，重复时添加数字后缀。"""
    if name not in networks:
        return name
    base = name
    idx = start
    while f"{base}_{idx}" in networks:
        idx += 1
    return f"{base}_{idx}"


def _load_dir_to_session(dir_path, session_id, ses):
    """扫描文件夹下所有 .sNp 文件并加载到会话中（用于 chat 命令）。"""
    snp_exts = {f".s{p}p" for p in range(1, 65)} | {".ts"}
    files = sorted(
        f for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f)) and Path(f).suffix.lower() in snp_exts
    )
    if not files:
        return {"type": "error", "message": f"文件夹中未找到 .sNp 文件: {dir_path}"}

    loaded = []
    for fname in files:
        fpath = os.path.join(dir_path, fname)
        try:
            info = _register_path(session_id, fpath, name=os.path.splitext(fname)[0], dedup=True)
            loaded.append(info["name"])
        except Exception:
            pass
    return {
        "type": "text",
        "message": f"📂 文件夹加载完成: {len(loaded)} 个文件\n  " + ", ".join(loaded),
        "ntwk_name": loaded[-1] if loaded else None,
    }


@app.route("/api/upload/glob", methods=["POST"])
def upload_glob():
    """通过通配符模式批量加载本地文件。
    接收 JSON: { "pattern": "data/*.s2p", "session": "default" }
    """
    data = request.get_json()
    pattern = data.get("pattern", "").strip()
    session_id = data.get("session", "default")

    if not pattern:
        return jsonify({"error": "请提供文件匹配模式，例如 data/*.s2p"}), 400

    import glob as globmod
    matches = globmod.glob(pattern, recursive=True)
    if not matches:
        # 尝试递归搜索
        matches = globmod.glob(f"**/{pattern}", recursive=True)

    if not matches:
        return jsonify({"error": f"未找到匹配 '{pattern}' 的文件"}), 404

    if session_id not in sessions:
        sessions[session_id] = {"networks": {}}

    results = []
    for path in matches:
        try:
            results.append(_register_path(session_id, path, name=os.path.splitext(os.path.basename(path))[0], dedup=True))
        except Exception as e:
            results.append({"ok": False, "name": os.path.basename(path), "error": str(e)})

    return jsonify({
        "ok": True,
        "loaded": [r for r in results if r.get("ok")],
        "failed": [r for r in results if not r.get("ok")],
        "total": len(results),
    })


@app.route("/api/networks", methods=["GET"])
def list_networks():
    """列出已上传的网络（大端口文件参数截断至前200个）"""
    session_id = request.args.get("session", "default")
    if session_id not in sessions:
        return jsonify({"networks": {}})
    nets = {}
    for name, info in sessions[session_id]["networks"].items():
        entry = dict(info)
        if "network_kind" not in entry and entry.get("_ntwk") is not None:
            entry.update(_inspect_network(entry["_ntwk"]))
        # 移除 _ntwk 对象（不可序列化），替换为状态标志
        entry["loaded"] = "_ntwk" in info and info["_ntwk"] is not None
        entry.pop("_ntwk", None)
        if len(entry.get("params", [])) > 200:
            entry["params_truncated"] = True
            entry["total_params"] = len(entry["params"])
            entry["params"] = entry["params"][:200]
        nets[name] = entry
    return jsonify({"networks": nets})


@app.route("/api/networks/<name>/load", methods=["POST"])
def load_network(name):
    """触发网络延迟加载（后台线程，立即返回）。"""
    session_id = request.args.get("session", "default")
    if session_id not in sessions:
        return jsonify({"ok": False, "error": "会话不存在"}), 404
    nets = sessions[session_id].get("networks", {})
    if name not in nets:
        return jsonify({"ok": False, "error": f"找不到网络 '{name}'"}), 404

    entry = nets[name]
    # 已加载 → 直接返回
    if entry.get("_ntwk") is not None:
        return jsonify({"ok": True, "name": name, "status": "loaded",
                        "nports": entry["_ntwk"].nports, "npoints": len(entry["_ntwk"].f)})

    # 正在加载 → 返回 loading 状态
    if entry.get("_loading"):
        return jsonify({"ok": True, "name": name, "status": "loading"})

    # 启动后台线程加载
    entry["_loading"] = True
    import threading
    def _load_worker():
        try:
            path = entry["path"]
            if os.path.exists(path):
                ntwk = rf.Network(path)
                entry["_ntwk"] = ntwk
        except Exception as e:
            entry["_load_error"] = str(e)
        finally:
            entry["_loading"] = False

    t = threading.Thread(target=_load_worker, daemon=True)
    t.start()

    return jsonify({"ok": True, "name": name, "status": "loading"})


@app.route("/api/networks/<name>/status", methods=["GET"])
def network_status(name):
    """查询网络是否已加载到内存。"""
    session_id = request.args.get("session", "default")
    if session_id not in sessions:
        return jsonify({"loaded": False, "error": "会话不存在"})
    nets = sessions[session_id].get("networks", {})
    if name not in nets:
        return jsonify({"loaded": False, "error": "网络不存在"})
    entry = nets[name]
    return jsonify({
        "loaded": entry.get("_ntwk") is not None,
        "loading": entry.get("_loading", False),
        "error": entry.get("_load_error", ""),
        "name": name,
        "nports": entry.get("nports", 0),
        "npoints": entry.get("npoints", 0),
    })


@app.route("/api/networks/<name>", methods=["DELETE"])
def remove_network(name):
    """移除已上传的网络"""
    session_id = request.args.get("session", "default")
    if session_id in sessions and name in sessions[session_id]["networks"]:
        info = sessions[session_id]["networks"][name]
        # 只删除临时文件（位于系统 temp 目录），不删用户原始文件
        fpath = info.get("path", "")
        if fpath and os.path.exists(fpath):
            try:
                _tmp = tempfile.gettempdir()
                if os.path.normpath(fpath).startswith(os.path.normpath(_tmp)):
                    os.unlink(fpath)
            except Exception:
                pass
        del sessions[session_id]["networks"][name]
    return jsonify({"ok": True})


# ──────────────────────────────────────────────────────────────
#  图表生成 API
# ──────────────────────────────────────────────────────────────

@app.route("/api/chart", methods=["POST"])
def generate_chart():
    """生成交互式 Plotly 图表，返回 JSON（plotly.js 可直接渲染）"""
    data = request.get_json()
    session_id = data.get("session", "default")
    chart_type = data.get("type", "db")
    networks = data.get("networks", [])     # [{name, params: ["S11",...], label}]
    freq_range = data.get("freq_range")     # [start, stop]  or null
    title = data.get("title", "")
    options = data.get("options", {})       # {smith_type, show_grid, ...}

    if session_id not in sessions:
        return jsonify({"error": "没有上传文件"}), 400

    try:
        fig_data = []

        for entry in networks:
            name = entry["name"]
            params = entry.get("params", [])
            label = entry.get("label", name)

            ntwk = _get_network(session_id, name)
            if ntwk is None:
                continue
            if freq_range:
                ntwk = sp.slice_freq(ntwk, freq_range[0], freq_range[1])

            for p in params:
                p_name = str(p).strip().upper()

                if p_name.startswith("VSWR") or chart_type == "vswr":
                    if p_name.startswith("VSWR"):
                        m = sp.parse_vswr_param(p_name, ntwk.nports)
                    else:
                        _, m, _ = sp.parse_network_param(p_name, ntwk.nports, allowed_prefixes=("S",))
                    trace = _make_vswr_trace(ntwk, m, label, p_name)
                else:
                    prefix, m, n = sp.parse_network_param(p_name, ntwk.nports, allowed_prefixes=("S", "Z", "Y"))
                    if prefix == "Y":
                        return jsonify({"error": "暂不支持 Y 参数图表"}), 400
                    if prefix == "Z":
                        trace = _make_zmag_trace(ntwk, m, n, label, p_name)
                    elif chart_type == "db":
                        trace = _make_db_trace(ntwk, m, n, label, p_name)
                    elif chart_type == "deg":
                        trace = _make_deg_trace(ntwk, m, n, label, p_name)
                    elif chart_type == "smith":
                        trace = _make_smith_trace(ntwk, m, n, label, p_name)
                    elif chart_type == "groupdelay":
                        trace = _make_groupdelay_trace(ntwk, m, n, label, p_name)
                    elif chart_type == "mag":
                        trace = _make_mag_trace(ntwk, m, n, label, p_name)
                    elif chart_type == "zmag":
                        trace = _make_zmag_trace(ntwk, m, n, label, p_name)
                    else:
                        continue
                fig_data.append(trace)

        if not fig_data:
            return jsonify({"error": "没有可绘制的数据"}), 400

        layout = _make_layout(chart_type, title, options)
        fig = {"data": fig_data, "layout": layout}

        return jsonify(json.loads(
            json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        ))

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart/html", methods=["POST"])
def generate_chart_html():
    """生成独立 HTML 文件（可离线打开，自包含 plotly.js CDN）"""
    data = request.get_json()
    session_id = data.get("session", "default")
    chart_type = data.get("type", "db")
    title = data.get("title", "S-Parameter Chart")

    # 先拿到 JSON
    resp = generate_chart()
    response, status = _normalize_view_response(resp)
    if status != 200:
        return resp
    fig_json = response.get_json()
    safe_title = html.escape(str(title), quote=True)
    data_json = _json_for_html_script(fig_json["data"])
    layout_json = _json_for_html_script(fig_json["layout"])

    page_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{safe_title}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ margin: 0; padding: 20px; background: #1a1a2e; font-family: -apple-system, sans-serif; }}
  #chart {{ width: 100%; height: 85vh; }}
  h2 {{ color: #e0e0e0; text-align: center; }}
</style>
</head>
<body>
<h2>{safe_title}</h2>
<div id="chart"></div>
<script>
  Plotly.newPlot('chart', {data_json}, {layout_json}, {{ responsive: true }});
</script>
</body>
</html>"""

    return jsonify({"html": page_html, "title": title})


@app.route("/api/export/html", methods=["POST"])
def export_html():
    """下载独立 HTML 报告"""
    data = request.get_json()
    resp = generate_chart_html()
    response, status = _normalize_view_response(resp)
    if status != 200:
        return resp
    html = response.get_json()["html"]

    bio = io.BytesIO()
    bio.write(html.encode("utf-8"))
    bio.seek(0)
    return send_file(bio, mimetype="text/html", as_attachment=True,
                     download_name="s_param_report.html")


# ──────────────────────────────────────────────────────────────
#  数据导出 API
# ──────────────────────────────────────────────────────────────

@app.route("/api/export/csv", methods=["POST"])
def export_csv():
    """导出 S 参数为 CSV"""
    data = request.get_json()
    session_id = data.get("session", "default")
    network_name = data["network"]
    params = data.get("params", [])

    ntwk = _get_network(session_id, network_name)
    if ntwk is None:
        return jsonify({"error": "网络不存在"}), 404

    freq_range = data.get("freq_range")
    if freq_range:
        try:
            ntwk = sp.slice_freq(ntwk, freq_range[0], freq_range[1])
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    bio = io.BytesIO()
    sp.export_csv(ntwk, params, bio)
    bio.seek(0)
    return send_file(bio, mimetype="text/csv", as_attachment=True,
                     download_name=f"{network_name}_export.csv")


@app.route("/api/cascade/chain", methods=["POST"])
def cascade_chain_api():
    """
    链式级联多个网络。
    接收 JSON: { "session": "default", "chain": ["LNA", "BPF", "AMP"], "result_name": "LNA_BPF_AMP" }
    返回级联后的网络信息。
    """
    data = request.get_json()
    session_id = data.get("session", "default")
    chain_names = data.get("chain", [])
    result_name = data.get("result_name", "_".join(chain_names))

    if len(chain_names) < 2:
        return jsonify({"error": "级联需要至少 2 个网络名称"}), 400

    ses = sessions.get(session_id, {}).get("networks", {})
    networks = []
    for name in chain_names:
        ntwk = _get_network(session_id, name)
        if ntwk is None:
            return jsonify({"error": f"找不到网络 '{name}'，已加载: {list(ses.keys())}"}), 404
        networks.append(ntwk)

    try:
        result = sp.cascade_chain(networks)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"级联失败: {e}"}), 500

    # 保存到 session
    tmp_path = tempfile.mktemp(suffix=f".s{result.nports}p")
    result.write_touchstone(tmp_path)
    info = _store_network(session_id, result_name, result, tmp_path)

    return jsonify({
        "ok": True,
        "name": result_name,
        "chain": chain_names,
        **{k: v for k, v in info.items() if k not in ("ok", "name")},
    })


@app.route("/api/compare", methods=["POST"])
def compare_networks():
    """
    多文件对比视图。
    接收 JSON: {
        "session": "default",
        "networks": ["LNA", "BPF", "AMP"],
        "params": ["S21"],
        "chart_type": "db",
        "show_diff": true,
        "reference": "LNA",
        "title": "对比标题"
    }
    返回 Plotly JSON 图表 + 差异统计
    """
    data = request.get_json()
    session_id = data.get("session", "default")
    network_names = data.get("networks", [])
    param_strs = data.get("params", ["S21"])
    chart_type = data.get("chart_type", "db")
    show_diff = data.get("show_diff", True)
    ref_name = data.get("reference", "")
    title = data.get("title", "Multi-File Comparison")
    freq_range = data.get("freq_range")

    if len(network_names) < 2:
        return jsonify({"error": "对比至少需要 2 个网络"}), 400

    # 收集网络对象
    networks = []
    names = []
    for name in network_names:
        ntwk = _get_network(session_id, name)
        if ntwk is None:
            return jsonify({"error": f"找不到网络 '{name}'"}), 404
        if freq_range:
            try:
                ntwk = sp.slice_freq(ntwk, freq_range[0], freq_range[1])
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
        networks.append(ntwk)
        names.append(name)

    # 找参考索引
    reference_idx = 0
    if ref_name and ref_name in names:
        reference_idx = names.index(ref_name)

    # 解析参数
    try:
        params = []
        for ps in param_strs:
            _, m, n = sp.parse_network_param(ps, min(nw.nports for nw in networks), allowed_prefixes=("S",))
            params.append((m, n))
        if not params:
            params = [(1, 0)]  # 默认 S21
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        # 插值到共同频率
        interpolated = sp.interpolate_to_common_freq(
            networks, npoints=max(len(n.f) for n in networks)
        )

        if chart_type == "smith":
            fig = sp.plot_multi_smith(interpolated, names=names, param=params[0], title=title)
        elif len(params) == 1 and chart_type == "db":
            fig = sp.plot_multi_db(
                interpolated, names=names, param=params[0],
                title=title, show_diff=show_diff, reference_idx=reference_idx,
            )
        else:
            fig = sp.plot_compare(
                interpolated, names=names, params=params,
                title=title, show_diff=show_diff, reference_idx=reference_idx,
            )

        # 计算差异统计
        stats = None
        if show_diff:
            try:
                stats = sp.compute_diff_stats(interpolated, params[0], reference_idx)
            except Exception:
                pass

        chart_json = json.loads(json.dumps(
            {"data": fig.data, "layout": fig.layout},
            cls=plotly.utils.PlotlyJSONEncoder,
        ))

        return jsonify({
            "ok": True,
            "chart": chart_json,
            "title": title,
            "diff_stats": stats,
        })

    except ValueError as e:
        return jsonify({"error": f"频率范围不兼容: {e}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/export/touchstone", methods=["POST"])
def export_touchstone():
    """导出为 Touchstone 文件"""
    data = request.get_json()
    session_id = data.get("session", "default")
    network_name = data["network"]

    ntwk = _get_network(session_id, network_name)
    if ntwk is None:
        return jsonify({"error": "网络不存在"}), 404

    freq_range = data.get("freq_range")
    if freq_range:
        try:
            ntwk = sp.slice_freq(ntwk, freq_range[0], freq_range[1])
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    bio = io.BytesIO()
    sp.save_touchstone(ntwk, bio)
    bio.seek(0)
    ext = f"s{ntwk.nports}p"
    return send_file(bio, mimetype="text/plain", as_attachment=True,
                     download_name=f"{network_name}_export.{ext}")


# ──────────────────────────────────────────────────────────────
#  级联 / 处理 API
# ──────────────────────────────────────────────────────────────

@app.route("/api/cascade", methods=["POST"])
def cascade():
    """级联两个网络"""
    data = request.get_json()
    session_id = data.get("session", "default")
    name_a = data["network_a"]
    name_b = data["network_b"]
    result_name = data.get("result_name", f"{name_a}_{name_b}_cascaded")

    ntwk_a = _get_network(session_id, name_a)
    ntwk_b = _get_network(session_id, name_b)
    if ntwk_a is None or ntwk_b is None:
        return jsonify({"error": "网络不存在"}), 404

    ntwk_result = sp.cascade(ntwk_a, ntwk_b)

    # 保存到临时文件
    tmp_path = tempfile.mktemp(suffix=f".s{ntwk_result.nports}p")
    ntwk_result.write_touchstone(tmp_path)
    return jsonify(_store_network(session_id, result_name, ntwk_result, tmp_path))


@app.route("/api/deembed", methods=["POST"])
def deembed():
    """去嵌"""
    data = request.get_json()
    session_id = data.get("session", "default")
    name_dut = data["network_dut"]
    name_fixture = data["network_fixture"]

    ntwk_dut = _get_network(session_id, name_dut)
    ntwk_fixture = _get_network(session_id, name_fixture)

    if ntwk_dut is None or ntwk_fixture is None:
        return jsonify({"error": "网络不存在"}), 404

    ntwk_result = sp.deembed(ntwk_dut, ntwk_fixture)
    result_name = f"{name_dut}_deembedded"

    tmp_path = tempfile.mktemp(suffix=f".s{ntwk_result.nports}p")
    ntwk_result.write_touchstone(tmp_path)
    _store_network(session_id, result_name, ntwk_result, tmp_path)

    return jsonify({"ok": True, "name": result_name})


# ──────────────────────────────────────────────────────────────
#  内部辅助
# ──────────────────────────────────────────────────────────────

def _get_network(session_id, name):
    """从会话中获取网络对象（优先内存缓存，fallback 磁盘）。"""
    if session_id not in sessions:
        return None
    nets = sessions[session_id]["networks"]

    def _resolve(key):
        if key in nets:
            entry = nets[key]
            # 优先返回内存中的对象
            if "_ntwk" in entry and entry["_ntwk"] is not None:
                return entry["_ntwk"]
            # fallback: 从磁盘加载并缓存
            if "path" in entry and os.path.exists(entry["path"]):
                ntwk = rf.Network(entry["path"])
                entry["_ntwk"] = ntwk  # 缓存到内存
                return ntwk
        return None

    # 精确匹配
    result = _resolve(name)
    if result is not None:
        return result
    # basename 匹配（去掉路径和扩展名）
    base = os.path.splitext(os.path.basename(name))[0] if name else ""
    if base:
        result = _resolve(base)
        if result is not None:
            return result
        # 模糊匹配
        for k in nets:
            if k.lower() == base.lower() or base.lower() in k.lower() or k.lower() in base.lower():
                result = _resolve(k)
                if result is not None:
                    return result
    return None


def _normalize_view_response(resp):
    """Return (response, status_code) for direct or tuple Flask view returns."""
    if isinstance(resp, tuple):
        response = resp[0]
        status = resp[1] if len(resp) > 1 else response.status_code
        return response, status
    return resp, resp.status_code


def _json_for_html_script(value) -> str:
    """JSON safe to embed in a script tag."""
    return (
        json.dumps(value, cls=plotly.utils.PlotlyJSONEncoder)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _inspect_network(ntwk):
    try:
        return network_inspector.inspect_network(ntwk)
    except Exception:
        return {
            "network_kind": "unknown",
            "port_names": [],
            "port_pairs": [],
            "mixed_mode": {},
            "quick_actions": [],
        }


def _read_touchstone_header(path: str) -> dict:
    """
    快速读取 Touchstone 文件头，不解析完整 S 矩阵。
    适用于大端口文件（.s64p 等），毫秒级返回。
    自动跳过任意长度的注释头（仿真软件可能导出上百行 ! 注释）。
    """
    import re as _re
    data_re = _re.compile(r"^\s*-?\d")
    ext_match = _re.match(r"\.s(\d+)p$", Path(path).suffix.lower())

    # ── 第一遍：扫描找到 # 行和第一个数据行 ──
    freq_unit = "ghz"
    nports = int(ext_match.group(1)) if ext_match else 0
    f_min = None
    first_data_line = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            # 跳过空行和注释
            if not stripped or stripped.startswith("!"):
                continue
            # # 行：频率单位等元数据
            if stripped.startswith("#"):
                parts = stripped.split()
                freq_str = parts[1].lower() if len(parts) > 1 else "ghz"
                if "ghz" in freq_str: freq_unit = "ghz"
                elif "mhz" in freq_str: freq_unit = "mhz"
                elif "khz" in freq_str: freq_unit = "khz"
                elif "hz" in freq_str: freq_unit = "hz"
                continue
            # 数据行
            if data_re.match(stripped):
                first_data_line = stripped
                break

    if first_data_line is None:
        return None

    # ── 从第一个数据行推导端口数（扩展名无法判断时） ──
    cols = first_data_line.split()
    nvals = len(cols) - 1  # 减掉频率列
    if nports == 0 and nvals > 0:
        for cols_per_param in [2, 1]:  # RI/MA(2列) 或 DB(1列)
            n2 = nvals // cols_per_param
            n = int(n2 ** 0.5)
            if n * n == n2 and n > 0:
                nports = n
                break

    if nports == 0:
        return None

    # ── 频率范围 ──
    freq_mul = {"ghz": 1e9, "mhz": 1e6, "khz": 1e3, "hz": 1.0}.get(freq_unit, 1e9)
    try:
        f_min = float(cols[0]) * freq_mul
    except Exception:
        f_min = 0.0

    # ── 第二遍：统计总数据行数，取最后一行频率 ──
    npoints = 0
    last_freq = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if data_re.match(line):
                npoints += 1
                try:
                    last_freq = float(line.split()[0])
                except Exception:
                    pass

    f_max = (last_freq * freq_mul) if last_freq is not None else f_min

    return {
        "nports": nports,
        "f_min": float(f_min),
        "f_max": float(f_max),
        "npoints": npoints,
        "freq_unit": freq_unit,
    }


def _guess_freq_unit(ntwk):
    f = ntwk.f[0]
    if f > 1e9:
        return "ghz"
    elif f > 1e6:
        return "mhz"
    elif f > 1e3:
        return "khz"
    return "hz"


def _parse_param(p: str, nports: int = None):
    """'S2_1' → (1, 0), 'Z64_64' → (63, 63). Legacy 'S21' also accepted."""
    p = p.strip().upper()
    if p.startswith("VSWR"):
        return sp.parse_vswr_param(p, nports), None
    _, m, n = sp.parse_network_param(p, nports, allowed_prefixes=("S", "Z", "Y"))
    return m, n


def _freq_label(session_id):
    unit = sessions.get(session_id, {}).get("freq_unit", "ghz")
    return f"Frequency ({'GHz' if unit == 'ghz' else 'MHz' if unit == 'mhz' else 'Hz'})"


# ── trace 构建函数 ─────────────────────────────────────────────

def _make_db_trace(ntwk, m, n, label, p_name):
    freq = ntwk.f / 1e9
    return {
        "x": freq.tolist(),
        "y": ntwk.s_db[:, m, n].tolist(),
        "type": "scatter",
        "mode": "lines",
        "name": f"{label} {p_name}",
        "hovertemplate": f"<b>{label} {p_name}</b><br>%{{x:.4f}} GHz<br>%{{y:.3f}} dB<extra></extra>",
    }


def _make_deg_trace(ntwk, m, n, label, p_name):
    freq = ntwk.f / 1e9
    return {
        "x": freq.tolist(),
        "y": ntwk.s_deg[:, m, n].tolist(),
        "type": "scatter",
        "mode": "lines",
        "name": f"{label} {p_name}",
        "hovertemplate": f"<b>{label} {p_name}</b><br>%{{x:.4f}} GHz<br>%{{y:.2f}}°<extra></extra>",
    }


def _make_mag_trace(ntwk, m, n, label, p_name):
    freq = ntwk.f / 1e9
    return {
        "x": freq.tolist(),
        "y": ntwk.s_mag[:, m, n].tolist(),
        "type": "scatter",
        "mode": "lines",
        "name": f"{label} {p_name}",
        "hovertemplate": f"<b>{label} {p_name}</b><br>%{{x:.4f}} GHz<br>Mag: %{{y:.4f}}<extra></extra>",
    }


def _make_zmag_trace(ntwk, m, n, label, p_name):
    freq = ntwk.f / 1e9
    zmag = np.abs(ntwk.z[:, m, n])
    return {
        "x": freq.tolist(),
        "y": zmag.tolist(),
        "type": "scatter",
        "mode": "lines",
        "name": f"{label} {p_name}",
        "hovertemplate": f"<b>{label} {p_name}</b><br>%{{x:.4f}} GHz<br>|Z|: %{{y:.4f}} Ω<extra></extra>",
    }


def _make_smith_trace(ntwk, m, n, label, p_name):
    """Smith chart via scatter on complex plane + reference circles"""
    s = ntwk.s[:, m, n]
    return {
        "x": s.real.tolist(),
        "y": s.imag.tolist(),
        "type": "scatter",
        "mode": "lines+markers",
        "name": f"{label} {p_name}",
        "hovertemplate": f"<b>{label} {p_name}</b><br>Re: %{{x:.4f}}<br>Im: %{{y:.4f}}<extra></extra>",
    }


def _make_vswr_trace(ntwk, m, label, p_name):
    freq = ntwk.f / 1e9
    return {
        "x": freq.tolist(),
        "y": ntwk.s_vswr[:, m, m].tolist(),
        "type": "scatter",
        "mode": "lines",
        "name": f"{label} VSWR{m+1}",
        "hovertemplate": f"<b>{label} VSWR{m+1}</b><br>%{{x:.4f}} GHz<br>VSWR: %{{y:.3f}}<extra></extra>",
    }


def _make_groupdelay_trace(ntwk, m, n, label, p_name):
    freq = ntwk.f / 1e9
    gd = sp.get_group_delay(ntwk, m, n)
    return {
        "x": freq.tolist(),
        "y": gd.tolist(),
        "type": "scatter",
        "mode": "lines",
        "name": f"{label} GD({p_name})",
        "hovertemplate": f"<b>{label} GD({p_name})</b><br>%{{x:.4f}} GHz<br>%{{y:.4f}} ns<extra></extra>",
    }


def _make_layout(chart_type, title, options):
    base = {
        "title": {"text": title or "S-Parameter Chart", "font": {"color": "#e0e0e0"}},
        "paper_bgcolor": "#1a1a2e",
        "plot_bgcolor": "#16213e",
        "font": {"color": "#c0c0c0"},
        "xaxis": {"gridcolor": "#2a2a4a", "zerolinecolor": "#444"},
        "yaxis": {"gridcolor": "#2a2a4a", "zerolinecolor": "#444"},
        "hovermode": "closest",
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
    }

    if chart_type == "smith":
        base["xaxis"]["title"] = "Real (Γ)"
        base["yaxis"]["title"] = "Imag (Γ)"
        base["xaxis"]["scaleanchor"] = "y"
        base["xaxis"]["scaleratio"] = 1
        base["xaxis"]["range"] = [-1.1, 1.1]
        base["yaxis"]["range"] = [-1.1, 1.1]
        # 添加 Smith 参考圆
        base["shapes"] = _smith_circles()
    elif chart_type == "db":
        base["xaxis"]["title"] = "Frequency (GHz)"
        base["yaxis"]["title"] = "Magnitude (dB)"
    elif chart_type == "deg":
        base["xaxis"]["title"] = "Frequency (GHz)"
        base["yaxis"]["title"] = "Phase (°)"
    elif chart_type == "vswr":
        base["xaxis"]["title"] = "Frequency (GHz)"
        base["yaxis"]["title"] = "VSWR"
    elif chart_type == "groupdelay":
        base["xaxis"]["title"] = "Frequency (GHz)"
        base["yaxis"]["title"] = "Group Delay (ns)"
    elif chart_type == "zmag":
        base["xaxis"]["title"] = "Frequency (GHz)"
        base["xaxis"]["type"] = "log"
        base["yaxis"]["title"] = "Magnitude |Z| (ohm)"
        base["yaxis"]["type"] = "log"

    # 覆盖用户选项
    if options.get("title"):
        base["title"]["text"] = options["title"]
    if options.get("xlabel"):
        base["xaxis"]["title"] = options["xlabel"]
    if options.get("ylabel"):
        base["yaxis"]["title"] = options["ylabel"]

    return base


def _smith_circles():
    """生成 Smith 圆图参考圆（简化版：单位圆 + 几个 r/x 参考圆）"""
    shapes = []
    # 单位圆
    shapes.append({
        "type": "circle",
        "xref": "x", "yref": "y",
        "x0": -1, "y0": -1, "x1": 1, "y1": 1,
        "line": {"color": "#555", "width": 1, "dash": "dash"},
    })
    # 电阻圆 r = 0.2, 0.5, 1, 2
    for r in [0.2, 0.5, 1.0, 2.0]:
        cx = r / (r + 1)
        rad = 1 / (r + 1)
        shapes.append({
            "type": "circle",
            "xref": "x", "yref": "y",
            "x0": cx - rad, "y0": -rad, "x1": cx + rad, "y1": rad,
            "line": {"color": "#444", "width": 0.5},
        })
    # 电抗弧 x = ±0.5, ±1, ±2
    for x in [0.5, 1.0, 2.0, -0.5, -1.0, -2.0]:
        cy = 1 / x if x != 0 else 1000
        rad = abs(1 / x) if x != 0 else 1000
        shapes.append({
            "type": "circle",
            "xref": "x", "yref": "y",
            "x0": -1, "y0": cy - rad, "x1": 1, "y1": cy + rad,
            "line": {"color": "#444", "width": 0.5},
        })
    return shapes


# ──────────────────────────────────────────────────────────────
#  启动
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║   S-Parameter Web Dashboard             ║")
    print("║   打开浏览器 → http://localhost:5050     ║")
    print("╚══════════════════════════════════════════╝")
    app.run(host="0.0.0.0", port=5050, debug=True)
