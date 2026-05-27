#!/usr/bin/env python3
"""
S-Parameter Web Dashboard
Flask 驱动的本地 Web 仪表盘：上传 .sNp → 交互式 Plotly 图表 → 导出
启动: python app.py  →  浏览器打开 http://localhost:5050
"""

import os
import io
import json
import tempfile
import traceback
from pathlib import Path

import numpy as np
import skrf as rf
import plotly
from flask import Flask, render_template, request, jsonify, send_file

# ── 导入本地工具库 ───────────────────────────────────────────
import s_params as sp
import nl_parser
import llm_chat
import code_agent

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB 上传上限（支持大端口文件如 .s64p）

# 会话级存储：上传文件的解析结果
sessions = {}  # { session_id: { "networks": {name: Network}, "freq_unit": "ghz" } }

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
        file_path = os.path.abspath(last.get("path", ""))

    # 如果文本中包含文件路径，优先使用（提取 .sNp 路径）
    import re as _re
    snp_match = _re.search(r"([\w./\\-]+\.s\dp)", text, _re.IGNORECASE)
    if snp_match:
        file_path = os.path.abspath(snp_match.group(1))

    # ── 构建网络字典（名称 → dict），供 Agent 执行时预加载 ──
    nets_dict = {}
    for net_name, net_info in ses.items():
        nets_dict[net_name] = {
            "path": net_info.get("path", ""),
            "nports": net_info.get("nports", 0),
        }

    # ── 调用代码生成 Agent ──
    result = code_agent.generate_code(text, file_path=file_path,
                                       networks=nets_dict if nets_dict else None)

    if "error" in result:
        return jsonify({
            "reply": f"❌ {result['error']}",
            "results": [],
            "mode": "agent",
            "code": result.get("code", ""),
            "llm_raw": result.get("llm_raw", ""),
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
        "history": result.get("history", []),
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
        return jsonify({"reply": "请说点什么吧 😊", "results": []})

    # 列出已有文件
    available = list(sessions.get(session_id, {}).get("networks", {}).keys())

    # ── 解析自然语言：LLM 优先，规则 fallback ──
    ops = None
    parse_mode = "rule"

    if llm_chat.is_available():
        llm_ops = llm_chat.parse_with_llm(text, available)
        if llm_ops:
            ops = llm_chat.ops_to_nl_parser_format(llm_ops)
            parse_mode = "llm"

    if ops is None:
        try:
            ops = nl_parser.parse(text, available)
        except Exception as e:
            return jsonify({"reply": f"解析失败: {str(e)}", "results": []})
        parse_mode = "rule"

    if not ops:
        return jsonify({
            "reply": "不太确定你想做什么。试试这样说：\n"
                     "• \"读取 filter.s2p\"\n"
                     "• \"画 S11 和 S21 的 dB 图\"\n"
                     "• \"级联 A 和 B，画 S21 Smith 圆图\"\n"
                     "• \"导出 S21 为 CSV\"",
            "results": []
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
    })


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
        # 尝试在当前目录及子目录搜索文件
        import glob as globmod
        candidates = globmod.glob(f"**/{op.target}", recursive=True)
        if not candidates:
            candidates = globmod.glob(f"**/{os.path.basename(op.target)}", recursive=True)
        if candidates:
            op.target = candidates[0]

        ntwk = sp.load_ntwk(op.target)
        name = os.path.splitext(os.path.basename(op.target))[0]
        ses["networks"][name] = {
            "path": op.target,
            "nports": ntwk.nports,
            "f_min": float(ntwk.f[0]),
            "f_max": float(ntwk.f[-1]),
            "npoints": len(ntwk.f),
            "params": sp.list_params(ntwk),
        }
        info_str = sp.info(ntwk)
        return {
            "type": "text",
            "message": f"✅ 已加载 **{name}**\n{info_str}",
            "ntwk_name": name,
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
        ses["networks"][result_name] = {
            "path": tmp_path,
            "nports": ntwk_result.nports,
            "f_min": float(ntwk_result.f[0]),
            "f_max": float(ntwk_result.f[-1]),
            "npoints": len(ntwk_result.f),
            "params": sp.list_params(ntwk_result),
        }
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
        ses["networks"][sliced_name] = {
            "path": tmp_path,
            "nports": sliced.nports,
            "f_min": float(sliced.f[0]),
            "f_max": float(sliced.f[-1]),
            "npoints": len(sliced.f),
            "params": sp.list_params(sliced),
        }
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
            ports = []
            for p in (params or [f"S11"]):
                if isinstance(p, str) and p.upper().startswith("VSWR"):
                    ports.append(int(p[4:]) - 1)
                elif isinstance(p, str) and p.upper().startswith("S"):
                    ports.append(int(p[1]) - 1)
            if not ports:
                ports = [0]
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
        ses["networks"][result_name] = {
            "path": tmp_path,
            "nports": result.nports,
            "f_min": float(result.f[0]),
            "f_max": float(result.f[-1]),
            "npoints": len(result.f),
            "params": sp.list_params(result),
        }
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
        import glob as globmod
        matches = globmod.glob(pattern, recursive=True)
        if not matches:
            matches = globmod.glob(f"**/{pattern}", recursive=True)
        if not matches:
            return {"type": "error", "message": f"未找到匹配 '{pattern}' 的文件"}
        loaded = []
        for path in matches:
            try:
                ntwk = sp.load_ntwk(path)
                name = os.path.splitext(os.path.basename(path))[0]
                ses["networks"][name] = {
                    "path": path,
                    "nports": ntwk.nports,
                    "f_min": float(ntwk.f[0]),
                    "f_max": float(ntwk.f[-1]),
                    "npoints": len(ntwk.f),
                    "params": sp.list_params(ntwk),
                }
                loaded.append(name)
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
            ps = str(ps).strip().upper()
            if ps.startswith("S") and len(ps) >= 3:
                m = int(ps[1]) - 1
                n = int(ps[2]) - 1
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


def _basename(path_or_name: str) -> str:
    return os.path.splitext(os.path.basename(path_or_name))[0]


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

        # ── 快速解析文件头（不加载完整 S 矩阵）──
        header = _read_touchstone_header(tmp_path)
        if header is None:
            os.unlink(tmp_path)
            return jsonify({"error": "无法解析 Touchstone 文件头"}), 400

        if session_id not in sessions:
            sessions[session_id] = {"networks": {}}

        nports = header["nports"]
        all_params = [f"S{m+1}{n+1}" for m in range(nports) for n in range(nports)]

        sessions[session_id]["networks"][name] = {
            "path": tmp_path,
            "_ntwk": None,  # 延迟加载
            "nports": nports,
            "f_min": header["f_min"],
            "f_max": header["f_max"],
            "npoints": header["npoints"],
            "params": all_params,
        }

        freq_unit = header["freq_unit"]
        sessions[session_id]["freq_unit"] = freq_unit

        display_params = all_params if len(all_params) <= 200 else all_params[:200]

        return jsonify({
            "ok": True,
            "name": name,
            "nports": nports,
            "f_min": header["f_min"],
            "f_max": header["f_max"],
            "f_unit": freq_unit,
            "npoints": header["npoints"],
            "params": display_params,
            "params_truncated": len(all_params) > 200,
            "total_params": len(all_params),
        })
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
            header = _read_touchstone_header(tmp_path)
            if header is None:
                os.unlink(tmp_path)
                results.append({"ok": False, "name": file.filename, "error": "无法解析文件头"})
                continue
            name = Path(file.filename).stem
            nports = header["nports"]
            all_params = [f"S{m+1}{n+1}" for m in range(nports) for n in range(nports)]
            sessions[session_id]["networks"][name] = {
                "path": tmp_path,
                "_ntwk": None,
                "nports": nports,
                "f_min": header["f_min"],
                "f_max": header["f_max"],
                "npoints": header["npoints"],
                "params": all_params,
            }
            results.append({
                "ok": True,
                "name": name,
                "nports": nports,
                "f_min": header["f_min"],
                "f_max": header["f_max"],
                "npoints": header["npoints"],
                "params": all_params[:200],
            })
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
    接收 JSON: { "path": "/data/big.s64p", "session": "default" }
    """
    data = request.get_json()
    local_path = data.get("path", "").strip()
    session_id = data.get("session", "default")

    if not local_path:
        return jsonify({"error": "请提供本地文件路径"}), 400
    if not os.path.exists(local_path):
        return jsonify({"error": f"文件不存在: {local_path}"}), 404

    try:
        header = _read_touchstone_header(local_path)
        if header is None:
            return jsonify({"error": "无法解析 Touchstone 文件头"}), 400

        name = data.get("name") or os.path.splitext(os.path.basename(local_path))[0]
        if session_id not in sessions:
            sessions[session_id] = {"networks": {}}

        nports = header["nports"]
        all_params = [f"S{m+1}{n+1}" for m in range(nports) for n in range(nports)]

        sessions[session_id]["networks"][name] = {
            "path": local_path,
            "_ntwk": None,
            "nports": nports,
            "f_min": header["f_min"],
            "f_max": header["f_max"],
            "npoints": header["npoints"],
            "params": all_params,
        }

        display_params = all_params if len(all_params) <= 200 else all_params[:200]

        return jsonify({
            "ok": True,
            "name": name,
            "nports": nports,
            "f_min": header["f_min"],
            "f_max": header["f_max"],
            "f_unit": header["freq_unit"],
            "npoints": header["npoints"],
            "params": display_params,
            "params_truncated": len(all_params) > 200,
            "total_params": len(all_params),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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
            header = _read_touchstone_header(path)
            if header is None:
                results.append({"ok": False, "name": os.path.basename(path), "error": "无法解析文件头"})
                continue
            name = os.path.splitext(os.path.basename(path))[0]
            # 处理重名：添加后缀
            if name in sessions[session_id]["networks"]:
                base = name
                idx = 2
                while f"{base}_{idx}" in sessions[session_id]["networks"]:
                    idx += 1
                name = f"{base}_{idx}"
            nports = header["nports"]
            all_params = [f"S{m+1}{n+1}" for m in range(nports) for n in range(nports)]
            sessions[session_id]["networks"][name] = {
                "path": path,
                "_ntwk": None,
                "nports": nports,
                "f_min": header["f_min"],
                "f_max": header["f_max"],
                "npoints": header["npoints"],
                "params": all_params,
            }
            results.append({
                "ok": True,
                "name": name,
                "nports": nports,
                "f_min": header["f_min"],
                "f_max": header["f_max"],
                "npoints": header["npoints"],
                "params": all_params[:200],
            })
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
        if os.path.exists(info["path"]):
            os.unlink(info["path"])
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
                m, n = _parse_param(p)

                if chart_type == "db":
                    trace = _make_db_trace(ntwk, m, n, label, p)
                elif chart_type == "deg":
                    trace = _make_deg_trace(ntwk, m, n, label, p)
                elif chart_type == "smith":
                    trace = _make_smith_trace(ntwk, m, n, label, p)
                elif chart_type == "vswr":
                    trace = _make_vswr_trace(ntwk, m, label, p)
                elif chart_type == "groupdelay":
                    trace = _make_groupdelay_trace(ntwk, m, n, label, p)
                elif chart_type == "mag":
                    trace = _make_mag_trace(ntwk, m, n, label, p)
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
    if resp[1] != 200:
        return resp
    fig_json = resp[0].get_json()

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ margin: 0; padding: 20px; background: #1a1a2e; font-family: -apple-system, sans-serif; }}
  #chart {{ width: 100%; height: 85vh; }}
  h2 {{ color: #e0e0e0; text-align: center; }}
</style>
</head>
<body>
<h2>{title}</h2>
<div id="chart"></div>
<script>
  Plotly.newPlot('chart', {json.dumps(fig_json['data'], cls=plotly.utils.PlotlyJSONEncoder)}, {json.dumps(fig_json['layout'], cls=plotly.utils.PlotlyJSONEncoder)}, {{ responsive: true }});
</script>
</body>
</html>"""

    return jsonify({"html": html, "title": title})


@app.route("/api/export/html", methods=["POST"])
def export_html():
    """下载独立 HTML 报告"""
    data = request.get_json()
    resp = generate_chart_html()
    if resp[1] != 200:
        return resp
    html = resp[0].get_json()["html"]

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
        ntwk = sp.slice_freq(ntwk, freq_range[0], freq_range[1])

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
    ses[result_name] = {
        "path": tmp_path,
        "nports": result.nports,
        "f_min": float(result.f[0]),
        "f_max": float(result.f[-1]),
        "npoints": len(result.f),
        "params": sp.list_params(result),
    }
    sessions[session_id]["networks"][result_name] = ses[result_name]

    return jsonify({
        "ok": True,
        "name": result_name,
        "chain": chain_names,
        "nports": result.nports,
        "f_min": float(result.f[0]),
        "f_max": float(result.f[-1]),
        "npoints": len(result.f),
        "params": sp.list_params(result),
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
            ntwk = sp.slice_freq(ntwk, freq_range[0], freq_range[1])
        networks.append(ntwk)
        names.append(name)

    # 找参考索引
    reference_idx = 0
    if ref_name and ref_name in names:
        reference_idx = names.index(ref_name)

    # 解析参数
    params = []
    for ps in param_strs:
        ps = ps.strip().upper()
        if ps.startswith("S") and len(ps) == 3:
            m = int(ps[1]) - 1
            n = int(ps[2]) - 1
            params.append((m, n))
    if not params:
        params = [(1, 0)]  # 默认 S21

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
        ntwk = sp.slice_freq(ntwk, freq_range[0], freq_range[1])

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

    sessions[session_id]["networks"][result_name] = {
        "path": tmp_path,
        "nports": ntwk_result.nports,
        "f_min": float(ntwk_result.f[0]),
        "f_max": float(ntwk_result.f[-1]),
        "npoints": len(ntwk_result.f),
        "params": sp.list_params(ntwk_result),
    }

    return jsonify({
        "ok": True,
        "name": result_name,
        "nports": ntwk_result.nports,
        "f_min": float(ntwk_result.f[0]),
        "f_max": float(ntwk_result.f[-1]),
        "npoints": len(ntwk_result.f),
        "params": sp.list_params(ntwk_result),
    })


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
    sessions[session_id]["networks"][result_name] = {
        "path": tmp_path,
        "nports": ntwk_result.nports,
        "f_min": float(ntwk_result.f[0]),
        "f_max": float(ntwk_result.f[-1]),
        "npoints": len(ntwk_result.f),
        "params": sp.list_params(ntwk_result),
    }

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


def _read_touchstone_header(path: str) -> dict:
    """
    快速读取 Touchstone 文件头，不解析完整 S 矩阵。
    适用于大端口文件（.s64p 等），毫秒级返回。
    自动跳过任意长度的注释头（仿真软件可能导出上百行 ! 注释）。
    """
    import re as _re
    data_re = _re.compile(r"^\s*-?\d")

    # ── 第一遍：扫描找到 # 行和第一个数据行 ──
    freq_unit = "ghz"
    nports = 0
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

    # ── 从第一个数据行推导端口数 ──
    cols = first_data_line.split()
    nvals = len(cols) - 1  # 减掉频率列
    if nvals > 0:
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


def _parse_param(p: str):
    """'S11' → (0, 0), 'S21' → (1, 0)"""
    p = p.strip().upper()
    if p.startswith("S") and len(p) == 3:
        m = int(p[1]) - 1
        n = int(p[2]) - 1
        return m, n
    if p.startswith("VSWR"):
        port = int(p[4:]) - 1
        return port, None
    raise ValueError(f"无法解析参数: {p}")


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
