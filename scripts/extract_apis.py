#!/usr/bin/env python3
"""
自动提取 skrf + plotly 的公开 API 签名，生成结构化 JSON 索引。
替换手写的 api_refs.py 硬编码条目。

用法: python extract_apis.py → 生成 api_index.json
"""

import inspect
import json
import sys
import os
from typing import get_type_hints, Any

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_index.json")

# ═══════════════════════════════════════════════════════════════
#  提取工具
# ═══════════════════════════════════════════════════════════════

def _safe_sig(obj) -> str:
    """安全获取函数签名字符串。"""
    try:
        return str(inspect.signature(obj))
    except (ValueError, TypeError):
        return "(...)"


def _safe_doc(obj) -> str:
    """安全获取文档字符串首行。"""
    try:
        doc = inspect.getdoc(obj)
        if doc:
            return doc.split("\n")[0].strip()
    except Exception:
        pass
    return ""


def _is_public(name: str) -> bool:
    """是否公开 API。"""
    return not name.startswith("_")


def _is_method(obj) -> bool:
    return inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isroutine(obj)


# ═══════════════════════════════════════════════════════════════
#  skrf 提取
# ═══════════════════════════════════════════════════════════════

def extract_skrf() -> list[dict]:
    """提取 scikit-rf 的公开 API。"""
    entries = []

    try:
        import skrf as rf
    except ImportError:
        print("[WARN] scikit-rf 未安装，跳过")
        return entries

    # ── 顶层函数 ──
    for name in dir(rf):
        if not _is_public(name):
            continue
        obj = getattr(rf, name)
        if inspect.ismodule(obj):
            continue
        if _is_method(obj) or inspect.isclass(obj):
            sig = _safe_sig(obj) if _is_method(obj) else ""
            doc = _safe_doc(obj)
            entries.append({
                "lib": "skrf",
                "name": f"skrf.{name}",
                "kind": "class" if inspect.isclass(obj) else "function",
                "sig": sig,
                "desc": doc,
            })

    # ── Network 类的方法和属性 ──
    net = rf.Network
    for name in dir(net):
        if not _is_public(name):
            continue
        obj = getattr(net, name)
        kind = "property"
        sig = ""
        if _is_method(obj):
            kind = "method"
            sig = _safe_sig(obj)
        elif inspect.isdatadescriptor(obj):
            kind = "property"
        doc = _safe_doc(obj)
        if doc or sig or kind == "property":
            entries.append({
                "lib": "skrf",
                "name": f"Network.{name}",
                "kind": kind,
                "sig": sig,
                "desc": doc,
            })

    # ── Frequency 类 ──
    freq = rf.Frequency
    for name in dir(freq):
        if not _is_public(name):
            continue
        obj = getattr(freq, name)
        if _is_method(obj):
            entries.append({
                "lib": "skrf",
                "name": f"Frequency.{name}",
                "kind": "method",
                "sig": _safe_sig(obj),
                "desc": _safe_doc(obj),
            })

    # ── 常用属性的特殊说明 ──
    extras = [
        {"lib": "skrf", "name": "Network.s", "kind": "property",
         "desc": "S参数复数矩阵, shape=(nfreqs,nports,nports), 索引: ntwk.s[:,m,n]",
         "sig": "ntwk.s[:, m, n] -> np.ndarray(complex)"},
        {"lib": "skrf", "name": "Network.s_db", "kind": "property",
         "desc": "S参数dB值, 20*log10(|S|), 索引: ntwk.s_db[:,m,n]",
         "sig": "ntwk.s_db[:, m, n] -> np.ndarray(float)"},
        {"lib": "skrf", "name": "Network.s_mag", "kind": "property",
         "desc": "S参数线性幅度, 索引: ntwk.s_mag[:,m,n]",
         "sig": "ntwk.s_mag[:, m, n] -> np.ndarray(float)"},
        {"lib": "skrf", "name": "Network.s_deg", "kind": "property",
         "desc": "S参数相位(度), 索引: ntwk.s_deg[:,m,n]",
         "sig": "ntwk.s_deg[:, m, n] -> np.ndarray(float)"},
        {"lib": "skrf", "name": "Network.z", "kind": "property",
         "desc": "Z参数(阻抗)复数矩阵, 索引: ntwk.z[:,m,n]; 幅度用np.abs()",
         "sig": "ntwk.z[:, m, n] -> np.ndarray(complex)"},
        {"lib": "skrf", "name": "Network.y", "kind": "property",
         "desc": "Y参数(导纳)复数矩阵",
         "sig": "ntwk.y[:, m, n] -> np.ndarray(complex)"},
        {"lib": "skrf", "name": "Network.s_vswr", "kind": "property",
         "desc": "VSWR(电压驻波比), 索引: ntwk.s_vswr[:,port,port]",
         "sig": "ntwk.s_vswr[:, port, port] -> np.ndarray(float)"},
        {"lib": "skrf", "name": "Network.f", "kind": "property",
         "desc": "频率数组(Hz), 转GHz: ntwk.f/1e9",
         "sig": "ntwk.f -> np.ndarray(float)"},
        {"lib": "skrf", "name": "Network.nports", "kind": "property",
         "desc": "端口数", "sig": "ntwk.nports -> int"},
        {"lib": "skrf", "name": "Network.z0", "kind": "property",
         "desc": "参考阻抗矩阵, shape=(nports,nports)",
         "sig": "ntwk.z0 -> np.ndarray"},
        {"lib": "skrf", "name": "cascade", "kind": "operator",
         "desc": "级联: ntwk1 ** ntwk2 → Network",
         "sig": "ntwk1 ** ntwk2"},
        {"lib": "skrf", "name": "slice", "kind": "operator",
         "desc": "频率切片: ntwk['2-4ghz'] 或 ntwk['1e9-6e9']",
         "sig": "ntwk['start-stop'] → Network"},
    ]
    entries.extend(extras)

    return entries


# ═══════════════════════════════════════════════════════════════
#  plotly 提取
# ═══════════════════════════════════════════════════════════════

def extract_plotly() -> list[dict]:
    """提取 plotly 的公开 API。"""
    entries = []

    try:
        import plotly.graph_objects as go
        from plotly import subplots as sp
    except ImportError:
        print("[WARN] plotly 未安装，跳过")
        return entries

    # ── Figure 类方法 ──
    fig_methods = {
        "add_trace", "update_layout", "update_xaxes", "update_yaxes",
        "update_traces", "update_annotations", "add_annotation",
        "add_shape", "add_hline", "add_vline",
        "set_subplots", "write_html", "write_image", "write_json",
        "to_json", "to_dict", "to_image", "show", "data", "layout",
    }

    for name in dir(go.Figure):
        if not _is_public(name):
            continue
        if name not in fig_methods and not name.startswith("add_"):
            continue
        obj = getattr(go.Figure, name)
        kind = "method" if _is_method(obj) else "property"
        sig = _safe_sig(obj) if _is_method(obj) else ""
        doc = _safe_doc(obj)
        entries.append({
            "lib": "plotly",
            "name": f"Figure.{name}",
            "kind": kind,
            "sig": sig,
            "desc": doc,
        })

    # ── go.Scatter / go.Bar 等 trace 类型 ──
    trace_types = ["Scatter", "Bar", "Heatmap", "Histogram", "Box", "Violin",
                    "Pie", "Scatter3d", "Surface", "Contour", "Carpet",
                    "Scatterpolar", "Scatterternary", "Table", "Indicator"]
    for tname in trace_types:
        if hasattr(go, tname):
            cls = getattr(go, tname)
            sig = _safe_sig(cls)
            doc = _safe_doc(cls)
            entries.append({
                "lib": "plotly",
                "name": f"go.{tname}",
                "kind": "class",
                "sig": sig,
                "desc": doc,
            })

    # ── make_subplots ──
    entries.append({
        "lib": "plotly",
        "name": "make_subplots",
        "kind": "function",
        "sig": _safe_sig(sp.make_subplots),
        "desc": "创建子图; 双Y轴: make_subplots(specs=[[{'secondary_y':True}]])",
    })

    # ── 布局关键属性说明 ──
    layout_extras = [
        {"lib": "plotly", "name": "layout(xaxis_type)", "kind": "param",
         "desc": "X轴类型: 'linear'|'log'|'date'|'category', 例: fig.update_layout(xaxis_type='log')",
         "sig": ""},
        {"lib": "plotly", "name": "layout(yaxis_type)", "kind": "param",
         "desc": "Y轴类型: 'linear'|'log', 例: fig.update_layout(yaxis_type='log')",
         "sig": ""},
        {"lib": "plotly", "name": "layout(xaxis_title)", "kind": "param",
         "desc": "X轴标题, 例: fig.update_layout(xaxis_title='Frequency (GHz)')",
         "sig": ""},
        {"lib": "plotly", "name": "layout(yaxis_title)", "kind": "param",
         "desc": "Y轴标题, 例: fig.update_layout(yaxis_title='dB')",
         "sig": ""},
        {"lib": "plotly", "name": "layout(template)", "kind": "param",
         "desc": "模板: 'plotly_white'|'plotly_dark'|'ggplot2'|'seaborn'|'simple_white'",
         "sig": ""},
        {"lib": "plotly", "name": "layout(hovermode)", "kind": "param",
         "desc": "悬浮模式: 'closest'|'x'|'y'|'x unified'|False",
         "sig": ""},
        {"lib": "plotly", "name": "layout(legend)", "kind": "param",
         "desc": "图例: fig.update_layout(legend=dict(orientation='h',y=-0.2,x=0.5))",
         "sig": ""},
    ]
    entries.extend(layout_extras)

    # ── 常见错误 ──
    mistakes = [
        {"lib": "plotly", "name": "❌→✅ update_xaxis", "kind": "mistake",
         "desc": "❌ fig.update_xaxis(type='log') → ✅ fig.update_layout(xaxis_type='log')",
         "sig": "没有 update_xaxis 方法, 用 update_xaxes(复数) 或 update_layout(xaxis_type=)"},
        {"lib": "plotly", "name": "❌→✅ update_yaxis", "kind": "mistake",
         "desc": "❌ fig.update_yaxis(title='dB') → ✅ fig.update_layout(yaxis_title='dB')",
         "sig": "同上"},
        {"lib": "skrf", "name": "❌→✅ s_db索引", "kind": "mistake",
         "desc": "❌ ntwk.s_db[1,0] → ✅ ntwk.s_db[:,1,0]",
         "sig": "s_db 是 3D 数组，必须用 [:, m, n]"},
        {"lib": "skrf", "name": "❌→✅ cascade", "kind": "mistake",
         "desc": "❌ ntwk1.cascade(ntwk2) → ✅ ntwk1 ** ntwk2",
         "sig": "skrf 用 ** 运算符级联"},
        {"lib": "skrf", "name": "❌→✅ slice", "kind": "mistake",
         "desc": "❌ ntwk.slice(2e9,4e9) → ✅ ntwk['2-4ghz']",
         "sig": "切片用 ntwk['start-stop'] 格式，支持 GHz/MHz/kHz 后缀"},
    ]
    entries.extend(mistakes)

    return entries


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("提取 skrf API...")
    skrf_entries = extract_skrf()
    print(f"  skrf: {len(skrf_entries)} 条")

    print("提取 plotly API...")
    plotly_entries = extract_plotly()
    print(f"  plotly: {len(plotly_entries)} 条")

    all_entries = skrf_entries + plotly_entries

    # 去重（按 name）
    seen = set()
    unique = []
    for e in all_entries:
        if e["name"] not in seen:
            seen.add(e["name"])
            unique.append(e)

    index = {
        "version": 1,
        "total": len(unique),
        "entries": unique,
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(OUTPUT) / 1024
    print(f"\n✅ 已生成: {OUTPUT}")
    print(f"   总计 {len(unique)} 条API, {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
