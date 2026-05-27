#!/usr/bin/env python3
"""
skrf + plotly API 参考查询模块。
从 api_index.json 加载索引 + api_graph 知识图谱，提供文本搜索和图谱遍历。
"""

import json
import os
import sys
from typing import List

# ── 加载索引 ───────────────────────────────────────────────────

_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_index.json")
if not os.path.exists(_INDEX_PATH) and getattr(sys, 'frozen', False):
    _INDEX_PATH = os.path.join(sys._MEIPASS, "api_index.json")
_ENTRIES: List[dict] = []

def _load():
    global _ENTRIES
    if _ENTRIES:
        return
    if os.path.exists(_INDEX_PATH):
        with open(_INDEX_PATH, "r", encoding="utf-8") as f:
            _ENTRIES = json.load(f).get("entries", [])

_load()

# ── 搜索 ───────────────────────────────────────────────────────

def search(query: str, top_k: int = 5) -> List[dict]:
    """按关键词搜索 API 条目。"""
    if not _ENTRIES:
        return []
    q = query.lower()
    scored = []
    for e in _ENTRIES:
        score = 0
        name = e.get("name", "").lower()
        desc = e.get("desc", "").lower()
        sig = e.get("sig", "").lower()
        if q in name:
            score += 10
        if q in desc:
            score += 5
        if q in sig:
            score += 3
        # 关键词拆分匹配
        for part in q.replace("_", " ").replace(".", " ").split():
            if len(part) >= 3 and part in name:
                score += 2
        if score > 0:
            scored.append((score, e))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:top_k]]


# ── 纠错提示 ───────────────────────────────────────────────────

def build_fix_prompt(error_msg: str) -> str:
    """根据错误信息，用图谱搜索相关的正确 API，生成纠错提示文本。"""
    import re
    from api_graph import graph_search as _graph_search

    # 提取 'xxx' 引号中的内容（报错的函数名），长的优先
    quoted = re.findall(r"'([^']+)'", error_msg)
    quoted.sort(key=len, reverse=True)

    # 用最长的引号内容做图谱搜索
    graph_results = []
    for q in quoted:
        if len(q) >= 4 and not q.startswith("__"):
            graph_results = _graph_search(q, max_hops=2, top_k=3)
            if graph_results:
                break

    # fallback: 常见错误关键词
    if not graph_results:
        for kw in ["xaxis", "yaxis", "update_layout", "cascade", "slice",
                    "vswr", "log", "s_db", "add_trace", "secondary_y"]:
            if kw.lower() in error_msg.lower():
                graph_results = _graph_search(kw, max_hops=2, top_k=3)
                if graph_results:
                    break

    lines = ["## 🔧 纠错参考：知识图谱搜索结果"]

    for r in (graph_results or [])[:3]:
        lines.append(f"\n### {r['node']} ({r.get('kind','')})")
        if r.get("desc"):
            lines.append(f"说明: {r['desc']}")
        if r.get("sig"):
            lines.append(f"签名: `{r['sig']}`")
        if r.get("paths_to_mistakes"):
            lines.append("🔗 纠错路径:")
            for p in r["paths_to_mistakes"]:
                lines.append(f"  {p}")
        if r.get("neighbors"):
            by_type = {}
            for nb in r["neighbors"]:
                t = nb["edge"]
                by_type.setdefault(t, []).append(nb["node"])
            lines.append("关联 API:")
            for t, nodes in by_type.items():
                lines.append(f"  {t}: {', '.join(nodes[:4])}")

    if len(lines) == 1:
        lines.append("\n未找到精确匹配。请检查：")
        lines.append("- plotly: update_xaxes/update_yaxes（复数！），xaxis_type 在 update_layout 里设")
        lines.append("- skrf: s_db[:,m,n]（3D数组），ntwk['2-4ghz']（切片），ntwk1 ** ntwk2（级联）")

    return "\n".join(lines)


# ── API 速查（精简版，注入 system prompt） ─────────────────────

def build_api_prompt() -> str:
    """生成一段精简的 API 速查文本（~1200 chars），注入 LLM system prompt。"""
    # 预定义的"高频必知"API——不搜索全库，只选最常用的
    must_know = [
        "plotly.Figure.update_layout",
        "plotly.Figure.add_trace",
        "plotly.layout(xaxis_type)",
        "plotly.layout(yaxis_type)",
        "plotly.❌→✅ update_xaxis",
        "plotly.make_subplots",
        "skrf.Network.s_db",
        "skrf.Network.s_deg",
        "skrf.Network.z",
        "skrf.Network.s_vswr",
        "skrf.Network.f",
        "skrf.❌→✅ s_db索引",
        "skrf.❌→✅ cascade",
        "skrf.❌→✅ slice",
    ]

    lines = ["## ⚠️ 高频 API 速查（生成代码前务必参考）", ""]
    for name in must_know:
        results = search(name, top_k=1)
        if results:
            r = results[0]
            desc = r.get("desc", "")
            sig = r.get("sig", "")
            if desc and sig:
                lines.append(f"- `{sig}` — {desc}")
            elif desc:
                lines.append(f"- {desc}")

    lines.append("")
    lines.append("### plotly 铁律")
    lines.append("- ❌ 没有 update_xaxis / update_yaxis（单数）")
    lines.append("- ✅ 用 update_xaxes / update_yaxes（复数）或 update_layout(xaxis_type=)")
    lines.append("- ✅ log 轴: `fig.update_layout(xaxis_type='log', yaxis_type='log')`")
    lines.append("- **每张图必须加**: `fig.update_xaxes(exponentformat='power',showexponent='all')`")
    lines.append("  否则出现 μ/k/B 等 SI 前缀（如 1B 替代 1e9）！同样 y 轴也要。")
    lines.append("")
    lines.append("### 端口终端（短路/开路/负载）")
    lines.append("- **不要用 S 参数手推公式！** 直接用 Z 参数:")
    lines.append("  `z=ntwk.z; ZL=0; zin=z[:,0,0]-(z[:,0,1]*z[:,1,0])/(z[:,1,1]+ZL)`")
    lines.append("- 短路 ZL=0; 开路 ZL=np.inf; 负载 ZL=50")
    lines.append("- `np.abs(zin)` 取幅度，log轴需要所有值 > 0")
    lines.append("")
    lines.append("### skrf 铁律")
    lines.append("- **频率必须转GHz**: `freq_ghz = ntwk.f / 1e9`，直接用 ntwk.f 会导致 X轴显示 1B/2B！")
    lines.append("- S参数是3D数组: `ntwk.s_db[:, m, n]` 不是 `[m, n]`")
    lines.append("- 切片: `ntwk['2-4ghz']` 不是 `.slice()`")
    lines.append("- 级联: `ntwk1 ** ntwk2` 不是 `.cascade()`")

    return "\n".join(lines)


# ── 索引统计 ───────────────────────────────────────────────────

def stats() -> dict:
    return {
        "total": len(_ENTRIES),
        "skrf": sum(1 for e in _ENTRIES if e.get("lib") == "skrf"),
        "plotly": sum(1 for e in _ENTRIES if e.get("lib") == "plotly"),
        "size_kb": os.path.getsize(_INDEX_PATH) / 1024 if os.path.exists(_INDEX_PATH) else 0,
    }


# ── 测试 ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("索引统计:", stats())
    print("\n── 搜索 'xaxis_type' ──")
    for r in search("xaxis_type"):
        print(f"  {r['name']}: {r.get('desc','')[:80]}")
    print("\n── 纠错提示 ──")
    print(build_fix_prompt("AttributeError: 'Figure' object has no attribute 'update_xaxis'"))
