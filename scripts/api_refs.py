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
    """生成精简 API 速查（不重复系统 prompt 已有规则）。"""
    return """## 补充 API 速查
- plotly 用 update_xaxes/update_yaxes（复数），不是 update_xaxis
- make_subplots(specs=[[{'secondary_y':True}]]) 创建双Y轴
- add_trace(trace, row=1, col=1, secondary_y=False) 添加曲线到子图
- s_db[:,m,n] 3D索引取S参数dB值；s_deg[:,m,n] 取相位
- ntwk.z[:,p,p] 取端口阻抗；ntwk.s_vswr[:,p,p] 取VSWR
- 级联: ntwk1 ** ntwk2；频率切片: ntwk['2-4ghz']
- 端口终端用Z参数: zin = z[:,0,0] - z[:,0,1]*z[:,1,0]/(z[:,1,1]+ZL)"""


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
