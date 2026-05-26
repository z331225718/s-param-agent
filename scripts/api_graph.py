#!/usr/bin/env python3
"""
API 知识图谱 — NetworkX 有向图。
节点 = API 实体（类/方法/属性/参数/错误）
边   = 类型化关系（has_method, has_parameter, mistaken_for, alternative_to...）

搜索时不仅文本匹配，还返回 1-2 跳邻居，让 LLM 理解 API 间的关联。
"""

import json
import os
from typing import List, Dict, Tuple

import networkx as nx

_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_index.json")

# ═══════════════════════════════════════════════════════════════
#  图谱构建
# ═══════════════════════════════════════════════════════════════

def build() -> nx.DiGraph:
    """构建 API 知识图谱。"""
    G = nx.DiGraph()

    # ── 从 api_index.json 加载节点 ──
    if os.path.exists(_INDEX_PATH):
        with open(_INDEX_PATH, "r", encoding="utf-8") as f:
            entries = json.load(f).get("entries", [])
    else:
        entries = []

    for e in entries:
        name = e["name"]
        # 规范化：去掉 "❌→✅ " 前缀（手写索引用这个标记常见错误）
        if name.startswith("❌→✅ "):
            name = name[5:]
        kind = e.get("kind", "")
        if name not in G:  # 避免覆盖手工定义的节点
            G.add_node(name, lib=e.get("lib", ""), kind=kind,
                        sig=e.get("sig", ""), desc=e.get("desc", ""))

        # 从名称推断类归属
        if "." in name and kind in ("method", "property", "param"):
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = parts[0]
                if kind == "method":
                    G.add_edge(parent, name, type="has_method")
                elif kind == "property":
                    G.add_edge(parent, name, type="has_property")

    # ── 手工定义关键关系边 ──
    _add_skrf_edges(G)
    _add_plotly_edges(G)
    _add_mistake_edges(G)
    _add_concept_edges(G)

    return G


def _ensure_node(G, name, **attrs):
    if name not in G:
        G.add_node(name, **attrs)


def _add_skrf_edges(G):
    """skrf 关键关系。"""
    # Network 类的属性 → 形状说明
    _ensure_node(G, "Network.s_db", kind="property", lib="skrf",
                  desc="S参数dB值，3D数组 [nfreqs,nports,nports]")
    _ensure_node(G, "Network.s_deg", kind="property", lib="skrf",
                  desc="S参数相位(度)，3D数组")
    _ensure_node(G, "Network.s_mag", kind="property", lib="skrf",
                  desc="S参数线性幅度")
    _ensure_node(G, "Network.z", kind="property", lib="skrf",
                  desc="Z参数(阻抗)复数矩阵; 幅度: np.abs(ntwk.z[:,m,n])")
    _ensure_node(G, "Network.y", kind="property", lib="skrf",
                  desc="Y参数(导纳)复数矩阵")
    _ensure_node(G, "Network.s_vswr", kind="property", lib="skrf",
                  desc="VSWR，索引: ntwk.s_vswr[:,port,port]")
    _ensure_node(G, "Network.f", kind="property", lib="skrf",
                  desc="频率数组(Hz); 转GHz: ntwk.f/1e9")
    _ensure_node(G, "Network.nports", kind="property", lib="skrf",
                  desc="端口数")
    _ensure_node(G, "Network.z0", kind="property", lib="skrf",
                  desc="参考阻抗矩阵")

    # 3D 数组形状关系
    for prop in ["Network.s", "Network.s_db", "Network.s_mag", "Network.s_deg",
                  "Network.z", "Network.y", "Network.s_vswr"]:
        _ensure_node(G, prop, kind="property", lib="skrf")
        _ensure_node(G, "3D_array_indexing", kind="concept",
                      desc="skrf属性是3D数组，索引: [:, m, n] 不是 [m, n]")
        G.add_edge(prop, "3D_array_indexing", type="requires_shape")

    # Network → 属性
    for prop in ["Network.s_db", "Network.s_mag", "Network.s_deg", "Network.s",
                  "Network.z", "Network.y", "Network.s_vswr", "Network.f",
                  "Network.nports", "Network.z0", "Network.name"]:
        G.add_edge("Network", prop, type="has_property")

    # Network 方法
    for method, desc in [
        ("renormalize", "重归一化: ntwk.renormalize(z0) → Network"),
        ("interpolate", "重采样: ntwk.interpolate(freq_obj) → Network"),
        ("flip", "翻转端口: ntwk.flip() → Network"),
        ("write_touchstone", "写入文件: ntwk.write_touchstone(path)"),
    ]:
        node = f"Network.{method}"
        _ensure_node(G, node, kind="method", lib="skrf", desc=desc)
        G.add_edge("Network", node, type="has_method")

    # 操作符
    _ensure_node(G, "cascade", kind="operator", lib="skrf",
                  desc="级联: ntwk1 ** ntwk2 → Network（不是 .cascade()！）")
    _ensure_node(G, "slice", kind="operator", lib="skrf",
                  desc="频率切片: ntwk['2-4ghz'] 或 ntwk['1e9-6e9']（不是 .slice()！）")
    G.add_edge("Network", "cascade", type="has_operator")
    G.add_edge("Network", "slice", type="has_operator")

    # np 工具
    _ensure_node(G, "np.abs", kind="function", lib="numpy",
                  desc="取复数幅度: np.abs(ntwk.z[:,0,0])")
    _ensure_node(G, "np.angle", kind="function", lib="numpy",
                  desc="取复数相位: np.angle(ntwk.s[:,1,0], deg=True)")
    G.add_edge("Network.z", "np.abs", type="related_to")
    G.add_edge("Network.s", "np.angle", type="related_to")


def _add_plotly_edges(G):
    """plotly 关键关系。"""
    # Figure 方法
    for method, desc in [
        ("add_trace", "添加迹线: fig.add_trace(go.Scatter(...))"),
        ("update_layout", "更新布局: fig.update_layout(title=,xaxis_title=,yaxis_title=,xaxis_type=,yaxis_type=,template=...)"),
        ("update_xaxes", "更新X轴属性（复数！）: fig.update_xaxes(title_text=,type=,range=,...)"),
        ("update_yaxes", "更新Y轴属性（复数！）: fig.update_yaxes(title_text=,type=,range=,...)"),
        ("to_json", "序列化为JSON"),
        ("write_html", "写入HTML文件"),
    ]:
        node = f"Figure.{method}"
        _ensure_node(G, node, kind="method", lib="plotly", desc=desc)
        G.add_edge("Figure", node, type="has_method")

    # update_layout 的参数
    for param, desc in [
        ("xaxis_type", "X轴类型: 'linear'|'log'|'date'|'category'"),
        ("yaxis_type", "Y轴类型: 'linear'|'log'"),
        ("xaxis_title", "X轴标题字符串"),
        ("yaxis_title", "Y轴标题字符串"),
        ("title", "图表标题"),
        ("template", "模板名: 'plotly_white'|'plotly_dark'|..."),
        ("hovermode", "悬浮模式: 'closest'|'x'|'y'"),
        ("width", "图表宽度(px)"),
        ("height", "图表高度(px)"),
    ]:
        node = f"layout({param})"
        _ensure_node(G, node, kind="parameter", lib="plotly", desc=desc)
        G.add_edge("Figure.update_layout", node, type="has_parameter")

    # update_xaxes / update_yaxes 的参数
    for ax in ["xaxes", "yaxes"]:
        for param, desc in [("title_text", "轴标题"), ("type", "轴类型: 'linear'|'log'"),
                             ("range", "轴范围: [min, max]"), ("gridcolor", "网格颜色")]:
            node = f"update_{ax}({param})"
            _ensure_node(G, node, kind="parameter", lib="plotly", desc=desc)
            G.add_edge(f"Figure.update_{ax}", node, type="has_parameter")

    # go.Scatter 参数
    _ensure_node(G, "go.Scatter", kind="class", lib="plotly",
                  desc="散点/线图迹线: go.Scatter(x=,y=,mode=,name=,line=...)")
    for param in ["x", "y", "mode", "name", "line", "marker", "text",
                   "hovertemplate", "customdata", "fill", "stackgroup"]:
        node = f"Scatter.{param}"
        _ensure_node(G, node, kind="parameter", lib="plotly", desc=f"Scatter 的 {param} 参数")
        G.add_edge("go.Scatter", node, type="has_parameter")

    # make_subplots
    _ensure_node(G, "make_subplots", kind="function", lib="plotly",
                  desc="创建子图; 双Y轴: make_subplots(specs=[[{'secondary_y':True}]])")
    G.add_edge("make_subplots", "Figure", type="returns")
    _ensure_node(G, "secondary_y", kind="parameter", lib="plotly",
                  desc="add_trace 的 secondary_y 参数: True=右轴, False=左轴")
    G.add_edge("Figure.add_trace", "secondary_y", type="has_parameter")


def _add_mistake_edges(G):
    """常见错误 → 正确 API 的映射边。"""
    mistakes = [
        ("update_xaxis", "Figure.update_xaxes",
         "❌ fig.update_xaxis(type='log') → ✅ fig.update_layout(xaxis_type='log')"),
        ("update_yaxis", "Figure.update_yaxes",
         "❌ fig.update_yaxis(title='dB') → ✅ fig.update_layout(yaxis_title='dB')"),
        ("xaxis.type", "layout(xaxis_type)",
         "❌ fig.xaxis.type='log' → ✅ fig.update_layout(xaxis_type='log')"),
        ("s_db[1,0]", "3D_array_indexing",
         "❌ ntwk.s_db[1,0] → ✅ ntwk.s_db[:,1,0]"),
        ("s_db[0,0]", "3D_array_indexing",
         "❌ ntwk.s_db[0,0] → ✅ ntwk.s_db[:,0,0]"),
        ("vswr[0]", "Network.s_vswr",
         "❌ ntwk.vswr[0] → ✅ ntwk.s_vswr[:,0,0]"),
        (".cascade(", "cascade",
         "❌ ntwk.cascade(ntwk2) → ✅ ntwk1 ** ntwk2"),
        (".slice(", "slice",
         "❌ ntwk.slice(2e9,4e9) → ✅ ntwk['2-4ghz']"),
        ("Network.z[0,0]", "3D_array_indexing",
         "❌ ntwk.z[0,0] → ✅ ntwk.z[:,0,0]; 幅度: np.abs()"),
    ]

    for wrong, correct, desc in mistakes:
        _ensure_node(G, wrong, kind="mistake", lib="", desc=desc)
        _ensure_node(G, correct, kind="correct_api", lib="", desc=desc)
        G.add_edge(wrong, correct, type="mistaken_for")


def _add_concept_edges(G):
    """概念节点和关系。"""
    _ensure_node(G, "log_axis", kind="concept",
                  desc="对数轴: fig.update_layout(xaxis_type='log', yaxis_type='log')")
    G.add_edge("layout(xaxis_type)", "log_axis", type="related_to")
    G.add_edge("layout(yaxis_type)", "log_axis", type="related_to")

    _ensure_node(G, "dual_y_axis", kind="concept",
                  desc="双Y轴: make_subplots(specs=[[{'secondary_y':True}]]) → add_trace(secondary_y=...)")
    G.add_edge("make_subplots", "dual_y_axis", type="related_to")
    G.add_edge("secondary_y", "dual_y_axis", type="related_to")

    _ensure_node(G, "smith_chart", kind="concept",
                  desc="Smith圆图: 用反射系数Γ画在极坐标/单位圆内")
    G.add_edge("Network.s", "smith_chart", type="related_to")

    _ensure_node(G, "3D_array_indexing", kind="concept",
                  desc="skrf属性(如s_db,z,vswr)是3D数组(nfreqs,nports,nports)，索引必须用[:,m,n]不是[m,n]")
    G.add_edge("s_db[1,0]", "3D_array_indexing", type="mistaken_for")
    G.add_edge("s_db[0,0]", "3D_array_indexing", type="mistaken_for")

    # 频率必须转 GHz
    _ensure_node(G, "freq_not_scaled", kind="mistake", lib="skrf",
                  desc="❌ 直接用 ntwk.f → X轴显示 1B/2B (Plotly把1e9缩写成1B)。✅ 必须 freq_ghz = ntwk.f / 1e9")
    _ensure_node(G, "ntwk.f_div_1e9", kind="correct_api", lib="skrf",
                  desc="✅ freq_ghz = ntwk.f / 1e9; 然后 x=freq_ghz, xaxis_title='Frequency (GHz)'",
                  sig="freq_ghz = ntwk.f / 1e9")
    G.add_edge("freq_not_scaled", "ntwk.f_div_1e9", type="mistaken_for")
    G.add_edge("Network.f", "ntwk.f_div_1e9", type="related_to")

    # 禁止 SI 前缀
    _ensure_node(G, "SI_prefix_bad", kind="mistake", lib="plotly",
                  desc="❌ Plotly 自动 SI 前缀(μ/k/M/B) → 标签丑陋。✅ 强制科学计数法")
    _ensure_node(G, "exponentformat_power", kind="concept", lib="plotly",
                  desc="✅ fig.update_xaxes(exponentformat='power', showexponent='all'); y轴同理",
                  sig="fig.update_xaxes(exponentformat='power', showexponent='all')")
    G.add_edge("SI_prefix_bad", "exponentformat_power", type="mistaken_for")
    G.add_edge("Figure.update_xaxes", "exponentformat_power", type="related_to")
    G.add_edge("Figure.update_yaxes", "exponentformat_power", type="related_to")


# ═══════════════════════════════════════════════════════════════
#  图谱查询
# ═══════════════════════════════════════════════════════════════

# 单例
_GRAPH: nx.DiGraph = None


def get_graph() -> nx.DiGraph:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build()
    return _GRAPH


def graph_search(query: str, max_hops: int = 2, top_k: int = 8) -> List[dict]:
    """
    图谱搜索：文本匹配 + N 跳邻居遍历。

    返回格式:
    [
      {
        "node": "Figure.update_xaxes",
        "score": 10,
        "kind": "method",
        "desc": "...",
        "neighbors": [
          {"node": "Figure", "edge": "has_method"},
          {"node": "update_xaxes(title_text)", "edge": "has_parameter"},
          {"node": "update_xaxis", "edge": "mistaken_for (反向)"},
        ],
        "paths_to_mistakes": ["update_xaxis →(mistaken_for)→ Figure.update_xaxes"],
      }
    ]
    """
    G = get_graph()
    if not G:
        return []

    q = query.lower()
    scored = []

    # 1. 文本匹配找到种子节点
    for node in G.nodes:
        score = 0
        nd = node.lower()
        data = G.nodes[node]
        desc = data.get("desc", "").lower()
        sig = data.get("sig", "").lower()

        if q in nd:
            score += 10
        elif any(part in nd for part in q.split() if len(part) >= 3):
            score += 6
        if q in desc:
            score += 4
        if q in sig:
            score += 3

        if score > 0:
            scored.append((score, node))

    scored.sort(key=lambda x: -x[0])
    seeds = [node for _, node in scored[:top_k]]

    # 2. 对每个种子，遍历 N 跳邻居
    results = []
    for seed in seeds[:5]:  # 最多展开 5 个种子
        seed_data = dict(G.nodes[seed])

        neighbors = []
        paths_to_mistakes = []

        # BFS 收集邻居
        visited = {seed}
        frontier = [seed]
        for hop in range(max_hops):
            next_frontier = []
            for node in frontier:
                for _, neighbor, edata in G.out_edges(node, data=True):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
                        neighbors.append({
                            "node": neighbor,
                            "edge": edata.get("type", "?"),
                            "hop": hop + 1,
                            "desc": G.nodes[neighbor].get("desc", "")[:100],
                        })
                # 入边（谁指向我，如 "update_xaxis →(mistaken_for)→ Figure.update_xaxes"）
                for pred, _, edata in G.in_edges(node, data=True):
                    if pred not in visited:
                        visited.add(pred)
                        next_frontier.append(pred)
                        edge_type = edata.get("type", "?")
                        neighbors.append({
                            "node": pred,
                            "edge": f"{edge_type} (反向)",
                            "hop": hop + 1,
                            "desc": G.nodes[pred].get("desc", "")[:100],
                        })
                        if edge_type == "mistaken_for":
                            paths_to_mistakes.append(
                                f"{pred} →({edge_type})→ {node}"
                            )
            frontier = next_frontier

        results.append({
            "node": seed,
            "kind": seed_data.get("kind", ""),
            "desc": seed_data.get("desc", ""),
            "sig": seed_data.get("sig", ""),
            "neighbors": neighbors[:15],
            "paths_to_mistakes": paths_to_mistakes,
        })

    return results


def build_rich_prompt(query: str) -> str:
    """
    生成丰富上下文提示：文本匹配 + 图谱邻居。
    用于注入 system prompt 的"相关 API 上下文"区域。
    """
    results = graph_search(query, max_hops=2, top_k=5)
    if not results:
        return ""

    lines = ["## 🔗 相关知识图谱（API 及关联实体）"]
    for r in results[:3]:
        lines.append(f"\n### {r['node']}")
        if r.get("desc"):
            lines.append(f"说明: {r['desc']}")
        if r.get("sig"):
            lines.append(f"签名: `{r['sig']}`")
        if r.get("paths_to_mistakes"):
            lines.append("纠错路径:")
            for p in r["paths_to_mistakes"]:
                lines.append(f"  {p}")
        if r.get("neighbors"):
            # 按边类型分组
            by_type = {}
            for nb in r["neighbors"]:
                t = nb["edge"]
                by_type.setdefault(t, []).append(nb["node"])
            for t, nodes in list(by_type.items())[:4]:
                lines.append(f"{t}: {', '.join(nodes[:5])}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    G = build()
    print(f"图谱: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    print()

    # 测试 1: 搜索错误 API
    print("=== 搜索 'update_xaxis' ===")
    for r in graph_search("update_xaxis", max_hops=1)[:2]:
        print(f"\n种子: {r['node']} ({r['kind']})")
        print(f"邻居 ({len(r['neighbors'])} 个):")
        for nb in r["neighbors"][:6]:
            print(f"  {nb['edge']}: {nb['node']}")
        if r["paths_to_mistakes"]:
            print("纠错路径:", r["paths_to_mistakes"])

    print("\n=== 搜索 's_db' ===")
    for r in graph_search("s_db", max_hops=1)[:2]:
        print(f"种子: {r['node']} → 邻居: {[n['node'] for n in r['neighbors'][:5]]}")
