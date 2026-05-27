#!/usr/bin/env python3
"""
自然语言解析器 — 将中文/英文口语指令转换为结构化 S 参数操作。

支持的中文口语模式：
  - "读 filter.s2p"              → load
  - "看看 amp.s2p 的信息"       → info
  - "画 S21 的 dB 图"           → plot(db, S21)
  - "画 S11 的 Smith 圆图，2到4G" → plot(smith, S11, 2-4GHz)
  - "级联 A 和 B，画 S21 相位"  → cascade + plot(deg, S21)
  - "导出 S21 为 CSV"           → export(csv, S21)
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class SParamOp:
    """单个 S 参数操作"""
    action: str                    # load | load_batch | info | plot | cascade | cascade_chain | deembed | slice | export | list | compare
    target: str = ""               # 文件路径 or 网络名称
    params: List[str] = field(default_factory=list)   # ["S11", "S21"]
    chart_type: str = ""           # db | deg | smith | vswr | groupdelay | mag
    freq_range: Optional[Tuple[float, float]] = None  # (start_Hz, stop_Hz)
    cascade_with: str = ""         # 级联的第二个网络
    chain: List[str] = field(default_factory=list)    # 链式级联的多个网络 ["LNA", "BPF", "AMP"]
    deembed_fixture: str = ""      # 去嵌夹具
    export_format: str = ""        # csv | touchstone | html
    title: str = ""                # 自定义图表标题
    result_name: str = ""          # 结果命名
    dual_axis: bool = False        # 是否使用双Y轴（左右两个坐标系）
    batch_pattern: str = ""        # 批量加载的通配符模式
    compare_networks: List[str] = field(default_factory=list)  # 对比的网络列表
    reference: str = ""            # 差异对比的参考网络
    show_diff: bool = True         # 是否显示差异


# ──────────────────────────────────────────────
#  关键词词典
# ──────────────────────────────────────────────

LOAD_WORDS = ["读", "读取", "加载", "载入", "打开", "load", "open", "read"]
INFO_WORDS = ["看", "看看", "查看", "信息", "概览", "基本信息", "info", "summary", "describe"]
PLOT_WORDS = ["画", "绘制", "显示", "plot", "draw", "show", "chart", "图"]
CASCADE_WORDS = ["级联", "串接", "连接", "串联", "cascade", "connect", "chain"]
CHAIN_WORDS = ["→", "->", ">>", "then", "然后接", "再接", "接着接"]
DEEMBED_WORDS = ["去嵌", "去嵌入", "反嵌", "deembed", "de-embed"]
SLICE_WORDS = ["截", "截取", "裁剪", "切片", "只看", "只保留", "slice", "crop", "cut", "keep"]
EXPORT_WORDS = ["导出", "保存", "下载", "输出", "export", "save", "download", "output"]
LIST_WORDS = ["列出", "有哪些", "哪些文件", "文件列表", "list", "show files"]
BATCH_WORDS = ["批量加载", "批量读取", "加载所有", "加载多个", "batch load", "load all", "load batch"]
COMPARE_WORDS = ["对比", "比较", "对比图", "比较图", "放在一起", "叠加", "compare", "overlay", "diff"]
GLOB_PATTERN = re.compile(r'["\']?([\w*?./\\-]+\.s\dp)["\']?', re.IGNORECASE)

CHART_DB = ["db", "dB", "分贝", "幅度", "增益", "损耗", "magnitude", "插损", "回损"]
CHART_DEG = ["相位", "角度", "phase", "deg", "degree", "angle"]
CHART_SMITH = ["smith", "史密斯", "圆图", "史密斯圆图", "反射系数"]
CHART_VSWR = ["vswr", "驻波", "驻波比", "电压驻波比"]
CHART_GD = ["群时延", "群延迟", "group delay", "groupdelay", "gd"]
CHART_MAG = ["线性", "幅度线性", "mag", "linear"]

DUAL_AXIS_WORDS = [
    "两个坐标", "双y轴", "双轴", "双纵轴", "左右轴",
    "两个y轴", "两个纵轴", "dual axis", "twin axis",
    "双坐标系", "两个坐标系", "左右坐标",
]

PARAM_PATTERN = re.compile(r"[sS](\d)(\d)", re.IGNORECASE)
VSWR_PARAM_PATTERN = re.compile(r"[vV][sS][wW][rR](\d+)")
FREQ_RANGE_PATTERN = re.compile(
    r"(\d+\.?\d*)\s*(?:-|到|至|~)\s*(\d+\.?\d*)\s*([gG][hH][zZ]?|[mM][hH][zZ]?|[kK][hH][zZ]?)?"
)
FREQ_SINGLE_PATTERN = re.compile(r"(\d+\.?\d*)\s*([gG][hH][zZ]?|[mM][hH][zZ]?)")

SNP_PATTERN = re.compile(r"([\w/\\.-]+\.s\dp)", re.IGNORECASE)
FILE_PATTERN = re.compile(r"([\w/\\.-]+\.[\w]+)")

# ──────────────────────────────────────────────
#  解析主函数
# ──────────────────────────────────────────────


def parse(text: str, available_files: List[str] = None) -> List[SParamOp]:
    """
    解析整句自然语言，返回操作列表。

    示例：
        parse("读取 amp.s2p，画 S11 和 S21 的 dB 图")
        → [SParamOp(action='load', target='amp.s2p'),
           SParamOp(action='plot', params=['S11','S21'], chart_type='db')]

        parse("把 LNA 和 BPF 级联，画 S21 Smith 圆图")
        → [SParamOp(action='cascade', target='LNA', cascade_with='BPF'),
           SParamOp(action='plot', params=['S21'], chart_type='smith')]

        parse("只看 2-4GHz，导出 S21 为 CSV")
        → [SParamOp(action='slice', freq_range=(2e9,4e9)),
           SParamOp(action='export', params=['S21'], export_format='csv')]
    """
    ops: List[SParamOp] = []

    # ── 分段：按 "然后"、"再"、"并"、"；"、";" 拆分 ──
    segments = _split_segments(text)

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        op = _parse_segment(seg, available_files)
        if op:
            ops.append(op)

    # ── 后处理：如果操作缺少 target，从上一个操作继承 ──
    for i in range(1, len(ops)):
        if not ops[i].target and ops[i].action in ("plot", "slice", "export", "info"):
            ops[i].target = ops[i - 1].target

    # ── 后处理：跨段双Y轴检测 ──
    # 扫描所有段，如果一个段没有产生 op 但包含双Y轴关键词，标记最近的 plot op
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        seg_lower = seg.lower()
        if any(w in seg_lower for w in DUAL_AXIS_WORDS):
            # 找最近的一个 plot op 并标记
            for op in reversed(ops):
                if op.action == "plot":
                    op.dual_axis = True
                    break

    return ops


def _split_segments(text: str) -> List[str]:
    """按连接词拆分句子为子句。"""
    return re.split(r"[，,;；\n]|然后|再|并|并且|接着|之后|同时|、", text)


def _parse_segment(seg: str, available_files: List[str] = None) -> Optional[SParamOp]:
    """解析单段自然语言为一个操作。"""
    seg_lower = seg.lower()

    # ── 检测操作类型 ──
    action = _detect_action(seg_lower)
    if not action:
        return None

    op = SParamOp(action=action)

    # ── 检测链式级联 (A → B → C) ──
    chain_result = _detect_chain(seg)
    if chain_result and len(chain_result) >= 2:
        op.action = "cascade_chain"
        op.chain = chain_result
        op.target = chain_result[0]
        return op

    # ── 检测批量加载 glob 模式 ──
    glob_match = re.search(r'["\']?([\w*?./\\-]+\.s\dp)["\']?', seg)
    if glob_match and any(c in seg for c in "*?[]"):
        op.batch_pattern = glob_match.group(1)

    # ── 提取文件 ──
    file_matches = SNP_PATTERN.findall(seg) or FILE_PATTERN.findall(seg)
    if file_matches:
        # 过滤掉非 Touchstone 的扩展名
        snp_files = [f for f in file_matches if re.match(r".*\.s\dp$", f, re.IGNORECASE)]
        if not snp_files:
            snp_files = [f for f in file_matches if not f.lower().endswith((".csv", ".html", ".png", ".pdf"))]
        if snp_files:
            op.target = snp_files[0]
            if len(snp_files) > 1 and action == "cascade":
                op.cascade_with = snp_files[1]
    elif available_files:
        # 尝试从名称匹配（不带扩展名）
        for f in available_files:
            basename = re.sub(r"\.[^.]+$", "", f)
            if basename.lower() in seg_lower or seg_lower in basename.lower():
                op.target = f
                break

    # ── 提取 S 参数 ──
    param_matches = PARAM_PATTERN.findall(seg)
    if param_matches:
        op.params = [f"S{m}{n}" for m, n in param_matches]

    vswr_matches = VSWR_PARAM_PATTERN.findall(seg)
    if vswr_matches:
        op.params = [f"VSWR{n}" for n in vswr_matches]

    # ── 提取图表类型 ──
    op.chart_type = _detect_chart_type(seg_lower)

    # ── 提取频率范围 ──
    op.freq_range = _extract_freq_range(seg)

    # ── 提取导出格式（仅当 action 为 export 时）──
    if action == "export":
        if "csv" in seg_lower:
            op.export_format = "csv"
        elif "touchstone" in seg_lower or re.search(r"\.s\dp", seg_lower):
            op.export_format = "touchstone"
        elif "html" in seg_lower or "网页" in seg:
            op.export_format = "html"
        else:
            op.export_format = "csv"  # 默认 CSV
        op.export_format = "html"

    # ── 提取标题 ──
    title_match = re.search(r"标题[:：]\s*(.+?)(?:[，,;；\n]|$)", seg)
    if title_match:
        op.title = title_match.group(1).strip()

    # ── 检测双Y轴 ──
    if op.action == "plot" and any(w in seg_lower for w in DUAL_AXIS_WORDS):
        op.dual_axis = True

    return op


# ──────────────────────────────────────────────
#  检测函数
# ──────────────────────────────────────────────


def _detect_action(text: str) -> str:
    # 链式语法检测（→ / ->  / >>）
    if any(sep in text for sep in ["→", "->", ">>"]):
        return "cascade"
    if any(w in text for w in BATCH_WORDS):
        return "load_batch"
    if any(w in text for w in LOAD_WORDS):
        return "load"
    if any(w in text for w in COMPARE_WORDS):
        return "compare"
    if any(w in text for w in CASCADE_WORDS):
        return "cascade"
    if any(w in text for w in DEEMBED_WORDS):
        return "deembed"
    if any(w in text for w in SLICE_WORDS):
        return "slice"
    if any(w in text for w in PLOT_WORDS):
        return "plot"
    if any(w in text for w in EXPORT_WORDS):
        return "export"
    if any(w in text for w in INFO_WORDS):
        return "info"
    if any(w in text for w in LIST_WORDS):
        return "list"
    return ""


def _detect_chain(text: str) -> Optional[List[str]]:
    """
    检测链式级联语法: LNA → BPF → AMP 或 LNA -> BPF -> AMP
    返回网络名称列表。
    """
    # 尝试各种链式分隔符
    for sep in ["→", "->", ">>"]:
        if sep in text:
            parts = [p.strip() for p in text.split(sep)]
            names = []
            for p in parts:
                p = p.strip()
                # 移除末尾的扩展名
                p = re.sub(r"\.s\dp$", "", p, flags=re.IGNORECASE)
                # 取第一个词（网络名通常是连续的字母/数字/下划线）
                # 过滤掉纯中文描述部分
                word_match = re.match(r"([A-Za-z0-9_]+)", p)
                if word_match:
                    names.append(word_match.group(1))
                elif p and not re.search(r"[，。；！？、\s]", p[:3]) and len(p) < 30:
                    # 没有空格/标点的短文本，可能是中文名或数字名
                    names.append(p)
            if len(names) >= 2:
                return names
    return None


def _detect_chart_type(text: str) -> str:
    if any(w in text for w in CHART_SMITH):
        return "smith"
    if any(w in text for w in CHART_VSWR):
        return "vswr"
    if any(w in text for w in CHART_GD):
        return "groupdelay"
    if any(w in text for w in CHART_DEG):
        return "deg"
    if any(w in text for w in CHART_MAG):
        return "mag"
    if any(w in text for w in CHART_DB):
        return "db"
    return "db"  # 默认幅度


def _extract_freq_range(text: str) -> Optional[Tuple[float, float]]:
    """提取 '2-4GHz'、'1到3G' 这样的频率范围，返回 (start_Hz, stop_Hz)。"""
    # 模式：数字-数字+单位
    m = FREQ_RANGE_PATTERN.search(text)
    if m:
        start_val = float(m.group(1))
        stop_val = float(m.group(2))
        unit = (m.group(3) or "g").lower()
        multiplier = _unit_multiplier(unit)
        return (start_val * multiplier, stop_val * multiplier)

    # 模式："低于X"、"高于Y"、"X以上"
    below = re.search(r"(?:低于|小于|<|below|under)\s*(\d+\.?\d*)\s*([gGmMkK]?)", text)
    if below:
        val = float(below.group(1))
        unit = below.group(2) or "g"
        return (0, val * _unit_multiplier(unit))

    above = re.search(r"(?:高于|大于|>|above|over)\s*(\d+\.?\d*)\s*([gGmMkK]?)", text)
    if above:
        val = float(above.group(1))
        unit = above.group(2) or "g"
        return (val * _unit_multiplier(unit), float("inf"))

    return None


def _unit_multiplier(unit: str) -> float:
    u = unit.lower().rstrip("hz")
    if u in ("g", "ghz"):
        return 1e9
    if u in ("m", "mhz"):
        return 1e6
    if u in ("k", "khz"):
        return 1e3
    return 1e9  # 默认 GHz


# ──────────────────────────────────────────────
#  格式化回复
# ──────────────────────────────────────────────


def format_ops(ops: List[SParamOp]) -> str:
    """将操作列表格式化为可读的描述。"""
    lines = ["解析到以下操作："]
    for i, op in enumerate(ops, 1):
        desc = _describe_op(op)
        lines.append(f"  {i}. {desc}")
    return "\n".join(lines)


def _describe_op(op: SParamOp) -> str:
    parts = [f"[{op.action}]"]
    if op.target:
        parts.append(op.target)
    if op.params:
        parts.append(", ".join(op.params))
    if op.chart_type and op.action in ("plot", "compare"):
        tag = op.chart_type
        if op.dual_axis:
            tag += "|dualY"
        parts.append(f"({tag})")
    if op.freq_range:
        f0 = op.freq_range[0]
        f1 = op.freq_range[1]
        parts.append(f"{f0/1e9:.2f}–{f1/1e9:.2f} GHz")
    if op.cascade_with:
        parts.append(f"+ {op.cascade_with}")
    if op.chain:
        parts.append(" → ".join(op.chain))
    if op.compare_networks:
        parts.append("vs " + ", ".join(op.compare_networks))
    if op.batch_pattern:
        parts.append(f"glob:{op.batch_pattern}")
    if op.reference:
        parts.append(f"ref:{op.reference}")
    if op.export_format:
        parts.append(f"→ {op.export_format}")
    return " ".join(parts)


# ──────────────────────────────────────────────
#  测试
# ──────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "读取 amp.s2p，画 S11 和 S21 的 dB 图",
        "看看 filter.s2p 的基本信息",
        "画 S11 的 Smith 圆图，只看 2-4GHz",
        "把 LNA.s2p 和 BPF.s2p 级联，然后画 S21 相位",
        "导出 S21 为 CSV",
        "画 S11 VSWR ，1到6G",
        "加载 amp.s2p，画 S21 群时延",
        # 新增：链式级联
        "LNA → BPF → AMP 级联，画 S21 dB",
        "LNA -> BPF -> AMP 然后画 S21 smith",
        # 新增：批量加载
        "批量加载 data/*.s2p",
        "加载所有 data/ 下的文件",
        # 新增：多文件对比
        "对比 LNA 和 BPF 的 S21",
        "画 LNA BPF AMP 的 S21 比较图",
    ]
    for t in tests:
        print(f"\n输入: {t}")
        ops = parse(t)
        print(format_ops(ops))
