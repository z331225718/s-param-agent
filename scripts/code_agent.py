#!/usr/bin/env python3
"""
强约束代码生成 Agent。
流程: 用户自然语言 → LLM 生成受限 Python 代码 → AST 校验 → 沙箱执行 → 返回结果

约束:
  - LLM 只能输出 skrf + plotly + numpy 代码
  - 所有 import 被 AST 校验白名单过滤
  - 危险内置函数 (eval/exec/__import__) 被拦截
  - 执行有超时限制 (15s)
  - 执行结果中的 plotly Figure 被自动序列化返回
"""

import os
import ast
import sys
import io
import json
import re
import traceback
import contextlib
import importlib
from typing import Optional, Tuple

# ── PyInstaller 兼容：定位资源目录 ──
def _base_dir():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

_BASE = _base_dir()

# ── 白名单 ─────────────────────────────────────────────────────

ALLOWED_IMPORTS = {
    "skrf",
    "numpy",
    "plotly.graph_objects",
    "plotly.subplots",
    "plotly",
    "json",
    "textwrap",
    "itertools",
    "functools",
    "collections",
    "dataclasses",
    "typing",
    "copy",
}

# skrf 的全量子模块白名单（常见且安全）
ALLOWED_PREFIXES = [
    "skrf",
    "numpy",
    "plotly",
    "json",
    "math",
    "textwrap",
    "itertools",
    "functools",
    "collections",
    "dataclasses",
    "typing",
    "copy",
    "re",
    "string",
    "datetime",
    "enum",
]

FORBIDDEN_BUILTINS = {
    "eval", "exec", "compile", "__import__", "open",
    "breakpoint", "input", "getattr", "setattr", "delattr",
    "globals", "locals", "vars",
}

FORBIDDEN_FILE_METHODS = {
    "read_text", "write_text", "read_bytes", "write_bytes", "open",
    "mkdir", "unlink", "rename", "replace", "rmdir", "glob", "rglob",
    "iterdir", "touch", "symlink_to", "hardlink_to",
}

# 无条件物理删除的危险调用（在 AST 校验前用正则移除）
import re as _re
_DANGEROUS_PATTERNS = [
    (r'^\s*fig\.show\s*\(\s*\)', '# [removed] fig.show()'),
    (r'^\s*fig\.write_html\s*\(', '# [removed] fig.write_html()'),
    (r'^\s*fig\.write_image\s*\(', '# [removed] fig.write_image()'),
    (r'^\s*plt\.show\s*\(\s*\)', '# [removed] plt.show()'),
]

# 允许 open() 写入的扩展名
ALLOWED_OPEN_EXTENSIONS = {".s1p", ".s2p", ".s3p", ".s4p", ".sNp",
                           ".csv", ".tsv", ".html", ".png", ".pdf", ".svg",
                           ".json", ".txt", ".md", ".log", ".dat", ".touchstone"}


# ── System Prompt ──────────────────────────────────────────────

import api_refs
import lessons as _lessons
import agent_plan

_SYSTEM_PROMPT_BASE = """你是 RF/微波工程的 Python 代码生成助手。你的唯一任务是：根据用户的自然语言描述，生成一段可执行的 Python 代码来操作 S 参数文件。

## 严格规则

### 网络对象获取
**禁止使用 `rf.Network(path)` 读取文件！** 网络对象已预加载在 `_nets` 字典中。
从 `_nets["名称"]` 直接获取，无需任何路径。网络元数据在 `_meta` 字典中。
```python
# ✅ 正确
ntwk = _nets["LNA"]
meta = _meta["LNA"]
# ❌ 错误
ntwk = rf.Network("/path/to/LNA.s2p")
```

### 允许的 import
你 **只能** 使用以下库，任何其他 import 将被拒绝执行：
```python
import skrf as rf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```

### ⚠️ 铁律（违反必错）
- **先按固定骨架写代码**：
  1. 选择网络：如果用户没有点名，优先用提示中的默认网络。
  2. 做 RF 变换：只在用户要求时做差分/共模/级联/重归一化等。
  3. 只提取用户指定的数据曲线；没有“全部/所有/all”时不要遍历所有端口。
  4. 用 Plotly `go.Figure()` 或 `make_subplots()` 生成 `fig`。
- **频率必须转 GHz**：`ntwk.f` 单位是 Hz，直接用会出现 1B/2B 的丑陋标签。
  所有 X 轴必须用 `freq_ghz = ntwk.f / 1e9`，然后 `x=freq_ghz`。
  同时设 `xaxis_title='Frequency (GHz)'`。
- **S/Z/Y 参数是 3D 数组**：必须用 `[:, m, n]` 索引，不是 `[m, n]`！
- **默认用线性频率轴**，仅当用户明确要求"对数轴"/"log"时才用 `type='log'`，按此规则：
  ```python
  fig.update_xaxes(
      type='log',
      dtick=1,
      showgrid=True, gridcolor='#2a2a4a', zerolinecolor='#444',
      minor=dict(showgrid=True, gridcolor='#2a2a4a', griddash='dash'),
  )
  fig.update_yaxes(gridcolor='#2a2a4a', zerolinecolor='#444')
  ```
- **用户指定参数时只画指定参数**：如果用户说 SDD11/S21/Z11，只能画这些参数，禁止自动展开全部 S 参数。
- **混模/差分参数**：用户说 SDD11、SDC1_1、SCD2_1、SCC1_1 时，必须先使用 `_meta[name]["mixed_mode"]` 的 P/N 端口映射，不能猜默认相邻配对：
  ```python
  mixed = _meta["LNA"].get("mixed_mode", {})
  if mixed.get("status") != "ready":
      raise ValueError("差分 P/N 端口未确认，不能生成权威混模结果")
  order = mixed["se2gmm_order"]  # 单端口重排为 [d0_p, d0_n, d1_p, d1_n, ...]
  mm = ntwk.copy()
  if order != list(range(len(order))):
      mm.renumber(order, list(range(len(order))))
  mm.se2gmm(p=mixed["pair_count"])
  db = mm.s_db[:, 0, 0]          # SDD11
  ```
  转换后端口顺序是差分端口在前、共模端口在后；SDDmn 用差分-差分块。
- **端口和网络元数据**：优先参考 `_meta[name]["mixed_mode"]`、`_meta[name]["port_names"]`、`_meta[name]["port_pairs"]`、`_meta[name]["network_kind"]`。`port_pairs` 是单端传输路径，不是 P/N 差分对；P/N 只看 `mixed_mode.differential_pairs`。

### 禁止
- 不要 import os, sys, subprocess, requests, urllib, shutil
- 不要使用 eval(), exec(), __import__()
- 不要写死绝对路径
- 不要调用 fig.show() 或 fig.write_html()
- 不要使用 matplotlib、pyplot、plt、pylab 或类似 `plt.plot(...)` 的风格
- **不要直接用 ntwk.f 作为 X 轴数据！必须先除以 1e9！**
- **不要使用 rf.Network() 读取文件！用 _nets 字典！**

### 画图风格规范（必须严格遵循，与内置按钮图表完全一致）
```python
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=freq_ghz, y=db, mode='lines', name='S11 (filter)',
    hovertemplate='<b>S11 (filter)</b><br>%{x:.4f} GHz<br>%{y:.3f} dB<extra></extra>',
))

fig.update_layout(
    # 暗色主题（与按钮图表一致，不要用 plotly_white）
    paper_bgcolor='#1a1a2e',
    plot_bgcolor='#16213e',
    font=dict(color='#c0c0c0'),
    # 不要设 width/height，让图表响应式撑满容器
    hovermode='closest',
    margin=dict(l=60, r=30, t=60, b=50),
    title=dict(text='S-Parameter Magnitude', font=dict(color='#e0e0e0')),
    xaxis=dict(
        title='Frequency (GHz)',
        gridcolor='#2a2a4a', zerolinecolor='#444',
    ),
    yaxis=dict(
        title='Magnitude (dB)',
        gridcolor='#2a2a4a', zerolinecolor='#444',
    ),
)
```

### 输出格式
只输出代码，放在 ```python 代码块中。不要解释。
"""

def _build_full_system_prompt() -> str:
    prompt = _SYSTEM_PROMPT_BASE + "\n" + api_refs.build_api_prompt()
    lessons_prompt = _lessons.build_lessons_prompt(max_items=3)
    if lessons_prompt:
        prompt += "\n\n" + lessons_prompt
    prompt += """

## 最后硬约束（优先级最高）
- 输出必须是 Plotly `graph_objects` 代码，最终变量必须是 `fig = go.Figure(...)` 或 `fig = make_subplots(...)`。
- 禁止 matplotlib/pyplot/plt/pylab；不要写 `plt.plot`、`ax.plot`、`fig, ax = ...` 这种风格。
- 用户明确指定参数时，只画指定参数；绝不因为转换差分/混模而画全部 S 参数。
- 用户要求 SDD11 时，必须先按 `_meta[name]["mixed_mode"]["se2gmm_order"]` 重排 P/N 端口并调用 `se2gmm(p=pair_count)`，只取转换后 `mm.s_db[:, 0, 0]` 这一条曲线。
"""
    return prompt


_TOKEN_LEFT = r"(?<![A-Z0-9_])"
_TOKEN_RIGHT = r"(?![A-Z0-9_])"
_MIXED_PARAM_RE = re.compile(
    _TOKEN_LEFT + r"S(?:DD|DC|CD|CC)(?:\d+_\d+|[1-9][1-9])" + _TOKEN_RIGHT,
    re.IGNORECASE,
)
_EXPLICIT_PARAM_RE = re.compile(
    _TOKEN_LEFT + r"(?:S(?:DD|DC|CD|CC)(?:\d+_\d+|[1-9][1-9])|[SZY](?:\d+_\d+|[1-9][1-9])|VSWR\d+)" + _TOKEN_RIGHT,
    re.IGNORECASE,
)
_ALL_PARAM_WORDS = ("全部", "所有", "全参数", "all", "every", "each")
_ONLY_PARAM_WORDS = ("只画", "只看", "仅", "只要", "只输出", "only")
_BROAD_PORT_LOOP_RE = re.compile(
    r"for\s+\w+\s+in\s+range\([^)]*(?:\.nports|\bnports\b)[^)]*\)",
    re.IGNORECASE,
)


def _extract_requested_params(user_text: str) -> list:
    params = []
    for match in _EXPLICIT_PARAM_RE.findall(user_text or ""):
        value = match.upper()
        if value not in params:
            params.append(value)
    return params


def _extract_mixed_params(user_text: str) -> list:
    params = []
    for match in _MIXED_PARAM_RE.findall(user_text or ""):
        value = match.upper()
        if value not in params:
            params.append(value)
    return params


def _wants_all_params(user_text: str) -> bool:
    lower = (user_text or "").lower()
    return any(word in lower for word in _ALL_PARAM_WORDS)


def _wants_only_requested_params(user_text: str) -> bool:
    text = user_text or ""
    lower = text.lower()
    return any(word in lower for word in _ONLY_PARAM_WORDS)


def _has_method_call(code: str, method_name: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == method_name:
                return True
    return False


def _build_task_constraints(user_text: str) -> str:
    params = _extract_requested_params(user_text)
    mixed_params = _extract_mixed_params(user_text)
    lines = ["必须保持的语义约束："]
    if params and not _wants_all_params(user_text):
        lines.append(f"- 用户明确指定参数: {', '.join(params)}；只能画这些参数，不要展开全部 S 参数。")
    if mixed_params:
        lines.append("- 用户要求混模/差分参数：必须使用 `_meta[name]['mixed_mode']` 中确认过的 P/N 映射，按 `se2gmm_order` 重排后再调用 `se2gmm(...)`，不要改画单端 S 参数。")
        if "SDD11" in mixed_params:
            lines.append("- SDD11 对应混模网络的 `mm.s_db[:, 0, 0]`；图中只需要这一条曲线。")
    if _wants_all_params(user_text):
        lines.append("- 用户明确要求全部/所有参数，可以遍历端口。")
    if len(lines) == 1:
        lines.append("- 按用户原始需求生成图，不要添加用户没有要求的曲线或分析。")
    return "\n".join(lines)


def _format_list(values, limit=8) -> str:
    if not values:
        return ""
    items = [str(v) for v in values[:limit]]
    if len(values) > limit:
        items.append(f"... 共{len(values)}项")
    return ", ".join(items)


def _build_network_context(networks: dict = None, file_path: str = None) -> str:
    if not networks and file_path:
        networks = {"current": {"path": file_path, "source": "file_path"}}
    if not networks:
        return "\n当前没有已加载网络。"

    names = list(networks.keys())
    default_name = names[-1]
    lines = [
        "\n当前已加载的网络：",
        f"- 默认网络: _nets[\"{default_name}\"]（如果用户没有点名网络，就用它）",
        "- 运行时可用变量: `_nets` 是网络对象字典，`_meta` 是同名元数据字典。",
    ]
    for name, info in networks.items():
        nports = info.get("nports", "?")
        npoints = info.get("npoints", "?")
        f_min = info.get("f_min")
        f_max = info.get("f_max")
        freq = ""
        if f_min is not None and f_max is not None:
            try:
                freq = f", freq={float(f_min)/1e9:.4g}-{float(f_max)/1e9:.4g}GHz"
            except Exception:
                freq = ""
        loaded = info.get("loaded")
        loaded_text = "" if loaded is None else f", loaded={bool(loaded)}"
        kind = info.get("network_kind", "unknown")
        lines.append(f"- _nets[\"{name}\"]: nports={nports}, npoints={npoints}{freq}, kind={kind}{loaded_text}")
        port_names = info.get("port_names") or []
        if port_names:
            lines.append(f"  port_names: {_format_list(port_names)}")
        port_pairs = info.get("port_pairs") or []
        if port_pairs:
            lines.append(f"  inferred through-path port_pairs (不是 P/N 差分对): {_format_list(port_pairs, limit=5)}")
        mixed_mode = info.get("mixed_mode") or {}
        if mixed_mode:
            status = mixed_mode.get("status", "unknown")
            order = mixed_mode.get("se2gmm_order") or []
            pair_count = mixed_mode.get("pair_count", 0)
            lines.append(f"  mixed_mode: status={status}, pair_count={pair_count}, se2gmm_order={order}")
            diff_pairs = mixed_mode.get("differential_pairs") or []
            if diff_pairs:
                lines.append(f"  mixed_mode differential_pairs: {_format_list(diff_pairs, limit=4)}")
            alternatives = mixed_mode.get("alternatives") or []
            if alternatives:
                lines.append(f"  mixed_mode alternatives need user confirmation: {_format_list(alternatives, limit=3)}")
        quick_actions = info.get("quick_actions") or []
        if quick_actions:
            action_bits = []
            for action in quick_actions[:5]:
                action_bits.append(f"{action.get('label', action.get('id', '?'))}:{_format_list(action.get('params', []), 5)}")
            lines.append(f"  recommended actions: {'; '.join(action_bits)}")
    lines.append("**禁止用 rf.Network(path) 重新读取文件；直接使用 `_nets[...]`。**")
    return "\n".join(lines)


def _needs_mixed_mode_renumber(networks: dict = None) -> bool:
    for info in (networks or {}).values():
        mixed = info.get("mixed_mode") or {}
        if mixed.get("status") != "ready":
            continue
        order = mixed.get("se2gmm_order") or []
        if order and order != list(range(len(order))):
            return True
    return False


def _code_mentions_plan_param(code: str, param: str) -> bool:
    compact = param.replace("_", "")
    if param in code or compact in code:
        return True
    match = re.match(r"^(?:S(?:DD|DC|CD|CC)?|[SZY])(\d+)_(\d+)$", param)
    if not match:
        return False
    m = int(match.group(1)) - 1
    n = int(match.group(2)) - 1
    index_pat = re.compile(r"\[:,\s*" + str(m) + r"\s*,\s*" + str(n) + r"\s*\]")
    return index_pat.search(code) is not None


def validate_code_semantics(code: str, user_text: str, networks: dict = None, plan: dict = None) -> Tuple[bool, str]:
    """校验生成代码是否明显违背用户语义。"""
    params = _extract_requested_params(user_text)
    mixed_params = _extract_mixed_params(user_text)
    if plan:
        for name in agent_plan.selected_networks(plan):
            if f'_nets["{name}"]' not in code and f"_nets['{name}']" not in code:
                return False, f"代码没有使用 Plan 指定的网络: {name}"
    if (mixed_params or agent_plan.plan_has_op(plan, "mixed_mode")) and not _has_method_call(code, "se2gmm"):
        return False, f"用户要求混模/差分参数 {', '.join(mixed_params)}，代码必须调用 se2gmm()"
    if mixed_params and _needs_mixed_mode_renumber(networks) and not _has_method_call(code, "renumber"):
        return False, "已确认的 P/N 端口顺序不是当前顺序，代码必须按 mixed_mode.se2gmm_order 调用 renumber() 后再 se2gmm()"
    plan_params = agent_plan.trace_params(plan)
    if plan_params and not _wants_all_params(user_text):
        for param in plan_params:
            if not _code_mentions_plan_param(code, param):
                return False, f"代码没有体现 Plan 指定参数: {param}"
    if params and not _wants_all_params(user_text):
        if _BROAD_PORT_LOOP_RE.search(code) and "add_trace" in code:
            return False, "用户指定了明确参数，代码却按 nports 遍历画图，可能会画出全部 S 参数"
    if _wants_only_requested_params(user_text) and params:
        if "range(ntwk.nports)" in code or "range(mm.nports)" in code:
            return False, "用户要求只画指定参数，代码不能遍历所有端口"
    return True, "OK"


def validate_figure_semantics(user_text: str, figure_json: dict, network_count: int = 1, plan: dict = None) -> Tuple[bool, str]:
    """校验执行结果是否明显违背用户语义。"""
    if not figure_json:
        return True, "OK"
    params = _extract_requested_params(user_text)
    mixed_params = _extract_mixed_params(user_text)
    if not params or _wants_all_params(user_text):
        expected = agent_plan.expected_trace_count(plan)
        if expected is None:
            return True, "OK"
        data = figure_json.get("data", []) or []
        if len(data) != expected:
            return False, f"Plan 期望 {expected} 条曲线，但图里有 {len(data)} 条"
        return True, "OK"
    data = figure_json.get("data", []) or []
    expected = agent_plan.expected_trace_count(plan)
    if expected is not None and len(data) != expected:
        return False, f"Plan 期望 {expected} 条曲线，但图里有 {len(data)} 条"
    max_expected_traces = max(1, int(network_count or 1)) * max(1, len(params))
    if mixed_params and len(data) > max_expected_traces:
        return False, (
            f"用户只指定 {', '.join(mixed_params)}，但图里有 {len(data)} 条曲线；"
            "不要展开全部 S 参数"
        )
    if _wants_only_requested_params(user_text) and len(data) > max_expected_traces:
        return False, (
            f"用户只指定 {', '.join(params)}，但图里有 {len(data)} 条曲线；"
            "只能画指定参数"
        )
    return True, "OK"


# ── AST 校验器 ─────────────────────────────────────────────────

class CodeValidator(ast.NodeVisitor):
    """遍历 AST，检查所有 import 和危险调用。"""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.name
            if not self._is_allowed(name):
                self.errors.append(f"禁止 import: '{name}'（仅允许 skrf, plotly, numpy 等）")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ""
        full = module + "." + node.names[0].name if module else node.names[0].name
        # 检查顶层模块
        top = module.split(".")[0] if module else node.names[0].name
        if not self._is_allowed(top) and not self._is_allowed(module):
            self.errors.append(f"禁止 import from: '{module}'（仅允许 skrf, plotly, numpy 等）")
        if module == "skrf":
            for alias in node.names:
                if alias.name == "Network":
                    self.errors.append("禁止 import: skrf.Network（请使用 _nets 中已加载的网络）")
        self.generic_visit(node)

    def visit_Call(self, node):
        # 检查危险函数调用
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_BUILTINS:
                self.errors.append(f"禁止调用: {node.func.id}()")
            if node.func.id == "Network":
                self.errors.append("禁止调用: Network()（请使用 _nets 中已加载的网络）")
        if isinstance(node.func, ast.Attribute):
            full = self._get_attr_chain(node.func)
            if full in ("rf.Network", "skrf.Network"):
                self.errors.append("禁止调用: rf.Network()（请使用 _nets 中已加载的网络）")
            if node.func.attr in FORBIDDEN_FILE_METHODS:
                self.errors.append(f"禁止调用文件方法: {node.func.attr}()")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # 检查是否通过属性访问危险模块
        full = self._get_attr_chain(node)
        if full:
            parts = full.split(".")
            if parts[0] in ("os", "subprocess", "sys", "shutil"):
                self.errors.append(f"禁止访问: {full}")
            if parts[0] in ("plt", "matplotlib", "pylab"):
                self.errors.append(f"禁止使用 matplotlib/plt 风格绘图: {full}")
        self.generic_visit(node)

    def _is_allowed(self, name: str) -> bool:
        if name in ALLOWED_IMPORTS:
            return True
        for prefix in ALLOWED_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                return True
        return False

    def _get_attr_chain(self, node) -> Optional[str]:
        """递归构建 os.path.join 这样的属性链。"""
        if isinstance(node, ast.Attribute):
            parent = self._get_attr_chain(node.value)
            if parent:
                return f"{parent}.{node.attr}"
            return node.attr
        elif isinstance(node, ast.Name):
            return node.id
        return None


def validate_code(code: str) -> Tuple[bool, str]:
    """
    校验代码安全性。
    返回: (是否通过, 错误信息)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"语法错误: {e}"

    validator = CodeValidator()
    validator.visit(tree)

    if validator.errors:
        return False, " | ".join(validator.errors)

    if validator.warnings:
        # 警告不阻止执行
        pass

    return True, "OK"


# ── 代码提取 ───────────────────────────────────────────────────

def extract_code(llm_response: str) -> Optional[str]:
    """从 LLM 回复中提取 ```python ... ``` 代码块。"""
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, llm_response, re.DOTALL)
    if matches:
        return "\n".join(matches)
    # fallback: 尝试 ``` 任意语言
    pattern2 = r"```\s*\n(.*?)```"
    matches2 = re.findall(pattern2, llm_response, re.DOTALL)
    if matches2:
        return "\n".join(matches2)
    return None


# ── 对数轴关键词检测 ─────────────────────────────────────────────

_LOG_KEYWORDS = [
    "log scale", "log-scale", "logscale",
    "log freq", "log frequency",
    "对数", "对数轴", "对数坐标", "对数频率",
    "set_xscale", "type='log'", 'type="log"',
]

_LOG_INJECTION = '''
# [injected] log scale format
fig.update_xaxes(
    type='log',
    dtick=1,
    showgrid=True, gridcolor='#2a2a4a', zerolinecolor='#444',
    minor=dict(showgrid=True, gridcolor='#2a2a4a', griddash='dash'),
)
fig.update_yaxes(gridcolor='#2a2a4a', zerolinecolor='#444')
'''


def _user_wants_log_scale(text: str) -> bool:
    """检测用户输入是否要求对数轴。"""
    lower = text.lower()
    return any(kw in lower for kw in _LOG_KEYWORDS)


def _inject_log_scale(code: str) -> str:
    """在代码末尾注入规范对数轴配置（如果还没有的话）。"""
    if "type='log'" in code or 'type="log"' in code or "xaxis_type='log'" in code:
        return code  # 已有 log 配置，不重复注入
    return code.rstrip() + '\n' + _LOG_INJECTION + '\n'


# ── 沙箱执行 ───────────────────────────────────────────────────

def execute_code(code: str, file_paths: dict = None, networks: dict = None, timeout_sec: int = 30) -> dict:
    """
    在子进程中执行代码（用 subprocess 隔离，跨平台安全）。

    Args:
        code: Python 代码字符串
        file_paths: {"file_path": "/path/to/file.s2p"} 映射（向后兼容）
        networks: {"name": {"path": "...", "nports": N}, ...} 预加载到 _nets
        timeout_sec: 超时秒数

    Returns:
        {
            "ok": bool,
            "figure_json": {...} or None,
            "stdout": "...",
            "stderr": "...",
            "error": "..." or None,
        }
    """
    import subprocess
    import tempfile

    # 构建网络预加载代码
    nets_init = ""
    if networks:
        networks_json = json.dumps(networks)
        nets_init = f'''
# ── 预加载网络对象到 _nets ──
import skrf as rf
_nets = {{}}
_networks_config = json.loads({json.dumps(networks_json)})
_meta = _networks_config
for _name, _info in _networks_config.items():
    try:
        _nets[_name] = rf.Network(_info["path"])
    except Exception:
        pass  # 跳过损坏的文件
'''

    # 构建完整的可执行脚本（.format() 命名参数，避免 f-string 吃掉代码中的 {}）
    _user_code = _indent(code, "    ")
    _file_path_json = json.dumps(file_paths.get("file_path", "") if file_paths else "")
    _out_path_json = json.dumps(tempfile.mktemp(suffix=".json"))

    wrapper = '''import sys, io, json, traceback
{nets_init}
# 注入文件路径（兼容旧代码）
file_path = {file_path_json}

# 捕获输出
stdout_buf = io.StringIO()
stderr_buf = io.StringIO()
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr
sys.stdout = stdout_buf
sys.stderr = stderr_buf

result = {{"ok": False, "figure_json": None, "stdout": "", "stderr": "", "error": None}}

try:
{user_code}
    fig = locals().get("fig")
    if fig is not None and hasattr(fig, "to_json"):
        result["figure_json"] = json.loads(fig.to_json())
    result["ok"] = True
except Exception as e:
    result["error"] = str(type(e).__name__) + ": " + str(e) + "\\n" + traceback.format_exc()
finally:
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
    result["stdout"] = stdout_buf.getvalue()
    result["stderr"] = stderr_buf.getvalue()

out_path = {out_path_json}
with open(out_path, "w") as f:
    json.dump(result, f)
print("__RESULT_FILE__:" + out_path)
'''.format(nets_init=nets_init, file_path_json=_file_path_json,
           user_code=_user_code, out_path_json=_out_path_json)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapper],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=os.getcwd(),
        )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        # 查找结果文件路径
        result_file = None
        for line in stdout.split("\n"):
            if line.startswith("__RESULT_FILE__:"):
                result_file = line.split(":", 1)[1].strip()
                break

        if result_file and os.path.exists(result_file):
            with open(result_file, "r") as f:
                result = json.load(f)
            os.unlink(result_file)
            return result

        # 没有结果文件 → 执行失败
        return {
            "ok": False,
            "figure_json": None,
            "stdout": stdout,
            "stderr": stderr,
            "error": f"执行失败（无结果文件）\nstdout: {stdout[-500:]}\nstderr: {stderr[-500:]}",
        }

    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "figure_json": None,
            "stdout": "",
            "stderr": "",
            "error": f"代码执行超时 ({timeout_sec}s)，已被终止",
        }
    except Exception as e:
        return {
            "ok": False,
            "figure_json": None,
            "stdout": "",
            "stderr": "",
            "error": f"执行异常: {e}",
        }


def _indent(code: str, prefix: str) -> str:
    """给每行代码加缩进前缀。"""
    return "\n".join(prefix + line if line.strip() else "" for line in code.split("\n"))


# ── LLM 调用 ───────────────────────────────────────────────────

def _get_llm_config():
    """读取 LLM 配置：config.json 优先（先 scripts/ 再项目根），其次环境变量。"""
    import sys as _sys

    # 候选路径：scripts/config.json → 项目根/config.json
    _candidates = [
        os.path.join(_BASE, "config.json"),
        os.path.join(_BASE, "..", "config.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.json"),
    ]

    for config_path in _candidates:
        config_path = os.path.normpath(config_path)
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f).get("llm", {})
            api_key = cfg.get("api_key", "")
            if not api_key:
                api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
            if api_key:
                return {
                    "api_key": api_key,
                    "base_url": cfg.get("base_url", "https://api.deepseek.com"),
                    "model": cfg.get("model", "deepseek-chat"),
                    "think": cfg.get("think", False),
                    "timeout_sec": cfg.get("timeout_sec", 60),
                }
        except json.JSONDecodeError as e:
            _sys.stderr.write(f"[WARN] config.json 解析失败 ({config_path}): {e}\n")
        except Exception as e:
            _sys.stderr.write(f"[WARN] 读取 config.json 出错 ({config_path}): {e}\n")

    # 2. fallback: 环境变量
    if os.environ.get("DEEPSEEK_API_KEY"):
        return {
            "api_key": os.environ["DEEPSEEK_API_KEY"],
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        }
    elif os.environ.get("OPENAI_API_KEY"):
        return {
            "api_key": os.environ["OPENAI_API_KEY"],
            "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        }
    return None


def is_available() -> bool:
    return _get_llm_config() is not None


MAX_RETRIES = 3
MAX_PLAN_RETRIES = 1


def _call_llm(config: dict, messages: list, max_tokens: int, timeout_sec: int):
    import urllib.request

    payload = {
        "model": config["model"],
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{config['base_url'].rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


def build_rf_plan(user_text: str, networks: dict = None, file_path: str = None, config: dict = None) -> dict:
    """Ask the LLM for a short structured RF Plan, then validate it deterministically."""
    config = config or _get_llm_config()
    if not config:
        return {"ok": False, "error": "未配置 LLM API Key"}
    if file_path and not networks:
        networks = {"current": {"path": file_path, "source": "file_path"}}
    networks = networks or {}

    context = _build_network_context(networks, file_path=file_path)
    _think = config.get("think", False)
    timeout = min(int(config.get("timeout_sec", 120 if _think else 60)), 60)
    messages = agent_plan.build_plan_prompt(user_text, context)
    history = []

    for attempt in range(1, MAX_PLAN_RETRIES + 2):
        try:
            raw = _call_llm(config, messages, max_tokens=1200, timeout_sec=timeout)
            plan = agent_plan.extract_plan_json(raw)
        except Exception as e:
            history.append({"attempt": attempt, "error": f"Plan JSON 解析失败: {e}"})
            if attempt <= MAX_PLAN_RETRIES:
                messages.append({"role": "user", "content": f"上次没有输出合法 JSON: {e}\n请只输出符合 schema 的 JSON。"})
            continue

        validation = agent_plan.validate_plan(plan, networks, user_text=user_text)
        history.append({
            "attempt": attempt,
            "plan": validation.plan or plan,
            "ok": validation.ok,
            "error": validation.error,
            "needs_confirmation": validation.needs_confirmation,
        })
        if validation.needs_confirmation:
            return {
                "ok": False,
                "needs_confirmation": True,
                "reply": validation.confirmation,
                "plan": validation.plan,
                "history": history,
            }
        if validation.ok:
            return {"ok": True, "plan": validation.plan, "history": history}
        if attempt <= MAX_PLAN_RETRIES:
            messages.append({"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)})
            messages.append({"role": "user", "content": f"Plan 校验失败: {validation.error}\n请修正并只输出 JSON。"})

    return {
        "ok": False,
        "error": history[-1]["error"] if history else "Plan 生成失败",
        "history": history,
    }

def generate_code(
    user_text: str,
    file_path: str = None,
    networks: dict = None,
    rf_plan: dict = None,
    planning_history: list = None,
) -> dict:
    """
    完整流程：LLM 生成代码 → 验证 → 执行 → 失败自动纠错（最多 3 次）。

    Args:
        user_text: 用户自然语言
        file_path: 当前会话中的 S 参数文件路径（兼容旧代码）
        networks: {"name": {"path": "...", "nports": N}, ...} 预加载到 _nets 字典
        rf_plan: 已由上层校验通过的 RF Plan；传入后不再重复规划
        planning_history: 上层规划历史，原样透传给前端

    Returns:
        { "code": "...", "validated": bool, "exec_result": {...}, "retries": int, "history": [...] }
    """
    config = _get_llm_config()
    if not config:
        return {"error": "未配置 LLM API Key（DEEPSEEK_API_KEY 或 OPENAI_API_KEY）"}

    if file_path and not networks:
        networks = {"current": {"path": file_path, "source": "file_path"}}

    context = _build_network_context(networks, file_path=file_path)
    task_constraints = _build_task_constraints(user_text)

    if rf_plan is None:
        plan_result = build_rf_plan(user_text, networks=networks, file_path=file_path, config=config)
        if plan_result.get("needs_confirmation"):
            return {
                "needs_confirmation": True,
                "reply": plan_result.get("reply", "需要确认后才能执行"),
                "plan": plan_result.get("plan"),
                "planning_history": plan_result.get("history", []),
            }
        if not plan_result.get("ok"):
            return {
                "error": f"Agent 规划失败: {plan_result.get('error', '未知错误')}",
                "plan": plan_result.get("plan"),
                "planning_history": plan_result.get("history", []),
            }
        rf_plan = plan_result["plan"]
        plan_history = plan_result.get("history", [])
    else:
        validation = agent_plan.validate_plan(rf_plan, networks or {}, user_text=user_text)
        if validation.needs_confirmation:
            return {
                "needs_confirmation": True,
                "reply": validation.confirmation or "需要确认后才能执行",
                "plan": validation.plan,
                "planning_history": planning_history or [],
            }
        if not validation.ok:
            return {
                "error": f"Agent 规划失败: {validation.error}",
                "plan": rf_plan,
                "planning_history": planning_history or [],
            }
        rf_plan = validation.plan
        plan_history = planning_history or []
    plan_json = json.dumps(rf_plan, ensure_ascii=False, indent=2)

    messages = [
        {"role": "system", "content": _build_full_system_prompt()},
        {"role": "user", "content": (
            f"用户原始需求：\n{user_text}\n\n"
            f"{task_constraints}\n"
            f"后端已校验通过的 RF Plan（这是生成代码的唯一权威，不要重新解释用户需求）：\n{plan_json}\n\n"
            f"{context}\n\n"
            "请按固定骨架生成完整 Python 代码。只输出 ```python 代码块。"
        )},
    ]

    history = []
    last_code = ""

    _think = config.get("think", False)
    _timeout = config.get("timeout_sec", 120 if _think else 60)

    for attempt in range(1, MAX_RETRIES + 2):  # 1 初始 + 最多 3 次纠错 = 最多 4 次
        try:
            llm_response = _call_llm(
                config,
                messages,
                max_tokens=8192 if _think else 1500,
                timeout_sec=_timeout,
            )
        except Exception as e:
            return {"error": f"LLM 调用失败: {e}", "retries": attempt - 1, "history": history}

        code = extract_code(llm_response)

        if not code:
            history.append({"attempt": attempt, "error": "LLM 未生成有效代码", "llm": llm_response[:300]})
            continue

        last_code = code

        # 无条件删除危险调用（fig.show/write_html/write_image/plt.show）
        for pattern, replacement in _DANGEROUS_PATTERNS:
            code = _re.sub(pattern, replacement, code, flags=_re.MULTILINE)

        # 用户指定 log scale → 强制注入规范对数轴
        if _user_wants_log_scale(user_text):
            code = _inject_log_scale(code)

        last_code = code  # 同步清理后的代码

        # 校验
        valid, msg = validate_code(code)
        if not valid:
            history.append({"attempt": attempt, "code": code, "error": f"校验失败: {msg}"})
            if attempt <= MAX_RETRIES:
                messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
                messages.append({"role": "user", "content": (
                    f"代码校验未通过: {msg}\n\n"
                    f"用户原始需求必须保持不变：{user_text}\n"
                    f"{task_constraints}\n"
                    "请修正后重新生成完整代码，只输出 ```python 代码块。"
                )})
            continue

        semantic_ok, semantic_msg = validate_code_semantics(code, user_text, networks=networks, plan=rf_plan)
        if not semantic_ok:
            history.append({"attempt": attempt, "code": code, "error": f"语义校验失败: {semantic_msg}"})
            if attempt <= MAX_RETRIES:
                messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
                messages.append({"role": "user", "content": (
                    f"代码语义不符合用户需求: {semantic_msg}\n\n"
                    f"用户原始需求必须保持不变：{user_text}\n"
                    f"{task_constraints}\n"
                    "请重写代码。不要增加用户没有要求的曲线。只输出 ```python 代码块。"
                )})
            continue

        # 执行
        file_paths = {"file_path": file_path} if file_path else {}
        exec_result = execute_code(code, file_paths, networks=networks)

        if exec_result.get("ok") and exec_result.get("figure_json"):
            network_count = len(networks) if networks else 1
            fig_ok, fig_msg = validate_figure_semantics(
                user_text,
                exec_result.get("figure_json"),
                network_count=network_count,
                plan=rf_plan,
            )
            if not fig_ok:
                history.append({"attempt": attempt, "code": code, "error": f"图表语义失败: {fig_msg}"})
                if attempt <= MAX_RETRIES:
                    messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
                    messages.append({"role": "user", "content": (
                        f"代码能执行，但图表不符合用户需求: {fig_msg}\n\n"
                        f"用户原始需求必须保持不变：{user_text}\n"
                        f"{task_constraints}\n"
                        "请修正代码后重新生成。只输出 ```python 代码块。"
                    )})
                continue

            # 成功！
            history.append({"attempt": attempt, "code": code, "ok": True})

            # 纠错成功后自动学习
            if attempt > 1:
                for h in reversed(history[:-1]):
                    if "error" in h:
                        _lessons.learn(h["error"],
                                       wrong_code=h.get("code", ""),
                                       correct_code=code)
                        break

            return {
                "code": code,
                "llm_raw": llm_response,
                "validated": True,
                "validation_msg": "OK",
                "exec_result": exec_result,
                "retries": attempt - 1,
                "plan": rf_plan,
                "planning_history": plan_history,
                "history": history,
            }

        # 执行失败，构建纠错提示
        error_msg = exec_result.get("error", "未知错误")
        history.append({"attempt": attempt, "code": code, "error": error_msg})

        if attempt <= MAX_RETRIES:
            fix_hint = api_refs.build_fix_prompt(error_msg)
            messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            messages.append({"role": "user", "content": (
                f"代码执行出错:\n{error_msg}\n\n{fix_hint}\n\n"
                f"用户原始需求必须保持不变：{user_text}\n"
                f"{task_constraints}\n"
                "请修正代码后重新生成。只输出 ```python 代码块。"
            )})

    # 所有尝试都失败
    return {
        "code": last_code,
        "validated": True,
        "validation_msg": "多次尝试后仍失败",
        "exec_result": {"ok": False, "error": f"经过 {MAX_RETRIES} 次纠错后仍执行失败", "figure_json": None, "stdout": "", "stderr": ""},
        "retries": MAX_RETRIES,
        "plan": rf_plan,
        "planning_history": plan_history,
        "history": history,
    }


# ── 测试 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"LLM 可用: {is_available()}")

    if is_available():
        config = _get_llm_config()
        print(f"Model: {config['model']}")

        # 测试验证器
        print("\n── 校验测试 ──")
        safe_code = """
import skrf as rf
import numpy as np
import plotly.graph_objects as go
ntwk = rf.Network('test.s2p')
fig = go.Figure()
fig.add_trace(go.Scatter(x=ntwk.f/1e9, y=ntwk.s_db[:,0,0]))
"""
        ok, msg = validate_code(safe_code)
        print(f"安全代码: {ok} ({msg})")

        dangerous_code = """
import os
os.system('rm -rf /')
fig = None
"""
        ok, msg = validate_code(dangerous_code)
        print(f"危险代码: {ok} ({msg})")

        eval_code = """
import numpy as np
eval('print(123)')
fig = None
"""
        ok, msg = validate_code(eval_code)
        print(f"eval代码: {ok} ({msg})")
    else:
        print("设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 后可用")
