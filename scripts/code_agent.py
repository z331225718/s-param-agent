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

# ── 白名单 ─────────────────────────────────────────────────────

ALLOWED_IMPORTS = {
    "skrf",
    "numpy",
    "plotly.graph_objects",
    "plotly.subplots",
    "plotly",
    "json",           # 仅用于 JSON 序列化（安全）
    "math",           # 数学函数
    "textwrap",       # 文本处理
    "itertools",
    "functools",
    "collections",
    "dataclasses",
    "typing",
    "pathlib",
    "io",
    "tempfile",
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
    "pathlib",
    "io",
    "tempfile",
    "copy",
    "re",
    "string",
    "datetime",
    "enum",
]

FORBIDDEN_BUILTINS = {
    "eval", "exec", "compile", "__import__", "open",
    "breakpoint", "input",
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

_SYSTEM_PROMPT_BASE = """你是 RF/微波工程的 Python 代码生成助手。你的唯一任务是：根据用户的自然语言描述，生成一段可执行的 Python 代码来操作 S 参数文件。

## 严格规则

### 允许的 import
你 **只能** 使用以下库，任何其他 import 将被拒绝执行：
```python
import skrf as rf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```

### ⚠️ 铁律（违反必错）
- **频率必须转 GHz**：`ntwk.f` 单位是 Hz，直接用会出现 1B/2B 的丑陋标签。
  所有 X 轴必须用 `freq_ghz = ntwk.f / 1e9`，然后 `x=freq_ghz`。
  同时设 `xaxis_title='Frequency (GHz)'`。
- **S/Z/Y 参数是 3D 数组**：必须用 `[:, m, n]` 索引，不是 `[m, n]`！
- **禁止 Plotly 自动 SI 前缀**：每张图必须加这两行，强制科学计数法：
  ```python
  fig.update_xaxes(exponentformat='power', showexponent='all')
  fig.update_yaxes(exponentformat='power', showexponent='all')
  ```
  否则会出现 μ (micro)、k (kilo)、B (billion) 等丑陋标签！

### 禁止
- 不要 import os, sys, subprocess, requests, urllib, shutil
- 不要使用 eval(), exec(), __import__()
- 不要写死绝对路径
- 不要调用 fig.show() 或 fig.write_html()
- **不要直接用 ntwk.f 作为 X 轴数据！必须先除以 1e9！**

### 输出格式
只输出代码，放在 ```python 代码块中。不要解释。
"""

def _build_full_system_prompt() -> str:
    prompt = _SYSTEM_PROMPT_BASE + "\n" + api_refs.build_api_prompt()
    lessons_prompt = _lessons.build_lessons_prompt(max_items=10)
    if lessons_prompt:
        prompt += "\n\n" + lessons_prompt
    return prompt


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
        self.generic_visit(node)

    def visit_Call(self, node):
        # 检查危险函数调用
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_BUILTINS:
                self.errors.append(f"禁止调用: {node.func.id}()")
        # 检查 open() 的文件扩展名
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            if node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    ext = os.path.splitext(first.value)[1].lower()
                    if ext not in ALLOWED_OPEN_EXTENSIONS:
                        self.warnings.append(f"open() 写入不可识别扩展名: '{ext}'")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # 检查是否通过属性访问危险模块
        full = self._get_attr_chain(node)
        if full:
            parts = full.split(".")
            if parts[0] in ("os", "subprocess", "sys", "shutil"):
                self.errors.append(f"禁止访问: {full}")
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


# ── 沙箱执行 ───────────────────────────────────────────────────

def execute_code(code: str, file_paths: dict = None, timeout_sec: int = 15) -> dict:
    """
    在子进程中执行代码（用 subprocess 隔离，跨平台安全）。

    Args:
        code: Python 代码字符串
        file_paths: {"file_path": "/path/to/file.s2p"} 映射
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

    # 构建完整的可执行脚本
    wrapper = f'''
import sys, io, json, traceback, os

# 注入文件路径
file_path = {json.dumps(file_paths.get("file_path", "") if file_paths else "")}

# 捕获输出
stdout_buf = io.StringIO()
stderr_buf = io.StringIO()
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr
sys.stdout = stdout_buf
sys.stderr = stderr_buf

result = {{"ok": False, "figure_json": None, "stdout": "", "stderr": "", "error": None}}

try:
{_indent(code, "    ")}

    # 提取 fig 变量
    fig = locals().get("fig")
    if fig is not None and hasattr(fig, "to_json"):
        result["figure_json"] = json.loads(fig.to_json())
    result["ok"] = True
except Exception as e:
    result["error"] = f"{{type(e).__name__}}: {{e}}\\n{{traceback.format_exc()}}"
finally:
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
    result["stdout"] = stdout_buf.getvalue()
    result["stderr"] = stderr_buf.getvalue()

# 输出 JSON 结果到临时文件
out_path = {json.dumps(tempfile.mktemp(suffix=".json"))}
with open(out_path, "w") as f:
    json.dump(result, f)
print("__RESULT_FILE__:" + out_path)
'''

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
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        os.path.join(_script_dir, "config.json"),
        os.path.join(_script_dir, "..", "config.json"),
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

def generate_code(user_text: str, file_path: str = None) -> dict:
    """
    完整流程：LLM 生成代码 → 验证 → 执行 → 失败自动纠错（最多 3 次）。

    Args:
        user_text: 用户自然语言
        file_path: 当前会话中的 S 参数文件路径（可选）

    Returns:
        { "code": "...", "validated": bool, "exec_result": {...}, "retries": int, "history": [...] }
    """
    config = _get_llm_config()
    if not config:
        return {"error": "未配置 LLM API Key（DEEPSEEK_API_KEY 或 OPENAI_API_KEY）"}

    context = ""
    if file_path:
        context = f"\n当前已加载的文件路径: {file_path}\n请用这个路径读取文件。"

    import urllib.request

    messages = [
        {"role": "system", "content": _build_full_system_prompt()},
        {"role": "user", "content": f"{user_text}{context}"},
    ]

    history = []
    last_code = ""

    for attempt in range(1, MAX_RETRIES + 2):  # 1 初始 + 最多 3 次纠错 = 最多 4 次
        payload = {
            "model": config["model"],
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1500,
        }

        try:
            req = urllib.request.Request(
                f"{config['base_url'].rstrip('/')}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {config['api_key']}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"error": f"LLM 调用失败: {e}", "retries": attempt - 1, "history": history}

        llm_response = result["choices"][0]["message"]["content"]
        code = extract_code(llm_response)

        if not code:
            history.append({"attempt": attempt, "error": "LLM 未生成有效代码", "llm": llm_response[:300]})
            continue

        last_code = code

        # 无条件删除危险调用（fig.show/write_html/write_image/plt.show）
        for pattern, replacement in _DANGEROUS_PATTERNS:
            code = _re.sub(pattern, replacement, code, flags=_re.MULTILINE)
        last_code = code  # 同步清理后的代码

        # 校验
        valid, msg = validate_code(code)
        if not valid:
            history.append({"attempt": attempt, "code": code, "error": f"校验失败: {msg}"})
            if attempt <= MAX_RETRIES:
                messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
                messages.append({"role": "user", "content": f"代码校验未通过: {msg}\n请修正后重新生成。"})
            continue

        # 执行
        file_paths = {"file_path": file_path} if file_path else {}
        exec_result = execute_code(code, file_paths)

        if exec_result.get("ok") and exec_result.get("figure_json"):
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
                "history": history,
            }

        # 执行失败，构建纠错提示
        error_msg = exec_result.get("error", "未知错误")
        history.append({"attempt": attempt, "code": code, "error": error_msg})

        if attempt <= MAX_RETRIES:
            fix_hint = api_refs.build_fix_prompt(error_msg)
            messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            messages.append({"role": "user", "content": f"代码执行出错:\n{error_msg}\n\n{fix_hint}\n\n请修正代码后重新生成。只输出 ```python 代码块。"})

    # 所有尝试都失败
    return {
        "code": last_code,
        "validated": True,
        "validation_msg": "多次尝试后仍失败",
        "exec_result": {"ok": False, "error": f"经过 {MAX_RETRIES} 次纠错后仍执行失败", "figure_json": None, "stdout": "", "stderr": ""},
        "retries": MAX_RETRIES,
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
