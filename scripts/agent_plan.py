#!/usr/bin/env python3
"""Structured RF plan validation for the code-generation Agent."""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


ALLOWED_TOP_LEVEL = {"schema_version", "steps", "output", "assumptions"}
ALLOWED_STEP_KEYS = {"id", "op", "inputs", "args"}
ALLOWED_OUTPUT_KEYS = {"kind", "inputs", "chart_type", "traces"}
ALLOWED_TRACE_KEYS = {"source", "param", "label"}
ALLOWED_OPS = {"select", "slice_freq", "renormalize", "mixed_mode", "cascade", "compare", "derive"}
ALLOWED_OUTPUT_KINDS = {"plot", "text", "export"}
ALLOWED_CHART_TYPES = {"db", "mag", "deg", "smith", "vswr", "zmag", "zreal", "zimag", "groupdelay"}

_TOKEN_LEFT = r"(?<![A-Z0-9_])"
_TOKEN_RIGHT = r"(?![A-Z0-9_])"
_PARAM_RE = re.compile(
    _TOKEN_LEFT + r"(?:S(?:DD|DC|CD|CC)(?:\d+_\d+|[1-9][1-9])|[SZY](?:\d+_\d+|[1-9][1-9])|VSWR\d+)" + _TOKEN_RIGHT,
    re.IGNORECASE,
)
_ALL_WORDS = ("全部", "所有", "全参数", "all", "every", "each")


@dataclass
class PlanValidation:
    ok: bool
    plan: Optional[Dict[str, Any]] = None
    error: str = ""
    needs_confirmation: bool = False
    confirmation: str = ""


def build_plan_prompt(user_text: str, network_context: str) -> List[Dict[str, str]]:
    """Return messages for the planning LLM call."""
    system = """你是 RF/S 参数任务规划器。你只输出一个 JSON 对象，不输出 Markdown，不写 Python 代码。

目标：先理解用户需求，输出可校验的 RF Plan。Plan 只是短结构化计划，不要输出长推理。

JSON schema:
{
  "schema_version": 1,
  "steps": [
    {"id": "s1", "op": "select", "inputs": ["network:<已加载网络名>"], "args": {}}
  ],
  "output": {
    "kind": "plot",
    "inputs": ["s1"],
    "chart_type": "db",
    "traces": [{"source": "s1", "param": "S2_1"}]
  },
  "assumptions": ["短假设，最多3条"]
}

规则：
- 顶层只能包含 schema_version, steps, output, assumptions。
- step 只能包含 id, op, inputs, args。
- op 只能是 select, slice_freq, renormalize, mixed_mode, cascade, compare, derive。
- output 只能包含 kind, inputs, chart_type, traces。
- trace 只能包含 source, param, label。
- 网络必须用 inputs: ["network:<名称>"] 精确引用；如果用户没点名，用提示中的默认网络。
- 参数用规范格式：S2_1、S10_9、Z1_1、Y2_1、VSWR1、SDD1_1。用户说 S21 就写 S2_1；用户说 SDD11 就写 SDD1_1。
- 用户只指定某个参数时，traces 只能列这些参数；不要自动展开所有端口。
- 混模/差分/共模需求必须包含 mixed_mode step；如果 mixed_mode metadata 不是 ready，也仍然写出 mixed_mode step，后端会阻止执行并要求确认。
- 不要写 confidence、needs_confirmation、questions、required_metadata 等字段。
"""
    user = f"用户需求:\n{user_text}\n\n可用网络与元数据:\n{network_context}\n\n只输出 JSON。"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_plan_json(text: str) -> Dict[str, Any]:
    """Extract and parse one JSON object from an LLM response."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def validate_plan(plan: Dict[str, Any], networks: Dict[str, Dict[str, Any]], user_text: str = "") -> PlanValidation:
    """Validate and normalize an RF Plan. Never trusts model confidence."""
    if not isinstance(plan, dict):
        return PlanValidation(False, error="Plan 必须是 JSON 对象")
    unknown = set(plan) - ALLOWED_TOP_LEVEL
    if unknown:
        return PlanValidation(False, error=f"Plan 包含未知顶层字段: {sorted(unknown)}")
    if plan.get("schema_version") != 1:
        return PlanValidation(False, error="Plan schema_version 必须是 1")

    steps = plan.get("steps")
    output = plan.get("output")
    assumptions = plan.get("assumptions", [])
    if not isinstance(steps, list) or not steps:
        return PlanValidation(False, error="Plan.steps 必须是非空列表")
    if not isinstance(output, dict):
        return PlanValidation(False, error="Plan.output 必须是对象")
    if not isinstance(assumptions, list) or len(assumptions) > 3:
        return PlanValidation(False, error="Plan.assumptions 必须是最多 3 条的列表")

    normalized = {
        "schema_version": 1,
        "steps": [],
        "output": {},
        "assumptions": [str(a)[:160] for a in assumptions],
    }
    step_ids = set()
    step_networks: Dict[str, List[str]] = {}

    for step in steps:
        if not isinstance(step, dict):
            return PlanValidation(False, error="每个 step 必须是对象")
        unknown = set(step) - ALLOWED_STEP_KEYS
        if unknown:
            return PlanValidation(False, error=f"Step 包含未知字段: {sorted(unknown)}")
        sid = step.get("id")
        op = step.get("op")
        inputs = step.get("inputs", [])
        args = step.get("args", {})
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", sid):
            return PlanValidation(False, error=f"非法 step id: {sid}")
        if sid in step_ids:
            return PlanValidation(False, error=f"重复 step id: {sid}")
        if op not in ALLOWED_OPS:
            return PlanValidation(False, error=f"不支持的 step op: {op}")
        if not isinstance(inputs, list) or not all(isinstance(i, str) for i in inputs):
            return PlanValidation(False, error=f"Step {sid} inputs 必须是字符串列表")
        if not isinstance(args, dict):
            return PlanValidation(False, error=f"Step {sid} args 必须是对象")

        resolved_networks = _resolve_input_networks(inputs, step_networks, networks)
        if op == "select":
            if len(inputs) != 1 or not inputs[0].startswith("network:"):
                return PlanValidation(False, error=f"select step {sid} 必须精确引用一个 network:<name>")
            name = inputs[0].split(":", 1)[1]
            if name not in networks:
                return PlanValidation(False, error=f"未知网络: {name}")
            resolved_networks = [name]
        elif not resolved_networks:
            return PlanValidation(False, error=f"Step {sid} 没有可解析的输入网络")

        if op == "mixed_mode":
            gate = _mixed_mode_gate(resolved_networks, networks)
            if gate:
                return PlanValidation(True, plan=normalized, needs_confirmation=True, confirmation=gate)

        step_ids.add(sid)
        step_networks[sid] = resolved_networks
        normalized["steps"].append({"id": sid, "op": op, "inputs": inputs, "args": args})

    output_validation = _validate_output(output, step_ids, step_networks, networks, user_text)
    if not output_validation.ok:
        return output_validation
    normalized["output"] = output_validation.plan["output"]
    return PlanValidation(True, plan=normalized)


def expected_trace_count(plan: Optional[Dict[str, Any]]) -> Optional[int]:
    if not plan:
        return None
    traces = ((plan.get("output") or {}).get("traces") or [])
    if traces:
        return len(traces)
    return None


def trace_params(plan: Optional[Dict[str, Any]]) -> List[str]:
    if not plan:
        return []
    return [t.get("param", "") for t in ((plan.get("output") or {}).get("traces") or []) if t.get("param")]


def plan_has_op(plan: Optional[Dict[str, Any]], op: str) -> bool:
    return any(step.get("op") == op for step in ((plan or {}).get("steps") or []))


def selected_networks(plan: Optional[Dict[str, Any]]) -> List[str]:
    result = []
    for step in ((plan or {}).get("steps") or []):
        if step.get("op") != "select":
            continue
        for item in step.get("inputs") or []:
            if isinstance(item, str) and item.startswith("network:"):
                result.append(item.split(":", 1)[1])
    return result


def _validate_output(output, step_ids, step_networks, networks, user_text) -> PlanValidation:
    unknown = set(output) - ALLOWED_OUTPUT_KEYS
    if unknown:
        return PlanValidation(False, error=f"Output 包含未知字段: {sorted(unknown)}")
    kind = output.get("kind")
    inputs = output.get("inputs", [])
    chart_type = output.get("chart_type", "db")
    traces = output.get("traces", [])
    if kind not in ALLOWED_OUTPUT_KINDS:
        return PlanValidation(False, error=f"不支持的 output.kind: {kind}")
    if chart_type not in ALLOWED_CHART_TYPES:
        return PlanValidation(False, error=f"不支持的 chart_type: {chart_type}")
    if not isinstance(inputs, list) or not inputs:
        return PlanValidation(False, error="output.inputs 必须是非空列表")
    for ref in inputs:
        if ref not in step_ids:
            return PlanValidation(False, error=f"output 引用了未知 step: {ref}")
    if kind == "plot":
        if not isinstance(traces, list) or not traces:
            return PlanValidation(False, error="plot output 必须包含 traces")
        normalized_traces = []
        for trace in traces:
            if not isinstance(trace, dict):
                return PlanValidation(False, error="trace 必须是对象")
            unknown = set(trace) - ALLOWED_TRACE_KEYS
            if unknown:
                return PlanValidation(False, error=f"Trace 包含未知字段: {sorted(unknown)}")
            source = trace.get("source")
            if source not in step_ids:
                return PlanValidation(False, error=f"Trace 引用了未知 source: {source}")
            param = normalize_param(trace.get("param", ""))
            if not param:
                return PlanValidation(False, error="Trace 缺少合法 param")
            for net_name in step_networks.get(source, []):
                err = _validate_param_for_network(param, networks[net_name])
                if err:
                    return PlanValidation(False, error=f"{net_name}: {err}")
            item = {"source": source, "param": param}
            if trace.get("label"):
                item["label"] = str(trace["label"])[:80]
            normalized_traces.append(item)

        expected = _expected_params_from_text(user_text)
        if expected and not _wants_all(user_text):
            actual = [t["param"] for t in normalized_traces]
            if set(actual) != set(expected):
                return PlanValidation(False, error=f"Plan traces {actual} 与用户明确指定参数 {expected} 不一致")
        return PlanValidation(True, plan={"output": {
            "kind": kind,
            "inputs": inputs,
            "chart_type": chart_type,
            "traces": normalized_traces,
        }})

    return PlanValidation(True, plan={"output": {"kind": kind, "inputs": inputs, "chart_type": chart_type, "traces": []}})


def normalize_param(value: str) -> str:
    value = str(value or "").upper().strip()
    if not value:
        return ""
    if value.startswith("VSWR"):
        suffix = value[4:]
        return value if suffix.isdigit() else ""
    for prefix in ("SDD", "SDC", "SCD", "SCC"):
        if value.startswith(prefix):
            body = value[len(prefix):]
            parsed = _parse_index_body(body)
            return f"{prefix}{parsed[0]}_{parsed[1]}" if parsed else ""
    if value[0] in "SZY":
        parsed = _parse_index_body(value[1:])
        return f"{value[0]}{parsed[0]}_{parsed[1]}" if parsed else ""
    return ""


def _parse_index_body(body: str):
    if "_" in body:
        left, right = body.split("_", 1)
        if left.isdigit() and right.isdigit():
            return int(left), int(right)
        return None
    if len(body) == 2 and body.isdigit():
        return int(body[0]), int(body[1])
    return None


def _validate_param_for_network(param: str, info: Dict[str, Any]) -> str:
    nports = int(info.get("nports") or 0)
    if param.startswith("VSWR"):
        port = int(param[4:])
        return "" if 1 <= port <= nports else f"{param} 端口越界"
    if param.startswith(("SDD", "SDC", "SCD", "SCC")):
        mixed = info.get("mixed_mode") or {}
        if mixed.get("status") != "ready":
            return f"{param} 需要确认 mixed_mode P/N 映射"
        pair_count = int(mixed.get("pair_count") or 0)
        m, n = _parse_index_body(param[3:])
        return "" if 1 <= m <= pair_count and 1 <= n <= pair_count else f"{param} 差分端口越界"
    if param[0] in "SZY":
        m, n = _parse_index_body(param[1:])
        return "" if 1 <= m <= nports and 1 <= n <= nports else f"{param} 端口越界"
    return f"不支持的参数 {param}"


def _resolve_input_networks(inputs, step_networks, networks):
    resolved = []
    for item in inputs:
        if item.startswith("network:"):
            name = item.split(":", 1)[1]
            if name in networks:
                resolved.append(name)
        elif item in step_networks:
            resolved.extend(step_networks[item])
    return list(dict.fromkeys(resolved))


def _mixed_mode_gate(network_names, networks) -> str:
    blocked = []
    for name in network_names:
        mixed = networks[name].get("mixed_mode") or {}
        if mixed.get("status") != "ready":
            blocked.append(name)
    if not blocked:
        return ""
    return "这些网络的差分 P/N 映射未确认，不能生成权威混模结果: " + ", ".join(blocked)


def _expected_params_from_text(user_text: str) -> List[str]:
    params = []
    for raw in _PARAM_RE.findall(user_text or ""):
        param = normalize_param(raw)
        if param and param not in params:
            params.append(param)
    return params


def _wants_all(user_text: str) -> bool:
    lower = (user_text or "").lower()
    return any(word in lower for word in _ALL_WORDS)
