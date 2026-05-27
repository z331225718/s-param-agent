#!/usr/bin/env python3
"""Inspect Touchstone networks and suggest domain-specific quick actions."""

import re
from collections import defaultdict
from typing import Dict, List

import numpy as np


POWER_TOKENS = {"VDD", "VCC", "VSS", "GND", "PWR", "POWER", "PDN", "VRM"}
SIGNAL_TOKENS = {"TX", "RX", "DP", "DN", "CH", "LINE", "NET", "P", "N"}
ENDPOINT_TOKENS = {
    "J1", "J2", "J3", "J4", "NEAR", "FAR", "IN", "OUT", "INPUT", "OUTPUT",
    "TX", "RX", "SRC", "DST", "SOURCE", "LOAD", "A", "B",
}


def get_port_names(ntwk) -> List[str]:
    """Return port names or stable fallback names."""
    names = getattr(ntwk, "port_names", None) or []
    result = []
    for i in range(getattr(ntwk, "nports", 0)):
        value = names[i] if i < len(names) and names[i] else ""
        result.append(str(value) if value else f"Port{i + 1}")
    return result


def classify_network(ntwk) -> str:
    """Classify the network as signal, power, mixed, or unknown."""
    names = get_port_names(ntwk)
    tokens = set()
    for name in names:
        tokens.update(_tokens(name))

    z0 = np.asarray(getattr(ntwk, "z0", []), dtype=complex)
    z0_abs = np.abs(z0[np.isfinite(z0)]) if z0.size else np.array([])

    power = bool(tokens & POWER_TOKENS)
    signal = bool(tokens & SIGNAL_TOKENS)
    if z0_abs.size:
        power = power or bool(np.nanmedian(z0_abs) <= 1.0)
        signal = signal or bool(np.nanmedian(z0_abs) >= 25.0)

    if power and signal:
        return "mixed"
    if power:
        return "power"
    if signal:
        return "signal"
    return "unknown"


def detect_port_pairs(ntwk) -> List[Dict]:
    """Detect likely two-ended nets using names first, then S-matrix strength."""
    nports = getattr(ntwk, "nports", 0)
    pairs = []
    used = set()

    groups = defaultdict(list)
    for idx, name in enumerate(get_port_names(ntwk)):
        key = _net_key(name)
        if key:
            groups[key].append(idx)

    for indices in groups.values():
        if len(indices) < 2:
            continue
        for i in range(0, len(indices) - 1, 2):
            a, b = indices[i], indices[i + 1]
            pairs.append(_pair(a, b, "name", "high"))
            used.update((a, b))

    for p in _matrix_pairs(ntwk):
        if p["a"] in used or p["b"] in used:
            continue
        pairs.append(p)
        if p["confidence"] == "high":
            used.update((p["a"], p["b"]))

    return pairs


def build_quick_actions(ntwk) -> List[Dict]:
    """Build concrete quick actions for the dashboard."""
    kind = classify_network(ntwk)
    nports = getattr(ntwk, "nports", 0)
    pairs = detect_port_pairs(ntwk)

    if kind == "power":
        return [{
            "id": "zmag",
            "label": "mag(Zxx)",
            "chart_type": "zmag",
            "params": [f"Z{i + 1}{i + 1}" for i in range(nports)],
        }]

    actions = []
    if kind in ("signal", "mixed"):
        actions.append({
            "id": "rl",
            "label": "RL",
            "chart_type": "db",
            "params": [f"S{i + 1}{i + 1}" for i in range(nports)],
        })

        il_params = []
        for pair in pairs:
            a, b = pair["a"], pair["b"]
            il_params.append(f"S{b + 1}{a + 1}")
        if il_params:
            actions.append({"id": "il", "label": "IL", "chart_type": "db", "params": _unique(il_params)})

        next_params, fext_params = _crosstalk_params(pairs)
        if next_params:
            actions.append({"id": "next", "label": "NEXT", "chart_type": "db", "params": next_params})
        if fext_params:
            actions.append({"id": "fext", "label": "FEXT", "chart_type": "db", "params": fext_params})

    return actions


def inspect_network(ntwk) -> Dict:
    """Return frontend-safe network metadata."""
    return {
        "network_kind": classify_network(ntwk),
        "port_names": get_port_names(ntwk),
        "port_pairs": detect_port_pairs(ntwk),
        "quick_actions": build_quick_actions(ntwk),
    }


def _tokens(name: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9]+", name.upper()) if t]


def _net_key(name: str) -> str:
    parts = []
    for token in _tokens(name):
        if token in ENDPOINT_TOKENS:
            continue
        if re.fullmatch(r"J\d+", token):
            continue
        parts.append(token)
    return "_".join(parts)


def _matrix_pairs(ntwk) -> List[Dict]:
    nports = getattr(ntwk, "nports", 0)
    if nports < 2 or not hasattr(ntwk, "s_db"):
        return []

    db = np.array(ntwk.s_db[0], dtype=float)
    best = {}
    for i in range(nports):
        row = db[i].copy()
        row[i] = -np.inf
        if np.all(~np.isfinite(row)):
            continue
        best[i] = int(np.nanargmax(row))

    pairs = []
    seen = set()
    for a, b in best.items():
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        confidence = "high" if best.get(b) == a else "low"
        pairs.append(_pair(a, b, "matrix", confidence))
    return pairs


def _pair(a: int, b: int, source: str, confidence: str) -> Dict:
    a, b = int(a), int(b)
    if b < a:
        a, b = b, a
    return {"a": a, "b": b, "source": source, "confidence": confidence}


def _crosstalk_params(pairs: List[Dict]):
    near = [p["a"] for p in pairs]
    far = [p["b"] for p in pairs]
    next_params = []
    fext_params = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            next_params.append(f"S{near[j] + 1}{near[i] + 1}")
            fext_params.append(f"S{far[j] + 1}{near[i] + 1}")
    return _unique(next_params), _unique(fext_params)


def _unique(values: List[str]) -> List[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
