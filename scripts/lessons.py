#!/usr/bin/env python3
"""
自进化教训库。
每次 LLM 纠错成功后，自动提取"错误模式 → 正确 API"规则，
存入 lessons.json。下次注入 system prompt，类似 Few-Shot 记忆。

教训不会无限膨胀——去重 + 过期 + 上限控制。
"""

import json
import os
import sys
import re
import time
from typing import List, Dict

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lessons.json")
if not os.path.exists(_PATH) and getattr(sys, 'frozen', False):
    _PATH = os.path.join(sys._MEIPASS, "lessons.json")

# ═══════════════════════════════════════════════════════════════
#  加载 / 保存
# ═══════════════════════════════════════════════════════════════

def _load() -> List[Dict]:
    if os.path.exists(_PATH):
        with open(_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save(lessons: List[Dict]):
    with open(_PATH, "w", encoding="utf-8") as f:
        json.dump(lessons, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
#  学习
# ═══════════════════════════════════════════════════════════════

def learn(error_msg: str, wrong_code: str = "", correct_code: str = "") -> str:
    """
    从一次成功的纠错中学习。

    自动提取：
      - 错误模式（从报错信息中提取关键 API 名）
      - 正确做法（从图谱搜索中获取）
      - 对比 wrong_code 和 correct_code 的差异

    返回: 生成的教训文本（供即时注入），同时持久化到 lessons.json
    """
    lessons = _load()

    # 1. 提取错误中的关键 API 名
    wrong_api = _extract_wrong_api(error_msg)

    # 过滤噪音：不是 API 错误的（文件路径、通用异常等）不学习
    if _is_noise(wrong_api, error_msg):
        return ""  # 不学习噪音

    # 2. 从图谱获取正确 API
    from api_graph import graph_search
    graph_results = graph_search(wrong_api, max_hops=1, top_k=2)

    correct_info = ""
    if graph_results:
        r = graph_results[0]
        correct_info = r.get("desc", "")
        # 添加纠错路径
        if r.get("paths_to_mistakes"):
            correct_info += " | 纠错路径: " + "; ".join(r["paths_to_mistakes"])

    # 3. 从代码 diff 中提取关键变化
    code_diff = _extract_diff(wrong_code, correct_code)

    # 4. 构建教训条目
    lesson_text = _format_lesson(wrong_api, correct_info, code_diff)

    # 5. 去重：检查是否已有类似教训
    existing = _find_similar(lessons, wrong_api)
    if existing:
        existing["count"] = existing.get("count", 1) + 1
        existing["last_seen"] = time.time()
        existing["correct_info"] = correct_info or existing.get("correct_info", "")
        _save(lessons)
        return lesson_text

    # 6. 新增
    lessons.append({
        "wrong_pattern": wrong_api,
        "correct_info": correct_info,
        "code_diff": code_diff,
        "count": 1,
        "first_seen": time.time(),
        "last_seen": time.time(),
    })

    # 7. 限制数量（保留最近 50 条，高频的不删）
    if len(lessons) > 100:
        lessons.sort(key=lambda l: -(l.get("count", 1) * 10 + l.get("last_seen", 0) / 86400))
        lessons = lessons[:50]
        lessons.sort(key=lambda l: l.get("last_seen", 0))

    _save(lessons)
    return lesson_text


def _is_noise(wrong_api: str, error_msg: str) -> bool:
    """判断是否是不值得学习的噪音错误。"""
    # 文件路径
    if ".s2p" in wrong_api or ".s1p" in wrong_api or ".csv" in wrong_api:
        return True
    if "\\\\" in wrong_api and len(wrong_api) > 50:
        return True
    # FileNotFoundError 本身
    if "FileNotFoundError" in error_msg and "No such file" in error_msg:
        return True
    # 太短或太长
    if len(wrong_api) < 3 or len(wrong_api) > 100:
        return True
    return False


def _extract_wrong_api(error_msg: str) -> str:
    """从报错信息中提取错误的 API 名。"""
    # AttributeError: 'Figure' object has no attribute 'update_xaxis'
    m = re.search(r"has no attribute '(\w+)'", error_msg)
    if m:
        return m.group(1)

    # NameError: name 'xxx' is not defined
    m = re.search(r"name '(\w+)' is not defined", error_msg)
    if m:
        return m.group(1)

    # TypeError: ... 'xxx'
    m = re.search(r"TypeError.*?'(\w+)'", error_msg)
    if m:
        return m.group(1)

    # IndexError: too many indices / 3-dimensional
    m = re.search(r"(\w+)\[\d+,\d+\]", error_msg)
    if m:
        return f"{m.group(1)}[m,n]"
    # IndexError: array is N-dimensional — 提取可能的变量名
    m = re.search(r"too many indices.*(?:array|for array)", error_msg)
    if m:
        return "3D_array_index[missing_colon]"

    # 兜底：取引号中的第一个长词（跳过文件路径）
    quoted = re.findall(r"'([^']+)'", error_msg)
    for q in sorted(quoted, key=len, reverse=True):
        if len(q) >= 5 and q[0].isalpha():
            # 跳过文件路径（含 \\ 或 / 或 .s2p .csv 等）
            if "\\\\" in q or ".s" in q.lower() or ".csv" in q.lower() or ".html" in q.lower():
                continue
            if len(q) > 100:  # 太长，大概率是路径
                continue
            return q

    return error_msg[:80]


def _extract_diff(wrong_code: str, correct_code: str) -> str:
    """提取两段代码的关键差异（简化版）。"""
    if not wrong_code or not correct_code:
        return ""
    # 只提取不同的行
    wrong_lines = [l.strip() for l in wrong_code.split("\n") if l.strip() and not l.strip().startswith("#")]
    correct_lines = [l.strip() for l in correct_code.split("\n") if l.strip() and not l.strip().startswith("#")]

    # 找差异行
    diffs = []
    for wl in wrong_lines:
        if wl not in correct_lines:
            # 在正确代码中找最相似的行
            for cl in correct_lines:
                if _similarity(wl, cl) > 0.5:
                    diffs.append("❌ " + wl[:80] + "\n✅ " + cl[:80])
                    break

    return "\n".join(diffs[:3])


def _similarity(a: str, b: str) -> float:
    """简单的行相似度（共享词比例）。"""
    wa = set(a.split())
    wb = set(b.split())
    if not wa or not wb:
        return 0
    return len(wa & wb) / max(len(wa), len(wb))


def _format_lesson(wrong_api: str, correct_info: str, code_diff: str) -> str:
    """格式化一条教训。"""
    parts = ["❌ `" + wrong_api + "`"]
    if correct_info:
        parts.append("→ " + correct_info[:150])
    if code_diff:
        parts.append("\n" + code_diff[:200])
    return " ".join(parts)


def _find_similar(lessons: List[Dict], pattern: str) -> Dict | None:
    """查找已有的相似教训。"""
    pattern_lower = pattern.lower()
    for l in lessons:
        if l.get("wrong_pattern", "").lower() == pattern_lower:
            return l
    return None


# ═══════════════════════════════════════════════════════════════
#  注入 prompt
# ═══════════════════════════════════════════════════════════════

def build_lessons_prompt(max_items: int = 15) -> str:
    """
    生成教训摘要，注入 system prompt。
    按「频次 × 最近性」排序，取 top-N。
    """
    lessons = _load()
    if not lessons:
        return ""

    # 排序：高频 + 最近
    now = time.time()
    scored = []
    for l in lessons:
        age_days = (now - l.get("last_seen", now)) / 86400
        score = l.get("count", 1) * 5 - age_days
        scored.append((score, l))
    scored.sort(key=lambda x: -x[0])

    lines = ["## 🧠 已学教训（避免再犯）"]
    for _, l in scored[:max_items]:
        wp = l.get("wrong_pattern", "")
        ci = l.get("correct_info", "")
        cnt = l.get("count", 1)
        tag = " (\u00d7" + str(cnt) + ")" if cnt > 1 else ""
        if ci:
            lines.append("- ❌ `" + wp + "`" + tag + " → " + ci[:120])
        else:
            lines.append("- ❌ `" + wp + "`" + tag)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  统计
# ═══════════════════════════════════════════════════════════════

def stats() -> dict:
    lessons = _load()
    return {
        "total": len(lessons),
        "top_3": [(l["wrong_pattern"], l.get("count", 1)) for l in
                   sorted(lessons, key=lambda l: -l.get("count", 1))[:3]],
    }


# ── 测试 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # 模拟学习
    print("=== 模拟学习 ===")
    r1 = learn(
        "AttributeError: 'Figure' object has no attribute 'update_xaxis'",
        wrong_code="fig.update_xaxis(type='log')",
        correct_code="fig.update_layout(xaxis_type='log')",
    )
    print(r1[:200])

    r2 = learn(
        "IndexError: too many indices for array: array is 3-dimensional",
        wrong_code="ntwk.s_db[1,0]",
        correct_code="ntwk.s_db[:,1,0]",
    )
    print(r2[:200])

    r3 = learn(
        "AttributeError: 'Figure' object has no attribute 'update_xaxis'",
        wrong_code="fig.update_xaxis(title='dB')",
        correct_code="fig.update_layout(yaxis_title='dB')",
    )
    print(f"\n重复教训（×{r3.count if hasattr(r3,'count') else '?'}）")

    print("\n=== 教训库 ===")
    print(f"统计: {stats()}")

    print("\n=== Prompt 注入 ===")
    print(build_lessons_prompt(max_items=5))
