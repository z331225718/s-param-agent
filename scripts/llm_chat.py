#!/usr/bin/env python3
"""
LLM 驱动的自然语言 → S参数操作 映射。
用 OpenAI 兼容 API（deepseek / gpt / 任何兼容服务），
让用户用任意自然语言表达，不再依赖关键词词典。

API Key 检测顺序:
  1. DEEPSEEK_API_KEY  环境变量
  2. OPENAI_API_KEY    环境变量
  3. 未设置 → 不可用，fallback 到规则解析

用法:
  from llm_chat import parse_with_llm, is_available
  if is_available():
      ops = parse_with_llm("帮我把这俩串起来看看传输", available_files)
"""

import os
import json
import re
from typing import List, Optional

# ── 检测 LLM 是否可用 ──────────────────────────────────────────

def is_available() -> bool:
    return bool(os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def get_llm_config() -> dict:
    """返回 LLM 配置：config.json 优先，其次环境变量。"""
    # 1. config.json
    for config_path in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.json"),
    ]:
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
                    "timeout_sec": cfg.get("timeout_sec", 60),
                }
        except Exception:
            pass

    # 2. 环境变量 fallback
    if os.environ.get("DEEPSEEK_API_KEY"):
        return {
            "api_key": os.environ["DEEPSEEK_API_KEY"],
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            "timeout_sec": 60,
        }
    elif os.environ.get("OPENAI_API_KEY"):
        return {
            "api_key": os.environ["OPENAI_API_KEY"],
            "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "timeout_sec": 60,
        }
    return {}


# ── 系统提示词 ──────────────────────────────────────────────────

SYSTEM_PROMPT = """你是 RF/微波工程的 S 参数助手。用户用自然语言（中文或英文）描述想要的操作，你把它翻译成结构化的 JSON 操作列表。

## 可用操作

### load — 加载 Touchstone 文件
{"action":"load","target":"文件名.s2p"}

### info — 查看网络信息
{"action":"info","target":"网络名"}

### plot — 画图
{"action":"plot","target":"网络名","params":["S11","S21"],"chart_type":"db|deg|smith|vswr|groupdelay|mag","freq_range":[start_Hz,stop_Hz],"dual_axis":true|false,"title":"可选标题"}

### cascade — 级联两个网络
{"action":"cascade","target":"网络A","cascade_with":"网络B","result_name":"结果名"}

### slice — 截取频率
{"action":"slice","target":"网络名","freq_range":[start_Hz,stop_Hz]}

### export — 导出
{"action":"export","target":"网络名","params":["S21"],"export_format":"csv|touchstone|html"}

### list — 列出已加载文件
{"action":"list"}

## 参数说明
- **target**: 文件名（.sNp）或已加载的网络名。如果用户没指定但之前加载过，可以省略
- **params**: S参数列表，如 ["S11","S21"]。S11=端口1反射，S21=端口2到1传输，S12=反向传输，S22=端口2反射
- **chart_type**: db=分贝幅度, deg=相位角度, smith=史密斯圆图, vswr=驻波比, groupdelay=群时延
- **freq_range**: [起始Hz, 终止Hz]。用户说"2到4G"→[2000000000,4000000000]，"1-3GHz"同理。省略=全频段
- **dual_axis**: 用户说"双Y轴"、"两个坐标系"、"左右轴"时为 true
- **export_format**: csv, touchstone, html

## 规则
1. 一句话可能包含多个操作（用逗号/然后分开），返回数组
2. 如果前面操作已加载文件，后面操作可以省略 target
3. 用户说"反射"/"回波损耗"/"return loss"→S11；"传输"/"插入损耗"/"增益"→S21
4. 只输出 JSON 数组，不要解释
5. 文件名保留用户原文"""

# ── LLM 调用 ────────────────────────────────────────────────────

def parse_with_llm(text: str, available_files: List[str] = None) -> Optional[List[dict]]:
    """
    用 LLM 解析自然语言，返回操作列表。

    Args:
        text: 用户输入的自然语言
        available_files: 当前已加载的网络名列表

    Returns:
        [{"action":"load","target":"a.s2p"}, {"action":"plot","params":["S21"],"chart_type":"db"}]
        失败返回 None
    """
    if not is_available():
        return None

    config = get_llm_config()

    # 构建上下文
    context = ""
    if available_files:
        context = f"\n当前已加载的网络: {', '.join(available_files)}"

    user_msg = f"用户输入: {text}{context}\n\n返回 JSON 操作数组:"

    try:
        import urllib.request

        payload = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,
            "max_tokens": 500,
        }

        req = urllib.request.Request(
            f"{config['base_url'].rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json",
            },
        )

        with urllib.request.urlopen(req, timeout=config.get("timeout_sec", 60)) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        content = result["choices"][0]["message"]["content"].strip()

        # 提取 JSON 数组
        json_match = re.search(r"\[.*\]", content, re.DOTALL)
        if json_match:
            ops = json.loads(json_match.group(0))
            # 补全默认值
            for op in ops:
                op.setdefault("params", [])
                op.setdefault("chart_type", "db")
                op.setdefault("dual_axis", False)
            return ops

        return None

    except Exception as e:
        # 静默失败，让调用方 fallback 到规则解析
        print(f"[llm_chat] LLM 调用失败: {e}")
        return None


def ops_to_nl_parser_format(llm_ops: List[dict]) -> List:
    """将 LLM 输出的 dict 列表转为 nl_parser 的 SParamOp 列表。"""
    from nl_parser import SParamOp
    result = []
    for op in llm_ops:
        result.append(SParamOp(
            action=op.get("action", ""),
            target=op.get("target", ""),
            params=op.get("params", []),
            chart_type=op.get("chart_type", "db"),
            freq_range=tuple(op["freq_range"]) if op.get("freq_range") else None,
            cascade_with=op.get("cascade_with", ""),
            export_format=op.get("export_format", ""),
            title=op.get("title", ""),
            dual_axis=op.get("dual_axis", False),
        ))
    return result


# ── 测试 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"LLM 可用: {is_available()}")
    if is_available():
        config = get_llm_config()
        print(f"Model: {config['model']} @ {config['base_url']}")

        tests = [
            "帮我把 LNA 和 BPF 串起来，看一下传输特性的 dB 曲线",
            "读 amp.s2p，把 S11 和 S21 画在一张图上，左右各一个纵轴",
            "只看 2.4 到 2.5G 这段，看驻波",
            "把当前的 S21 数据导成 CSV",
        ]
        for t in tests:
            print(f"\n输入: {t}")
            ops = parse_with_llm(t, available_files=["amp", "filter"])
            print(f"结果: {json.dumps(ops, ensure_ascii=False, indent=2)}")
    else:
        print("设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 环境变量后可用")
