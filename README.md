# 📡 S-Parameter Conversational Agent

**对话式 S 参数处理画图 Agent** — 用自然语言替代传统 RF 软件。

> "读取 filter.s2p，画 S21 和 S11 的 dB 图，双 Y 轴" → 出图

## 架构

```
用户自然语言
    │
    ▼
┌─────────────────────────────────┐
│  LLM (DeepSeek/GPT/Ollama/...)  │  生成受限 Python 代码
│  System Prompt + API 参考注入   │  只用 skrf + plotly
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  AST 校验器                      │  import 白名单
│  ❌ os/subprocess/eval 拦截     │  安全沙箱
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  subprocess 沙箱执行 (15s 超时)  │  捕获 fig → JSON
│  失败 → 知识图谱纠错 → 重试(×3) │  自动学习教训
└──────────┬──────────────────────┘
           │
           ▼
      Web UI 渲染交互式 Plotly 图表
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 LLM
cp config.example.json config.json
# 编辑 config.json，填入 api_key（或设环境变量 DEEPSEEK_API_KEY）

# 3. 启动
python app.py
# → 浏览器打开 http://localhost:5050
```

在右下角聊天框打字即可。

## 功能

| 类别 | 支持 |
|------|------|
| 文件 | 读取 .s1p/.s2p/.sNp Touchstone |
| 参数 | S/dB/相位/VSWR/Z/Y/群时延 |
| 图表 | dB/Smith/相位/VSWR/群时延/双Y轴/log轴 |
| 处理 | 截取频段/级联/去嵌/重归一化 |
| 导出 | Touchstone/CSV/HTML |

## 对话示例

```
"画 S21 的 Smith 圆图"
"Z11 幅度，X 和 Y 都用 log"
"把反射和传输画一张图，左右两个纵轴"
"只看 2-4GHz 的驻波"
"级联 LNA 和 BPF，看最终 S21"
```

## 自进化

Agent 纠错成功后自动学习，累积的教训注入后续请求的 system prompt：

```
lessons.json  ← 自动生成
  - ❌ update_xaxis (×2) → ✅ update_layout(xaxis_type=)
  - ❌ s_db[1,0]         → ✅ s_db[:, 1, 0]
```

## 配置

`config.json` — 支持任意 OpenAI 兼容 API：

```json
{
  "llm": {
    "api_key": "sk-xxx",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat"
  }
}
```

| 服务 | base_url |
|------|----------|
| DeepSeek | `https://api.deepseek.com` |
| OpenAI | `https://api.openai.com/v1` |
| Ollama | `http://localhost:11434` |
| vLLM | `http://localhost:8000` |

## 文件结构

```
scripts/
├── app.py              # Flask Web 仪表盘 + /api/agent 端点
├── code_agent.py       # 强约束代码生成 Agent（核心）
├── api_graph.py        # API 知识图谱 (876节点/494边)
├── api_refs.py         # 图谱搜索 + 高频速查
├── api_index.json      # 自动提取的 API 索引 (834条)
├── extract_apis.py     # API 自动提取脚本
├── lessons.py          # 自进化教训库
├── s_params.py         # skrf+plotly 便捷库（可选使用）
├── config.example.json # LLM 配置模板
├── requirements.txt    # scikit-rf, plotly, flask, numpy, networkx
├── templates/
│   └── dashboard.html  # 暗色主题 Web UI（聊天+手动面板）
└── demo_verify.py      # 自验证脚本
```

## License

MIT
