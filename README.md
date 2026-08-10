# Learn Agent — AI Agent 开发学习与实践

Datawhale《Hello-Agents》教程的系统学习与动手实践。从手写三大 Agent 范式开始，到自研 Agent 框架，再到记忆/RAG、上下文工程与智能体通信协议（MCP / A2A / ANP），每一章的代码都亲手编写并实际运行验证。

> English version: [README_EN.md](README_EN.md)

## 📚 章节内容

| 章节 | 主题 | 核心内容 |
|------|------|---------|
| chapter4 | 智能体经典范式 | 手写 ReAct / Plan-and-Solve / Reflection 三大范式（~200 行脚本） |
| chapter6 | LangGraph | State / Node / Edge 图编排，用图拓扑统一三种范式 |
| chapter7 | 自研 Agent 框架 | 从 0 实现分层框架 HelloAgents：Message → LLM 封装 → Agent 抽象基类 → SimpleAgent / ReActAgent → ToolRegistry，使用方代码只需 7 行 |
| chapter8 | 记忆与检索 | MemoryTool（JSON 持久化跨会话记忆）+ RAGTool（TF-IDF 检索），含 A/B 对比实验 |
| chapter9 | 上下文工程 | GSSC 流水线（Gather / Select / Structure / Compress）+ NoteTool 结构化笔记 |
| chapter10 | 智能体通信协议 | MCP（智能体-工具）/ A2A（智能体-智能体协作）/ ANP（智能体网络），含 7 个实测 demo |

每章都有 `学习笔记.md`，记录核心概念、架构图解、代码走查与踩坑记录。

## 🛠️ 技术栈

- Python 3.12
- DeepSeek API（deepseek-v4-flash）
- Tavily 搜索
- LangGraph
- hello-agents 0.2.2 + FastMCP 2.14.7 + a2a-sdk 0.3.26

## ▶️ 运行

各章代码在独立 venv 中运行（venv 不入库），需要先在对应章节目录创建虚拟环境并安装依赖：

```bash
cd chapter10
python3 -m venv venv
venv/bin/pip install "hello-agents==0.2.2" fastmcp a2a-sdk==0.3.26 flask
venv/bin/python 01_quick_demo.py
```

LLM API Key 放在各章 `.env`（不入库）：

```
LLM_API_KEY=sk-xxx
LLM_MODEL_ID=deepseek-v4-flash
LLM_BASE_URL=https://api.deepseek.com
TAVILY_API_KEY=tvly-xxx
```

## ⚠️ 安全说明

`.env`、虚拟环境（venv/）、缓存文件均已通过 `.gitignore` 排除，不会提交到仓库。

## 📎 参考

- 教程：https://datawhalechina.github.io/hello-agents/
