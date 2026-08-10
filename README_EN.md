# Learn Agent — AI Agent Development: Learning & Practice

Hands-on learning of AI Agent development following the Datawhale 《Hello-Agents》 tutorial. The journey starts from implementing the three classic agent paradigms by hand, moves on to building my own agent framework, and extends to memory/RAG, context engineering, and agent communication protocols (MCP / A2A / ANP). Every piece of code was written and actually run/verified, not just read.

## 📚 Chapters

| Chapter | Topic | Highlights |
|---------|-------|------------|
| chapter4 | Classic Agent Paradigms | Hand-written ReAct / Plan-and-Solve / Reflection implementations (~200 lines) |
| chapter6 | LangGraph | State / Node / Edge graph orchestration; unified the three paradigms as graph topologies |
| chapter7 | Own Agent Framework | Built a layered framework from scratch: Message → LLM wrapper → Agent ABC → SimpleAgent / ReActAgent → ToolRegistry; user code shrinks from ~200 lines to 7 lines |
| chapter8 | Memory & Retrieval | MemoryTool (JSON-persisted cross-session memory) + RAGTool (TF-IDF retrieval), with A/B comparison experiments |
| chapter9 | Context Engineering | GSSC pipeline (Gather / Select / Structure / Compress) + NoteTool structured notes |
| chapter10 | Agent Communication Protocols | MCP (agent↔tool) / A2A (agent↔agent collaboration) / ANP (agent network), including 7 runnable demos |

Each chapter includes a `学习笔记.md` (Chinese study notes) covering core concepts, architecture diagrams, code walkthroughs, and pitfalls discovered during hands-on practice.

## 🛠️ Tech Stack

- Python 3.12
- DeepSeek API (deepseek-v4-flash)
- Tavily search
- LangGraph
- hello-agents 0.2.2 + FastMCP 2.14.7 + a2a-sdk 0.3.26

## ▶️ Run

Each chapter uses its own venv (not committed). Create a virtual environment and install dependencies first:

```bash
cd chapter10
python3 -m venv venv
venv/bin/pip install "hello-agents==0.2.2" fastmcp a2a-sdk==0.3.26 flask
venv/bin/python 01_quick_demo.py
```

LLM API keys live in a per-chapter `.env` (not committed):

```
LLM_API_KEY=sk-xxx
LLM_MODEL_ID=deepseek-v4-flash
LLM_BASE_URL=https://api.deepseek.com
TAVILY_API_KEY=tvly-xxx
```

## ⚠️ Security

`.env`, virtual environments (`venv/`), and cache files are excluded via `.gitignore` and never committed.

## 🔗 Reference

- Tutorial: https://datawhalechina.github.io/hello-agents/
