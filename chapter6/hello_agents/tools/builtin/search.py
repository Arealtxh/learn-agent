"""
hello_agents/tools/builtin/search.py

搜索工具 — 封装 Tavily 搜索
和 ReActAgent 配合使用，让 Agent 具备实时搜索能力
"""

import os
from tavily import TavilyClient


def search_tool(query: str) -> str:
    """搜索互联网并返回摘要文本"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "错误: 未设置 TAVILY_API_KEY"

    client = TavilyClient(api_key=api_key)
    try:
        result = client.search(
            query=query,
            search_depth="basic",
            max_results=3,
            include_answer=True,
        )
        parts = []
        if result.get("answer"):
            parts.append(f"摘要: {result['answer']}")
        for i, r in enumerate(result.get("results", []), 1):
            parts.append(f"{i}. {r['title']}: {r['content'][:200]}")
        return "\n\n".join(parts) if parts else "未找到结果"
    except Exception as e:
        return f"搜索出错: {e}"
