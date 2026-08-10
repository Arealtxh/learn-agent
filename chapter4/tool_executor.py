"""
ToolExecutor — 管理 Agent 的"工具箱"
使用 Tavily 搜索（复用你在第1章的 API key）
"""
import os
from typing import Dict, Any
from tavily import TavilyClient


def search(query: str) -> str:
    """
    网页搜索工具 — 使用 Tavily API
    当 Agent 需要查询实时信息时使用
    """
    print(f"🔍 正在搜索: {query}")
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "错误: 未配置 TAVILY_API_KEY"

    try:
        tavily = TavilyClient(api_key=api_key)
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        if response.get("answer"):
            return response["answer"]

        results = response.get("results", [])
        if not results:
            return f"没有找到关于 '{query}' 的信息。"

        lines = []
        for i, r in enumerate(results[:3]):
            lines.append(f"[{i+1}] {r.get('title', '')}\n{r.get('content', '')}")
        return "\n\n".join(lines)

    except Exception as e:
        return f"搜索出错: {e}"


class ToolExecutor:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, name: str, description: str, func: callable):
        self.tools[name] = {"description": description, "func": func}
        print(f"🔧 工具 '{name}' 已注册")

    def get_tool(self, name: str) -> callable:
        info = self.tools.get(name)
        return info["func"] if info else None

    def get_available_tools(self) -> str:
        return "\n".join(
            [f"- {name}: {info['description']}" for name, info in self.tools.items()]
        )
