#!/usr/bin/env python3
"""第10章 MCP 自动展开 demo：SimpleAgent + MCPTool（教程 10.2.4 方式1）"""
import os
from dotenv import load_dotenv

# 直接加载第4章的 .env（不复制文件）
load_dotenv("/home/txh/learn-agent/chapter4/.env")
print(f"LLM: {os.getenv('LLM_MODEL_ID')} | base: {os.getenv('LLM_BASE_URL')}")

from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import MCPTool

agent = SimpleAgent(name="助手", llm=HelloAgentsLLM())

# 无需任何配置，自动使用内置演示服务器
mcp_tool = MCPTool(name="calculator")
agent.add_tool(mcp_tool)
# ✅ MCP工具 'calculator' 已展开为 6 个独立工具
#    calculator_add / subtract / multiply / divide / greet / get_system_info

print("\n=== 让 Agent 计算 25 乘以 16 ===")
response = agent.run("计算 25 乘以 16")
print(f"\n回答: {response}")
