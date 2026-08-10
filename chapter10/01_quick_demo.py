#!/usr/bin/env python3
"""第10章 快速体验：三种协议的核心功能（对应教程 10.1.4）"""
from hello_agents.tools import MCPTool, A2ATool, ANPTool

print("=" * 60)
print("1. MCP：访问工具（Memory 传输，内置演示服务器）")
print("=" * 60)
mcp_tool = MCPTool()
result = mcp_tool.run({
    "action": "call_tool",
    "tool_name": "add",
    "arguments": {"a": 10, "b": 20}
})
print(f"MCP计算结果: {result}")

print()
print("=" * 60)
print("2. ANP：服务发现")
print("=" * 60)
anp_tool = ANPTool()
anp_tool.run({
    "action": "register_service",
    "service_id": "calculator",
    "service_type": "math",
    "endpoint": "http://localhost:8080"
})
services = anp_tool.run({"action": "discover_services"})
print(f"发现的服务: {services}")

print()
print("=" * 60)
print("3. A2A：智能体通信")
print("=" * 60)
a2a_tool = A2ATool("http://localhost:5000")
print("A2A工具创建成功")
print()
print("✅ 三种协议快速体验完成")
