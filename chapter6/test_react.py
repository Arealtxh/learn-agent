"""
测试框架：ReActAgent + 搜索工具
"""
import sys
sys.path.insert(0, "/home/txh/learn-agent/chapter6")

from dotenv import load_dotenv
load_dotenv("/home/txh/learn-agent/chapter4/.env")

from hello_agents import HelloAgentsLLM, ReActAgent, ToolRegistry
from hello_agents.tools.builtin.search import search_tool
from hello_agents.tools.builtin.calculator import calculator_tool

llm = HelloAgentsLLM()

tools = ToolRegistry()
tools.register("Search", "搜索互联网获取最新信息", search_tool)
tools.register("Calc", "计算数学表达式，例如 Calc[2 + 3 * 4]", calculator_tool)

agent = ReActAgent(
    name="搜索助手",
    llm=llm,
    tool_registry=tools,
    max_steps=5,
)

question = "2025年诺贝尔物理学奖得主是谁？"
print(f"问: {question}")
answer = agent.run(question)
print(f"\n最终答案: {answer}")
