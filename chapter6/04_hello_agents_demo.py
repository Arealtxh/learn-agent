"""
第7章演示 — 用自己写的 HelloAgents 框架构建 Agent

和之前对比：
  第4章：写 200 行胶水代码，自己管 LLM 调用、prompt 拼接、历史管理
  第7章：import 3 个类，3 行代码搞定，底层实现完全透明
"""

import sys
sys.path.insert(0, "/home/txh/learn-agent/chapter6")

from dotenv import load_dotenv
load_dotenv("/home/txh/learn-agent/chapter4/.env")

from hello_agents import HelloAgentsLLM, SimpleAgent, ReActAgent, ToolRegistry
from hello_agents.tools.builtin.search import search_tool
from hello_agents.tools.builtin.calculator import calculator_tool


def demo_simple_agent():
    """SimpleAgent: 最简单的对话 Agent"""
    print("\n" + "=" * 55)
    print("  Demo 1: SimpleAgent — 基础对话")
    print("=" * 55)

    llm = HelloAgentsLLM()
    agent = SimpleAgent(
        name="聊天助手",
        llm=llm,
        system_prompt="你是一个友好的助手，请用中文简短回答。",
    )

    for question in [
        "中国最长的河流是什么？",
        "它的长度是多少公里？",
    ]:
        print(f"\n📝 问: {question}")
        answer = agent.run(question)
        print(f"💬 答: {answer[:200]}...")

    print(f"\n📊 历史消息数: {len(agent.get_history())}")


def demo_react_agent():
    """ReActAgent: 带搜索和计算的 Agent"""
    print("\n" + "=" * 55)
    print("  Demo 2: ReActAgent — 边想边做")
    print("=" * 55)

    llm = HelloAgentsLLM()

    # 注册工具（就像第4章 tool_executor.py 一样）
    tools = ToolRegistry()
    tools.register("Search", "搜索互联网获取最新信息", search_tool)
    tools.register("Calc", "计算数学表达式，例如 Calc[2 + 3 * 4]", calculator_tool)

    agent = ReActAgent(
        name="搜索助手",
        llm=llm,
        tool_registry=tools,
        max_steps=5,
    )

    # 选一个问题来测试
    question = "2025年诺贝尔物理学奖得主是谁？"
    print(f"\n📝 问: {question}")
    answer = agent.run(question)
    print(f"\n✅ 最终答案: {answer}")


def demo_compare_with_chapter4():
    """和第4章对比：同样功能，代码量对比"""
    print("\n" + "=" * 55)
    print("  Demo 3: 第4章 vs 第7章 代码量对比")
    print("=" * 55)

    chapter4_files = {
        "llm_client.py": "~30行",
        "tool_executor.py": "~30行",
        "react_agent.py": "~100行",
        "main.py": "~40行",
        "合计": "~200行",
    }

    chapter7_usage = {
        "import": "3行",
        "初始化": "3行",
        "调用": "1行",
        "合计": "7行",
    }

    print("\n第4章（手写全部）:")
    for f, lines in chapter4_files.items():
        print(f"  {f}: {lines}")

    print("\n第7章（用框架）:")
    for f, lines in chapter7_usage.items():
        print(f"  {f}: {lines}")

    print("\n这就是框架的价值 —— 把通用逻辑封装好，聚焦业务逻辑。")
    print("而且每一行框架代码都是你自己写的，没有黑盒。")


if __name__ == "__main__":
    demo_simple_agent()
    demo_react_agent()
    demo_compare_with_chapter4()
