"""
第4章 — ReAct & Plan-and-Solve 智能体演示入口
直接运行: python main.py
"""
from dotenv import load_dotenv
load_dotenv()

from llm_client import HelloAgentsLLM
from tool_executor import ToolExecutor, search
from react_agent import ReActAgent
from plan_and_solve_agent import PlanAndSolveAgent
from reflection_agent import ReflectionAgent


def demo_react():
    """演示 ReAct 范式"""
    print("\n" + "="*60)
    print("🅰️  ReAct 范式演示（边想边做）")
    print("="*60)

    llm = HelloAgentsLLM()
    executor = ToolExecutor()
    executor.register_tool(
        "Search",
        "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。",
        search,
    )
    agent = ReActAgent(llm_client=llm, tool_executor=executor, max_steps=5)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    result = agent.run(question)
    if result:
        print(f"\n✅ ReAct 任务完成！答案: {result}")


def demo_plan_and_solve():
    """演示 Plan-and-Solve 范式"""
    print("\n" + "="*60)
    print("🅱️  Plan-and-Solve 范式演示（先规划后执行）")
    print("="*60)

    llm = HelloAgentsLLM()
    agent = PlanAndSolveAgent(llm_client=llm)
    question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    agent.run(question)


def demo_reflection():
    """演示 Reflection 范式"""
    print("\n" + "="*60)
    print("🅲  Reflection 范式演示（执行→反思→优化）")
    print("="*60)

    llm = HelloAgentsLLM()
    agent = ReflectionAgent(llm_client=llm, max_iterations=3)
    task = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"
    agent.run(task)


def main():
    print("🚀 Hello-Agents 第4章 — 三种经典范式对比\n")

    # 运行 ReAct
    demo_react()

    print("\n" + "-"*60)

    # 运行 Plan-and-Solve
    demo_plan_and_solve()

    print("\n" + "-"*60)

    # 运行 Reflection
    demo_reflection()


if __name__ == "__main__":
    main()
