"""
PlanAndSolveAgent — 先规划，后执行 智能体
将 Planner + Executor 组合成完整的智能体
"""
from planner import Planner
from executor import Executor


class PlanAndSolveAgent:
    def __init__(self, llm_client):
        """初始化智能体，同时创建规划器和执行器实例"""
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self, question: str):
        """运行智能体的完整流程：先规划，后执行"""
        print(f"\n--- 开始处理问题 ---\n问题: {question}")

        # 1. 调用规划器生成计划
        plan = self.planner.plan(question)

        if not plan:
            print("\n--- 任务终止 ---\n无法生成有效的行动计划。")
            return

        print(f"\n📋 计划共 {len(plan)} 步:")
        for i, step in enumerate(plan, 1):
            print(f"  第{i}步: {step}")

        # 2. 调用执行器执行计划
        final_answer = self.executor.execute(question, plan)

        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")
