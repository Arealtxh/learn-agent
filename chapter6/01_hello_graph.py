"""
══════════════════════════════════════════════════
  01 ─ 极简入门：LangGraph 的三大核心概念
══════════════════════════════════════════════════

  LangGraph = 状态机 + 有向图

  三个概念你就能理解全部：
   ① State（状态） → 全局共享的数据
   ② Node（节点）  → 每个节点是一个函数，读写 State
   ③ Edge（边）    → 决定节点之间的流转路径

  我们用一个"计数器"来演示：
     start → add_one → add_one → add_one → ... → 到达 3 次就结束
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


# ══════════════════════════════════════════════
#  ① State（状态）— 整个图共享的数据结构
# ══════════════════════════════════════════════
#   就像第4章你手写的 self.history，但这里是强类型的
#   所有节点都能读和写这个 State

class CounterState(TypedDict):
    count: int           # 当前计数
    history: list[str]   # 记录每一步做了什么


# ══════════════════════════════════════════════
#  ② Node（节点）— 每个节点就是一个函数
# ══════════════════════════════════════════════
#   输入：当前 State
#   输出：要更新的字段（返回 dict，自动合并到 State）

def add_one_node(state: CounterState) -> dict:
    """节点1：给计数器加1"""
    new_count = state["count"] + 1
    print(f"  [节点 add_one] {state['count']} → {new_count}")

    return {
        "count": new_count,
        "history": [*state["history"], f"第{new_count}次计数"],
    }


# ══════════════════════════════════════════════
#  ③ Edge（边）— 决定下一步去哪
# ══════════════════════════════════════════════
#   普通边：固定流向（A → B）
#   条件边：根据 State 动态决定流向（最强大）

def route_after_add(state: CounterState) -> Literal["add_one", "__end__"]:
    """条件边：计数达到3次就结束，否则继续加"""
    if state["count"] >= 3:
        print(f"  → 达到3次，结束！")
        return "__end__"  # 返回 "__end__" 代表 END
    else:
        return "add_one"  # 继续回到 add_one 节点


# ══════════════════════════════════════════════
#  组装：构建图
# ══════════════════════════════════════════════

def build_counter_graph():
    # 1️⃣ 创建图，绑定 State 类型
    workflow = StateGraph(CounterState)

    # 2️⃣ 添加节点（就像在流程图里画框）
    workflow.add_node("add_one", add_one_node)

    # 3️⃣ 设置入口（START → add_one）
    workflow.add_edge(START, "add_one")

    # 4️⃣ 添加条件边（决定下一步去哪）
    workflow.add_conditional_edges(
        "add_one",        # 从哪个节点出发
        route_after_add,  # 判断函数
    )
    # route_after_add 返回 "add_one" → 继续加
    # route_after_add 返回 "__end__"  → 停止

    # 5️⃣ 编译 → 生成可执行的 App
    app = workflow.compile()
    return app


# ══════════════════════════════════════════════
#  运行
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 LangGraph 极简示例：计数器")
    print("=" * 50)

    app = build_counter_graph()

    # 初始状态
    initial_state = {"count": 0, "history": []}

    print(f"\n初始 State: {initial_state}")
    print("-" * 40)

    # stream() 会逐步输出每个节点执行后的 State 快照
    # 每执行一个节点，你就看到 State 被更新一次
    for step_output in app.stream(initial_state):
        # step_output 是 {"节点名": {更新后的字段}}
        node_name = list(step_output.keys())[0]
        node_result = step_output[node_name]
        print(f"  State 更新: {node_result}")

    print("-" * 40)
    print(f"✅ 最终结果!")
