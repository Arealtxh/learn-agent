"""
══════════════════════════════════════════════════
  02 ─ 三步问答助手（LangGraph 实战）
══════════════════════════════════════════════════

  流程：用户提问 → ① 理解意图 → ② 搜索 → ③ 回答

  和第一章的 travel_agent.py 区别：
    第一章：手写 ReAct 循环（自己管 prompt 拼接、正则解析）
    这章：LangGraph 帮你管节点流转和状态传递

  你只需要定义：
    - State：要存什么数据
    - Node：每步做什么
    - Edge：做完这步下一步去哪
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient

# ── 加载配置 ──
# 复用第4章的 API 配置
load_dotenv("/home/txh/learn-agent/chapter4/.env")

# 用 DeepSeek（完美兼容 OpenAI SDK，只需改 base_url + model）
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "deepseek-v4-flash"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
    temperature=0.3,
)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ══════════════════════════════════════════════
#  第一步：定义 State（整个图的共享数据）
# ══════════════════════════════════════════════

class SearchState(TypedDict):
    """三步问答助手的全局状态

    每个字段都是节点之间传递信息的"通道"：
    """
    user_query: str       # ① 理解阶段：LLM 提炼后的用户需求
    search_query: str     # ② 搜索阶段：优化后的搜索关键词
    search_results: str   # ③ 搜索阶段：Tavily 返回的结果
    final_answer: str     # ④ 回答阶段：最终答案
    step: str             # 标记当前步骤（用于条件判断）


# ══════════════════════════════════════════════
#  第二步：定义节点（每个节点是一个函数）
# ══════════════════════════════════════════════

def understand_node(state: SearchState) -> dict:
    """节点1：理解用户意图，生成搜索关键词"""
    print("\n── 🧠 阶段1：理解意图 ──")

    # 注意：state 里没有 messages 字段
    # 我们需要从某个地方拿到用户原始问题
    # 实际上 state 在第一次调用时只有 user_query
    user_msg = state["user_query"]

    prompt = f"""分析用户的查询："{user_msg}"

请完成两个任务：
1. 用一句话总结用户想了解什么
2. 生成最适合搜索引擎的关键词（中英文均可，要精准）

格式：
理解：[总结]
搜索词：[关键词]"""

    response = llm.invoke([SystemMessage(content=prompt)])
    text = response.content
    print(f"  用户问题: {user_msg}")
    print(f"  LLM 响应: {text}")

    # 解析搜索词
    search_q = user_msg  # 默认用原问题
    if "搜索词：" in text:
        search_q = text.split("搜索词：")[1].strip()

    return {
        "user_query": text,
        "search_query": search_q,
        "step": "understood",
    }


def search_node(state: SearchState) -> dict:
    """节点2：执行搜索"""
    print("\n── 🔍 阶段2：搜索 ──")
    query = state["search_query"]
    print(f"  搜索关键词: {query}")

    try:
        result = tavily.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=True,
        )
        # 整理搜索结果成文本
        results_text = f"摘要: {result.get('answer', '无')}\n\n"
        for i, r in enumerate(result.get("results", []), 1):
            results_text += f"{i}. {r['title']}\n   {r['content'][:200]}...\n"

        print(f"  ✅ 搜索完成，找到 {len(result.get('results', []))} 条结果")
        return {
            "search_results": results_text,
            "step": "searched",
        }
    except Exception as e:
        print(f"  ❌ 搜索失败: {e}")
        return {
            "search_results": f"搜索出错: {e}",
            "step": "search_failed",
        }


def answer_node(state: SearchState) -> dict:
    """节点3：生成最终答案"""
    print("\n── 💡 阶段3：生成答案 ──")

    if state["step"] == "search_failed":
        # 搜索失败 → 回退到 LLM 自身知识
        prompt = f"搜索暂时不可用，请基于你的知识回答：{state['user_query']}"
    else:
        prompt = f"""基于以下搜索结果回答用户问题。

用户问题：{state['user_query']}

搜索结果：
{state['search_results']}

请给出完整、准确、有条理的回答："""

    response = llm.invoke([SystemMessage(content=prompt)])
    answer = response.content

    return {
        "final_answer": answer,
        "step": "completed",
    }


# ══════════════════════════════════════════════
#  第三步：构建图 —— 把节点连起来
# ══════════════════════════════════════════════

def build_search_agent():
    # 1️⃣ 创建图，绑定 State 类型
    workflow = StateGraph(SearchState)

    # 2️⃣ 添加三个节点（就像画三个框）
    workflow.add_node("understand", understand_node)
    workflow.add_node("search", search_node)
    workflow.add_node("answer", answer_node)

    # 3️⃣ 连边（就像画箭头）
    #    START → understand → search → answer → END
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)

    # 4️⃣ 编译 → 生成可执行的 App
    return workflow.compile()


# ══════════════════════════════════════════════
#  运行
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  🤖 LangGraph 三步问答助手")
    print("=" * 55)

    agent = build_search_agent()

    # ── 你可以换问题试试 ──
    question = "deepseek r1 和 deepseek v4 有什么区别？"

    print(f"\n📝 用户提问: {question}\n")

    # 初始状态：只有用户问题
    initial = {"user_query": question}

    # stream() 逐步输出每个节点执行后的结果
    # 你会看到: understand 输出 → search 输出 → answer 输出
    for chunk in agent.stream(initial):
        # chunk 格式: {"节点名": {更新后的字段}}
        for node_name, output in chunk.items():
            if node_name == "answer":
                print("\n" + "=" * 55)
                print("📌 最终答案:")
                print("=" * 55)
                print(output.get("final_answer", ""))
