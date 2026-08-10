"""
══════════════════════════════════════════════════
  03 ─ 交互式问答助手（连续对话版）
══════════════════════════════════════════════════

  新增能力：
    - 循环提问，输入 quit 退出
    - 每次重新走一遍 理解→搜索→回答 流程
    - 显示每步耗时时长
"""

import os
import time
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient

# ── 加载配置 ──
load_dotenv("/home/txh/learn-agent/chapter4/.env")

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "deepseek-v4-flash"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
    temperature=0.3,
)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ══════════════════════════════════════════════
#  State 定义（没变）
# ══════════════════════════════════════════════

class SearchState(TypedDict):
    user_query: str
    search_query: str
    search_results: str
    final_answer: str
    step: str


# ══════════════════════════════════════════════
#  Node 定义（加了耗时统计）
# ══════════════════════════════════════════════

def understand_node(state: SearchState) -> dict:
    t0 = time.time()
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

    search_q = user_msg
    if "搜索词：" in text:
        search_q = text.split("搜索词：")[1].strip()

    elapsed = time.time() - t0
    print(f"  ├ 理解: {search_q}")
    print(f"  └ ⏱ {elapsed:.1f}s")

    return {"user_query": text, "search_query": search_q, "step": "understood"}


def search_node(state: SearchState) -> dict:
    t0 = time.time()
    query = state["search_query"]

    try:
        result = tavily.search(
            query=query, search_depth="basic", max_results=5, include_answer=True,
        )
        results_text = f"摘要: {result.get('answer', '无')}\n\n"
        for i, r in enumerate(result.get("results", []), 1):
            results_text += f"{i}. {r['title']}\n   {r['content'][:200]}...\n"

        elapsed = time.time() - t0
        print(f"  ├ 搜索到 {len(result.get('results', []))} 条结果")
        print(f"  └ ⏱ {elapsed:.1f}s")

        return {"search_results": results_text, "step": "searched"}
    except Exception as e:
        print(f"  └ ❌ 搜索失败: {e}")
        return {"search_results": f"搜索出错: {e}", "step": "search_failed"}


def answer_node(state: SearchState) -> dict:
    t0 = time.time()

    if state["step"] == "search_failed":
        prompt = f"搜索暂时不可用，请基于你的知识回答：{state['user_query']}"
    else:
        prompt = f"""基于以下搜索结果回答用户问题。

用户问题：{state['user_query']}

搜索结果：
{state['search_results']}

请给出完整、准确、有条理的回答："""

    response = llm.invoke([SystemMessage(content=prompt)])

    elapsed = time.time() - t0
    print(f"  └ ⏱ {elapsed:.1f}s")

    return {"final_answer": response.content, "step": "completed"}


# ══════════════════════════════════════════════
#  构建图（没变，还是线性三步）
# ══════════════════════════════════════════════

def build_search_agent():
    workflow = StateGraph(SearchState)
    workflow.add_node("understand", understand_node)
    workflow.add_node("search", search_node)
    workflow.add_node("answer", answer_node)
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)
    return workflow.compile()


# ══════════════════════════════════════════════
#  交互循环（新增！）
# ══════════════════════════════════════════════

def main():
    agent = build_search_agent()

    print("=" * 55)
    print("  🤖 LangGraph 交互式问答助手")
    print("  输入问题，按回车搜索")
    print("  输入 quit 退出")
    print("=" * 55)

    while True:
        question = input("\n📝 你的问题 > ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        print(f"\n── 处理中 ──")

        initial = {"user_query": question}
        total_t0 = time.time()

        # 只迭代一次，收集所有节点结果
        final_answer = None
        for chunk in agent.stream(initial):
            for node_name, output in chunk.items():
                if node_name == "understand":
                    print(f"\n🧠 理解意图")
                elif node_name == "search":
                    print(f"\n🔍 搜索信息")
                elif node_name == "answer":
                    print(f"\n💡 生成答案")
                    final_answer = output.get("final_answer", "")

        total = time.time() - total_t0

        if final_answer:
            print(f"\n{'=' * 55}")
            print(final_answer)
            print(f"\n⏱ 总耗时: {total:.1f}s")
            print("=" * 55)


if __name__ == "__main__":
    main()
