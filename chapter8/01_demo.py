"""
第8章演示 — 记忆与检索（Memory + RAG）

三个 Demo：
  Demo 1: 记忆持久化 — 跨会话记住用户信息
  Demo 2: RAG 检索 — 基于本地知识库回答问题
  Demo 3: 组合使用 — 记忆 + RAG + ReAct Agent

和第7章对比：
  第7章：_history 只在进程内有效，重启就失忆
  第8章：记忆存到 JSON 文件，重启后还能想起来
"""
import sys
import os

sys.path.insert(0, "/home/txh/learn-agent/chapter7")

from dotenv import load_dotenv

load_dotenv("/home/txh/learn-agent/chapter4/.env")

from hello_agents import HelloAgentsLLM

# 第8章新增工具
sys.path.insert(0, "/home/txh/learn-agent/chapter8")
from memory_tool import MemoryTool
from rag_tool import RAGTool


def demo_memory_persistence():
    """Demo 1: 记忆持久化

    和第7章 _history 的对比：
      第7章：agent.run("我叫张三") → 关闭程序 → 再问"我叫什么" → ❌ 忘了
      第8章：memory.save("用户是张三") → 关闭程序 → memory.search("张三") → ✅ 还记得
    """
    print("\n" + "=" * 55)
    print("  Demo 1: 记忆持久化 — 跨会话记忆")
    print("=" * 55)

    # 使用固定路径，模拟"跨会话"
    memory = MemoryTool(file_path="/tmp/ch8_memory_demo.json")

    # 先看看记忆里有什么
    print("\n📭 当前记忆:")
    print(memory.summary())

    # 保存一些记忆（模拟第一次会话）
    print("\n💾 保存记忆...")
    memory.save("用户名叫张三，是一名Python开发者", memory_type="semantic", importance=0.9)
    memory.save("用户正在学习AI Agent开发", memory_type="semantic", importance=0.8)
    memory.save("用户使用的模型是DeepSeek V4 Flash", memory_type="semantic", importance=0.7)
    print("✅ 3条记忆已保存到 JSON 文件")

    # 搜索记忆（模拟第二次会话，重启程序后）
    print("\n🔍 搜索 '张三 学习':")
    results = memory.search("张三 学习 Python")
    for r in results:
        print(f"  • [{r['importance']:.1f}] {r['content'][:60]}")

    print("\n📊 最终摘要:")
    print(memory.summary())

    print("\n💡 关键对比:")
    print("  第7章 _history:  ❌ 重启程序就丢失")
    print("  第8章 MemoryTool: ✅ JSON 文件持久化，重启后仍在")


def demo_rag():
    """Demo 2: RAG — 基于本地知识库回答问题

    核心流程：
      1. 往知识库里添加知识文本
      2. 提问时，RAG 搜索相关段落
      3. 把相关段落作为上下文拼进 Prompt
      4. LLM 基于上下文回答
    """
    print("\n" + "=" * 55)
    print("  Demo 2: RAG — 知识检索增强")
    print("=" * 55)

    rag = RAGTool(kb_path="/tmp/ch8_rag_demo.json")

    # 添加知识
    print("\n📚 添加知识到知识库...")
    rag.add_text(
        """Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。
Python的设计哲学强调代码的可读性和简洁的语法，使用缩进来表示代码块结构。
Python支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。
Python拥有庞大的标准库和活跃的社区生态。""",
        doc_id="python_intro",
    )
    rag.add_text(
        """装饰器（Decorator）是Python的一种高级功能，允许在不修改原函数代码的情况下，
动态地给函数添加新的功能。装饰器本质上是一个接受函数作为参数并返回新函数的高阶函数。
常见的装饰器用法包括日志记录、性能计时、权限检查等。""",
        doc_id="decorator",
    )
    rag.add_text(
        """机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。
主要分为三类：监督学习（有标签数据）、无监督学习（无标签数据）和强化学习（奖励驱动）。
深度学习是机器学习的一个子集，使用多层神经网络来学习复杂的模式。""",
        doc_id="ml_basics",
    )
    print("✅ 知识已添加")

    # 演示 RAG 检索
    query = "Python装饰器怎么用？"
    print(f"\n📝 问题: {query}")

    print("\n🔍 RAG 搜索中...")
    results = rag.search(query, top_k=2)
    print(f"  找到 {len(results)} 个相关段落:")
    for r in results:
        print(f"  • {r['content'][:60]}...")

    print("\n📋 组装 RAG 上下文...")
    context = rag.build_context(query, top_k=2)

    print("\n🧠 现在 LLM 会看到:")
    print("-" * 40)
    print(context)
    print("-" * 40)

    # 真正调 LLM 看效果
    llm = HelloAgentsLLM()
    prompt = f"""{context}

请基于以上资料回答问题：
{query}

如果你觉得资料不足以回答，可以补充你自己的知识。"""
    response = llm.invoke([{"role": "user", "content": prompt}])
    print(f"\n💬 LLM 回答: {response[:300]}...")


def demo_memory_agent():
    """Demo 3: 把记忆工具挂到 ReAct Agent 上

    让 Agent 既能搜索互联网，又能记住用户信息
    """
    print("\n" + "=" * 55)
    print("  Demo 3: Agent + Memory + RAG 组合")
    print("=" * 55)

    # 准备工具
    memory = MemoryTool(file_path="/tmp/ch8_agent_memory.json")
    rag = RAGTool(kb_path="/tmp/ch8_agent_rag.json")

    # 先清空测试数据
    import json
    with open(memory.file_path, "w") as f:
        json.dump([], f)
    with open(rag.kb_path, "w") as f:
        json.dump([], f)

    # 添加一些学习知识
    rag.add_text(
        "ReAct范式让Agent通过Thought→Action→Observation循环来解决问题。"
        "每轮先思考(Thought)，再行动(Action)，最后观察(Observation)结果。"
    )
    rag.add_text(
        "Plan-and-Solve范式分为两个阶段：先由Planner生成完整的执行计划，"
        "再由Executor按步骤逐一执行。适合结构化推理任务。"
    )

    llm = HelloAgentsLLM()

    print("\n📝 第一阶段：对话中保存记忆")
    # 模拟第一次交互
    info_to_remember = "用户正在学习第8章：记忆与检索"
    memory.save(info_to_remember, importance=0.9)
    print(f"  💾 记住了: {info_to_remember}")

    info2 = "用户已经完成了第4章（三大范式）、第6章（LangGraph）、第7章（HelloAgents框架）"
    memory.save(info2, importance=0.8)
    print(f"  💾 记住了: {info2[:40]}...")

    print("\n📝 第二阶段：提问时结合记忆 + RAG")
    query = "解释一下ReAct范式是怎么工作的"

    # 从记忆里找到相关上下文
    mem_results = memory.search("学习 进度 ReAct")
    rag_context = rag.build_context(query)

    # 拼装完整 prompt
    memory_context = ""
    if mem_results:
        memory_context = "关于用户的信息：\n" + "\n".join(
            f"  - {r['content']}" for r in mem_results
        )

    final_prompt = f"""{memory_context}

{rag_context}

请回答以下问题（结合你的知识和以上资料）：
{query}"""

    print(f"\n🧠 最终 Prompt 包含:")
    print(f"  • 用户记忆: {len(mem_results)} 条")
    print(f"  • RAG 知识: {rag_context.count('[资料')} 段")

    response = llm.invoke([{"role": "user", "content": final_prompt}])
    print(f"\n💬 LLM 回答: {response[:300]}...")


if __name__ == "__main__":
    demo_memory_persistence()
    demo_rag()
    demo_memory_agent()

    print("\n" + "=" * 55)
    print("  第8章三个 Demo 全部跑完 ✅")
    print("=" * 55)
    print(f"\n📂 项目文件:")
    print(f"  {os.path.dirname(__file__)}/memory_tool.py  ← 记忆工具")
    print(f"  {os.path.dirname(__file__)}/rag_tool.py     ← RAG 工具")
    print(f"  {os.path.dirname(__file__)}/01_demo.py      ← 本演示")
    print(f"\n📌 关键对比:")
    print(f"  第7章 _history: 内存级，重启即失忆")
    print(f"  第8章 MemoryTool: JSON持久化，跨会话")
    print(f"  第8章 RAGTool: 知识检索，让LLM基于本地资料回答")
