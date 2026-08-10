"""
第8章 A/B 对比实验 — Memory vs 无Memory / RAG vs 无RAG

用同一个问题对比"加工具"和"不加工具"的效果差异。
这就是你学得最快的方式 — 亲眼看到工具的作用。
"""
import sys
import os
import json

sys.path.insert(0, "/home/txh/learn-agent/chapter7")
sys.path.insert(0, "/home/txh/learn-agent/chapter8")

from dotenv import load_dotenv
load_dotenv("/home/txh/learn-agent/chapter4/.env")

from hello_agents import HelloAgentsLLM
from memory_tool import MemoryTool
from rag_tool import RAGTool

llm = HelloAgentsLLM()

print("=" * 62)
print("  第8章 A/B 对比实验")
print("  同一个问题 → 有工具 vs 没工具 → 看差异")
print("=" * 62)


# ═══════════════════════════════════════════════
# 实验1：Memory — 有记忆 vs 没记忆
# ═══════════════════════════════════════════════
print("\n" + "─" * 62)
print("  【实验1】记忆系统 — 有记忆 vs 没记忆")
print("─" * 62)

# 情景：用户之前说过"我叫张三，是一名Python开发者"
# 第二次会话时问"我叫什么名字？"

# A组：没有记忆（模拟第7章的情况 — 重启后 _history 被清空）
print("\n  🅰 没有记忆工具（第7章模式 — 重启失忆）:")
print("  ─────────────────────────────────────")
response_a = llm.invoke([{"role": "user", "content": "我叫什么名字？"}])
print(f"  LLM: {response_a[:100]}")
print("  → ❌ Agent 不知道你是谁")

# B组：有记忆（第8章 MemoryTool — 从 JSON 文件回忆）
print("\n  🅱 使用记忆工具（第8章 MemoryTool — 跨会话记忆）:")
print("  ─────────────────────────────────────────────")
memory = MemoryTool(file_path="/tmp/ch8_ab_memory.json")

# 清空并写入记忆（模拟第一次会话存下来的数据）
with open(memory.file_path, "w", encoding="utf-8") as f:
    json.dump([], f, ensure_ascii=False, indent=2)

memory.save("用户名叫张三，是一名Python开发者", importance=0.9)
memory.save("用户正在学习AI Agent开发", importance=0.8)

# 第二次会话：搜索记忆并注入 prompt
mem_results = memory.search("我叫什么 名字")
memory_context = "关于用户的信息：\n" + "\n".join(
    f"  - {r['content']}" for r in mem_results
)

prompt_b = f"""{memory_context}

请回答以下问题：
我叫什么名字？"""

response_b = llm.invoke([{"role": "user", "content": prompt_b}])
print(f"  从记忆中找到: {[r['content'][:40] for r in mem_results]}")
print(f"  LLM: {response_b[:100]}")
print("  → ✅ Agent 还记得你是谁")

# 总结
print("\n  📌 本质区别:")
print("     第7章 _history → 存在进程内存里，程序退出就没了")
print("     第8章 MemoryTool → 存在 JSON 文件里，下次启动还能读")
print("     ← 这就是「持久化」的意思")


# ═══════════════════════════════════════════════
# 实验2：RAG — 有知识库 vs 没知识库
# ═══════════════════════════════════════════════
print("\n" + "─" * 62)
print("  【实验2】RAG — 有知识库 vs 没知识库")
print("─" * 62)

# 问题是一个 LLM 可能不熟悉的细节知识
question = """请解释一下 Python 中 @dataclass 的 frozen=True 参数有什么作用？
给一个简单的代码示例。"""

# A组：没有 RAG，纯靠 LLM 训练知识
print("\n  🅰 没有 RAG 知识库 — 纯靠 LLM 训练数据:")
print("  ──────────────────────────────────────")
response_a2 = llm.invoke([{"role": "user", "content": question}])
print(f"  LLM: {response_a2[:200]}")
print("  → LLM 凭训练数据回答（可能不准/泛泛）")

# B组：有 RAG，从知识库获取参考资料
print("\n  🅱 有 RAG 知识库 — 先查资料再回答:")
print("  ─────────────────────────────────")
rag = RAGTool(kb_path="/tmp/ch8_ab_rag.json")

# 清空并写入自己的知识
with open(rag.kb_path, "w", encoding="utf-8") as f:
    json.dump([], f, ensure_ascii=False, indent=2)

rag.add_text(
    """@dataclass 是 Python 3.7+ 引入的装饰器，自动生成 __init__、__repr__、__eq__ 等方法。
frozen=True 时，生成的类实例变为不可变（类似元组），
任何尝试修改属性的操作都会引发 FrozenInstanceError。""",
    doc_id="dataclass_frozen",
)

rag.add_text(
    """使用 frozen=True 的 dataclass 示例：
from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    x: int
    y: int

p = Point(1, 2)
print(p.x)  # 1
p.x = 3     # ❌ 引发 FrozenInstanceError""",
    doc_id="dataclass_example",
)

rag_context = rag.build_context(question, top_k=2)
prompt_b2 = f"""{rag_context}

请基于以上资料回答问题：
{question}"""

print(f"  知识库中有 {len(rag._load_all())} 个知识块")
response_b2 = llm.invoke([{"role": "user", "content": prompt_b2}])
print(f"  RAG 注入的上下文:\n{rag_context[:200]}...")
print(f"\n  LLM: {response_b2[:200]}")
print("  → ✅ 回答基于具体资料，更准确、可验证")

print("\n  📌 本质区别:")
print("     无RAG: LLM 凭训练数据回答（可能过时、泛泛）")
print("     有RAG: 先查本地知识 → 再把资料给 LLM → LLM 基于资料回答")
print("     ← 这就是「检索增强生成」的意思")


# ═══════════════════════════════════════════════
# 实验3：组合 — 没工具 vs 全工具
# ═══════════════════════════════════════════════
print("\n" + "─" * 62)
print("  【实验3】终极对比 — 裸 LLM vs 记忆+RAG 全开")
print("─" * 62)

question3 = "介绍Python装饰器的基本用法"

# A组：裸 LLM
print("\n  🅰 裸 LLM（无记忆、无 RAG、无上下文）:")
response_a3 = llm.invoke([{"role": "user", "content": question3}])
print(f"  {response_a3[:200]}...")

# B组：LLM + Memory（记住了用户背景）+ RAG（有知识库）
print("\n  🅱 LLM + 记忆（用户背景）+ RAG（知识资料）:")
mem_results3 = memory.search("学习 Python Agent")
rag_context3 = rag.build_context(question3)

prompt_b3 = ""
if mem_results3:
    prompt_b3 += "关于用户的信息：\n" + "\n".join(
        f"  - {r['content']}" for r in mem_results3
    ) + "\n\n"

prompt_b3 += rag_context3
if rag_context3:
    prompt_b3 += "\n"
prompt_b3 += f"请回答以下问题（结合你的知识和以上资料）：\n{question3}"

print(f"  🧠 注入的用户记忆: {[r['content'][:30] for r in mem_results3]}")
print(f"  🧠 注入的RAG知识: {rag_context3[:150]}...")

response_b3 = llm.invoke([{"role": "user", "content": prompt_b3}])
print(f"\n  {response_b3[:300]}...")

print("\n" + "=" * 62)
print("  对比实验总结")
print("=" * 62)
print("""
  Memory（记忆）= 记用户说了什么
    比如：用户名、偏好、学习进度
    作用：让 Agent 在下次会话还记得你
    
  RAG（检索增强生成）= 查提前准备好的资料
    比如：产品文档、技术手册、学习笔记
    作用：让 Agent 能回答训练数据之外的问题
    
  第7章 _history → 只存在内存里，一次会话
  第8章 Memory  → 存到 JSON 文件，跨会话
  第8章 RAG     → 从知识库检索，让答案有据可查
""")

print("📂 本实验文件: /home/txh/learn-agent/chapter8/02_ab_comparison.py")
