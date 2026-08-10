"""
第9章 演示 — 上下文工程实战

三个 Demo:
  1. ContextBuilder GSSC 流水线
  2. NoteTool 结构化笔记
  3. 组合: 用 ContextBuilder 管理 NoteTool + Memory + RAG

前提: 确保 chapter8/memory_tool.py 和 chapter8/rag_tool.py 可导入
"""

import sys
import os

# 让 Python 能找到 chapter8 的工具
sys.path.insert(0, "/home/txh/learn-agent/chapter8")
sys.path.insert(0, "/home/txh/learn-agent/chapter9")

from context_builder import ContextBuilder, ContextConfig, ContextPacket
from note_tool import NoteTool
from datetime import datetime, timedelta
import tempfile
import shutil


def demo1_context_builder():
    """Demo 1: GSSC 流水线"""
    print("\n" + "=" * 55)
    print("  Demo 1: ContextBuilder — GSSC 流水线")
    print("=" * 55)

    # 模拟各种数据源
    memory_packets = [
        ContextPacket(
            content="用户叫张三，正在学习AI Agent开发，当前学习到第9章",
            source="memory", relevance_score=0.9, token_count=18,
        ),
        ContextPacket(
            content="张三之前完成了第8章 Memory + RAG 的学习",
            source="memory", relevance_score=0.75, token_count=14,
        ),
    ]

    rag_packets = [
        ContextPacket(
            content="上下文工程 (Context Engineering) 是在推理阶段策划和维护最优 token 集合的方法论。核心是 GSSC 流水线。",
            source="rag", relevance_score=0.85, token_count=28,
        ),
    ]

    history = [
        ("user", "上下文工程和提示工程有什么区别？", datetime.now() - timedelta(minutes=5)),
        ("assistant", "提示工程关注怎么写提示词，上下文工程关注怎么管理整个上下文状态", datetime.now() - timedelta(minutes=4)),
    ]

    # 构建上下文
    builder = ContextBuilder(ContextConfig(max_tokens=500))

    context = builder.build(
        user_query="GSSC流水线是什么？",
        system_instructions="你是一位AI Agent架构师，擅长用简单类比解释复杂概念。",
        memory_packets=memory_packets,
        rag_packets=rag_packets,
        conversation_history=history,
    )

    print("\n📋 最终上下文:")
    print("-" * 40)
    print(context)

    print("\n💡 关键观察:")
    print("  · 系统指令([Role & Policies])始终保留")
    print("  · 相关性高的 RAG 数据进入了 [Evidence]")
    print("  · 历史信息进入了 [Context]")
    print("  · 低相关性或超限数据被自动丢弃")


def demo2_note_tool():
    """Demo 2: NoteTool 结构化笔记"""
    print("\n" + "=" * 55)
    print("  Demo 2: NoteTool — 项目笔记管理")
    print("=" * 55)

    tmpdir = tempfile.mkdtemp(prefix="ch9_notetool_")
    notes = NoteTool(workspace=tmpdir)

    # 模拟一个代码库维护项目的笔记轨迹
    print("\n📝 第一天：探索阶段...")
    notes.create(
        "项目结构初探",
        "## 发现\n- Flask Web应用，约50个Python文件\n- 主要目录: models/, services/, api/",
        note_type="task_state",
        tags=["exploration", "day1"],
    )

    print("\n📝 发现阻塞问题...")
    notes.create(
        "User模型缺少email唯一约束",
        "## 问题\nUser.email 字段没有 unique=True，可能导致重复注册\n\n## 建议\n添加数据库迁移",
        note_type="blocker",
        tags=["database", "high_priority"],
    )

    print("\n📝 记录结论...")
    notes.create(
        "第一天总结",
        "## 结论\n代码结构清晰，但存在一些技术债务\n\n## 优先级\n1. 添加唯一约束 (高)\n2. 提取BaseService减少重复 (中)",
        note_type="conclusion",
        tags=["day1"],
    )

    # 展示检索和统计
    print("\n🔍 搜索 '约束'...")
    for r in notes.search("约束"):
        print(f"  [{r['type']}] {r['title']}")

    print("\n📊 统计摘要:")
    s = notes.summary()
    print(f"  总笔记: {s['total_notes']}")
    print(f"  类型分布: {s['type_distribution']}")
    for n in s['recent_notes']:
        print(f"  · [{n['type']}] {n['title']}")

    shutil.rmtree(tmpdir)


def demo3_context_with_notes():
    """Demo 3: ContextBuilder + NoteTool 组合"""
    print("\n" + "=" * 55)
    print("  Demo 3: ContextBuilder × NoteTool — 组合实战")
    print("=" * 55)

    tmpdir = tempfile.mkdtemp(prefix="ch9_combined_")
    notes = NoteTool(workspace=tmpdir)

    # 模拟已经有一些笔记
    notes.create(
        "重构进展 - 第一阶段",
        "已完成数据模型层重构，测试覆盖率85%",
        note_type="task_state", tags=["refactoring", "phase1"],
    )
    notes.create(
        "依赖冲突问题",
        "pandas 2.0 与 SQLAlchemy 1.4 不兼容，需要升级SQLAlchemy到2.0",
        note_type="blocker", tags=["dependency", "urgent"],
    )

    # 把笔记包装成 ContextPacket
    note_results = notes.search("依赖")
    note_packets = []
    for r in note_results:
        note_packets.append(ContextPacket(
            content=f"[笔记:{r['title']}]({r['type']})\n{r['content']}",
            source="note",
            relevance_score=0.9 if r['type'] == 'blocker' else 0.7,
            token_count=30,
        ))

    # 再加一些记忆
    memory_packets = [
        ContextPacket(
            content="用户在维护一个Flask Web应用代码库",
            source="memory", relevance_score=0.8, token_count=12,
        ),
    ]

    # 构建上下文
    builder = ContextBuilder(ContextConfig(max_tokens=400))
    context = builder.build(
        user_query="如何解决pandas和SQLAlchemy的版本冲突？",
        system_instructions="你是代码库维护助手，回答要结合项目上下文。",
        memory_packets=memory_packets,
        note_packets=note_packets,
    )

    print("\n📋 带笔记的上下文:")
    print("-" * 40)
    print(context)

    print("\n💡 重点: blocker 类型的笔记进入了 [Notes] 区域")
    print("   在 ContextBuilder 中，blocker 获得了更高的相关性分数(0.9)")
    print("   这样 LLM 在回答时会优先关注阻塞问题")

    shutil.rmtree(tmpdir)


if __name__ == "__main__":
    demo1_context_builder()
    demo2_note_tool()
    demo3_context_with_notes()

    print("\n" + "=" * 55)
    print("  第9章 Demo 全部完成！")
    print("=" * 55)
