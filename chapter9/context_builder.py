"""
第9章: ContextBuilder — GSSC 流水线 (精简版)

GSSC = Gather(收集) → Select(筛选) → Structure(结构化) → Compress(压缩)

核心思想:
  每次调用 LLM 前，不是把所有东西都塞进上下文，
  而是像"漏斗"一样：先大量收集 → 智能筛选 → 结构化 → 必要时压缩

对比第8章:
  第8章的 MemoryTool/RAGTool 是"数据源"
  第9章的 ContextBuilder 是"数据管家"——决定哪些数据进上下文
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Callable
import math
import re


# ============================================================
# 1. 数据单元：ContextPacket
# ============================================================

@dataclass
class ContextPacket:
    """上下文信息包 — 流水线上的基本数据单元"""
    content: str                              # 信息内容
    source: str = "unknown"                   # 来源 (memory/rag/note/history)
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.5              # 相关性分数 0.0~1.0
    priority: str = "normal"                  # high / normal / low
    token_count: int = 0                      # 预估 token 数
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 2. 配置
# ============================================================

@dataclass
class ContextConfig:
    """上下文构建器的配置参数"""
    max_tokens: int = 3000                    # 上下文预算上限
    reserve_ratio: float = 0.2                # 给系统指令预留的比例
    min_relevance: float = 0.15               # 最低相关性过滤阈值
    recency_weight: float = 0.3               # 新近性权重
    relevance_weight: float = 0.7             # 相关性权重
    enable_compression: bool = True           # 是否启用压缩

    def __post_init__(self):
        assert 0.0 <= self.reserve_ratio <= 1.0
        assert self.recency_weight + self.relevance_weight - 1.0 < 1e-6


# ============================================================
# 3. ContextBuilder — GSSC 流水线
# ============================================================

class ContextBuilder:
    """
    上下文构建器：GSSC 四阶段流水线

    用法:
        builder = ContextBuilder(config=ContextConfig(max_tokens=3000))
        context = builder.build(
            user_query="如何优化Pandas内存?",
            system_instructions="你是一个Python顾问",
            memory_packets=[...],    # 来自 MemoryTool
            rag_packets=[...],       # 来自 RAGTool
            note_packets=[...],      # 来自 NoteTool
            conversation_history=[(role, content, timestamp), ...]
        )
    """

    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()

    def build(
        self,
        user_query: str,
        system_instructions: Optional[str] = None,
        memory_packets: Optional[List[ContextPacket]] = None,
        rag_packets: Optional[List[ContextPacket]] = None,
        note_packets: Optional[List[ContextPacket]] = None,
        conversation_history: Optional[List[tuple]] = None,
    ) -> str:
        """GSSC 流水线主入口"""
        print(f"\n{'='*50}")
        print(f"📦 ContextBuilder 启动 (预算: {self.config.max_tokens} tokens)")
        print(f"{'='*50}")

        # ====== Phase 1: GATHER ======
        raw_packets = self._gather(
            user_query, system_instructions,
            memory_packets, rag_packets, note_packets,
            conversation_history
        )
        print(f"  [Gather] 收集了 {len(raw_packets)} 个候选包")

        # ====== Phase 2: SELECT ======
        selected = self._select(raw_packets, user_query)
        print(f"  [Select] 选出了 {len(selected)} 个包")
        for p in selected:
            print(f"    · [{p.source}] score={p.relevance_score:.2f}  {p.content[:60]}...")

        # ====== Phase 3: STRUCTURE ======
        structured = self._structure(selected, user_query)
        print(f"  [Structure] 结构化完成 ({len(structured)} 字符)")

        # ====== Phase 4: COMPRESS ======
        final = self._compress(structured)
        print(f"  [Compress] 最终上下文 ({len(final)} 字符, ~{self._count_tokens(final)} tokens)")
        print(f"{'='*50}\n")

        return final

    # ----- Phase 1: Gather -----

    def _gather(
        self,
        user_query: str,
        system_instructions: Optional[str] = None,
        memory_packets: Optional[List[ContextPacket]] = None,
        rag_packets: Optional[List[ContextPacket]] = None,
        note_packets: Optional[List[ContextPacket]] = None,
        conversation_history: Optional[List[tuple]] = None,
    ) -> List[ContextPacket]:
        """阶段1：从多个来源汇集候选信息"""
        packets = []

        # 1. 系统指令（最高优先级）
        if system_instructions:
            packets.append(ContextPacket(
                content=system_instructions,
                source="system",
                relevance_score=1.0,
                priority="high",
                token_count=self._count_tokens(system_instructions),
            ))

        # 2. 记忆包
        if memory_packets:
            for p in memory_packets:
                if p.relevance_score >= self.config.min_relevance:
                    packets.append(p)

        # 3. RAG 包
        if rag_packets:
            for p in rag_packets:
                if p.relevance_score >= self.config.min_relevance:
                    packets.append(p)

        # 4. 笔记包
        if note_packets:
            for p in note_packets:
                if p.relevance_score >= self.config.min_relevance:
                    packets.append(p)

        # 5. 对话历史（最近N条）
        if conversation_history:
            recent = conversation_history[-5:]  # 只保留最近5轮
            for role, content, ts in recent:
                packets.append(ContextPacket(
                    content=f"[{role}] {content}",
                    source="history",
                    timestamp=ts,
                    relevance_score=0.5,
                    token_count=self._count_tokens(content),
                ))

        return packets

    # ----- Phase 2: Select -----

    def _select(
        self,
        packets: List[ContextPacket],
        user_query: str,
    ) -> List[ContextPacket]:
        """阶段2：基于相关性和新近性评分，在预算内选择最优子集"""
        # 分离系统指令
        system_packets = [p for p in packets if p.priority == "high"]
        other_packets = [p for p in packets if p.priority != "high"]

        system_tokens = sum(p.token_count for p in system_packets)
        available = self.config.max_tokens - system_tokens

        if available <= 0:
            return system_packets

        # 为其他包计算综合分数
        query_keywords = set(self._tokenize(user_query))

        scored = []
        for p in other_packets:
            # 如果还没有相关性分数，计算一个
            if p.relevance_score == 0.5 and user_query:
                p.relevance_score = self._calc_relevance(p.content, query_keywords)

            # 新近性分数（指数衰减）
            recency = self._calc_recency(p.timestamp)

            # 综合分数 = 加权组合
            combined = (
                self.config.relevance_weight * p.relevance_score
                + self.config.recency_weight * recency
            )

            if p.relevance_score >= self.config.min_relevance:
                scored.append((combined, p))

        # 排序 + 贪心选择
        scored.sort(key=lambda x: x[0], reverse=True)

        selected = system_packets.copy()
        current_tokens = system_tokens

        for score, p in scored:
            if current_tokens + p.token_count <= available:
                selected.append(p)
                current_tokens += p.token_count
            else:
                break

        return selected

    def _calc_relevance(self, content: str, query_keywords: set) -> float:
        """计算内容与查询的相关性（Jaccard 相似度）"""
        content_words = set(self._tokenize(content))
        if not query_keywords or not content_words:
            return 0.0
        intersection = content_words & query_keywords
        union = content_words | query_keywords
        return len(intersection) / len(union) if union else 0.0

    def _calc_recency(self, timestamp: datetime) -> float:
        """新近性分数：指数衰减，24小时内保持高分"""
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        return max(0.1, min(1.0, math.exp(-0.1 * age_hours / 24)))

    def _tokenize(self, text: str) -> List[str]:
        """简单的分词（中英文兼容）"""
        # 英文单词
        words = re.findall(r'[a-zA-Z0-9_]+', text.lower())
        # 中文字符（拆单字）
        chars = re.findall(r'[\u4e00-\u9fff]', text)
        return words + chars

    # ----- Phase 3: Structure -----

    def _structure(
        self,
        packets: List[ContextPacket],
        user_query: str,
    ) -> str:
        """阶段3：将选中的包组织为结构化上下文"""
        sections = {
            "system": [],
            "evidence": [],
            "context": [],
            "notes": [],
        }

        for p in packets:
            if p.source == "system":
                sections["system"].append(p.content)
            elif p.source in ("rag", "knowledge"):
                sections["evidence"].append(p.content)
            elif p.source == "note":
                sections["notes"].append(p.content)
            else:
                sections["context"].append(p.content)

        lines = []

        if sections["system"]:
            lines.append("[Role & Policies]")
            lines.extend(sections["system"])
            lines.append("")

        lines.append("[Task]")
        lines.append(user_query)
        lines.append("")

        if sections["evidence"]:
            lines.append("[Evidence]")
            lines.append("\n---\n".join(sections["evidence"]))
            lines.append("")

        if sections["notes"]:
            lines.append("[Notes]")
            lines.append("\n".join(sections["notes"]))
            lines.append("")

        if sections["context"]:
            lines.append("[Context]")
            lines.append("\n".join(sections["context"]))
            lines.append("")

        lines.append("[Output]")
        lines.append("请基于以上信息，提供准确、有据的回答。")

        return "\n".join(lines)

    # ----- Phase 4: Compress -----

    def _compress(self, context: str) -> str:
        """阶段4：如果超限，进行压缩"""
        if not self.config.enable_compression:
            return context

        current = self._count_tokens(context)
        if current <= self.config.max_tokens:
            return context

        # 分区压缩：按段落截断
        sections = context.split("\n\n")
        compressed = []
        total = 0

        for sec in sections:
            tokens = self._count_tokens(sec)
            if total + tokens <= self.config.max_tokens:
                compressed.append(sec)
                total += tokens
            else:
                remaining = self.config.max_tokens - total
                if remaining > 50:
                    # 按比例截断
                    ratio = remaining / tokens
                    chars = int(len(sec) * ratio)
                    compressed.append(sec[:chars] + "\n[... 已压缩 ...]")
                break

        return "\n\n".join(compressed)

    # ----- Helper -----

    def _count_tokens(self, text: str) -> int:
        """粗略估算 token 数（中文1字≈1token，英文1词≈1.3token）"""
        chinese = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        english = len([w for w in text.split() if w])
        return int(chinese + english * 1.3)


# ============================================================
# 4. 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  ContextBuilder 演示 — GSSC 流水线")
    print("=" * 55)

    # 创建一些模拟数据包
    memory_packets = [
        ContextPacket(
            content="用户正在开发数据分析工具，使用Python和Pandas",
            source="memory", relevance_score=0.85,
            token_count=20,
        ),
        ContextPacket(
            content="已完成CSV读取模块的开发",
            source="memory", relevance_score=0.7,
            token_count=10,
        ),
    ]

    rag_packets = [
        ContextPacket(
            content="Pandas内存优化策略：1) 使用category类型 2) 分块读取 3) chunksize参数",
            source="rag", relevance_score=0.9,
            token_count=25,
        ),
    ]

    # 对话历史
    history = [
        ("user", "我正在开发数据分析工具", datetime.now() - timedelta(minutes=10)),
        ("assistant", "很好！你用什么技术栈？", datetime.now() - timedelta(minutes=9)),
        ("user", "Python和Pandas，已完成CSV读取", datetime.now() - timedelta(minutes=8)),
    ]

    # 创建 ContextBuilder
    builder = ContextBuilder(ContextConfig(max_tokens=500))

    context = builder.build(
        user_query="如何优化Pandas的内存占用？",
        system_instructions="你是一位资深的Python数据工程顾问。回答需要: 1) 具体可行 2) 解释原理 3) 给出代码示例",
        memory_packets=memory_packets,
        rag_packets=rag_packets,
        conversation_history=history,
    )

    print("\n最终上下文字符串：")
    print("-" * 50)
    print(context)
