"""
memory_tool.py — 为 HelloAgents 框架添加长期记忆能力

对比第7章的 _history（仅内存）：
  _history ← 程序重启就没了
  MemoryTool → 存到 JSON 文件，跨会话保留

核心设计：
  每次 save() 把记忆追加到 JSON 文件
  每次 search() 从 JSON 文件里按关键词搜索
"""
import json
import os
import re
from datetime import datetime
from typing import Optional


class MemoryTool:
    """简单的 JSON 持久化记忆工具

    用法：
        memory = MemoryTool(file_path="memories.json")
        memory.save("用户张三，正在学Python", memory_type="semantic")
        results = memory.search("Python")
    """

    def __init__(self, file_path: str = None):
        self.file_path = file_path or os.path.join(
            os.path.dirname(__file__), "memory_store.json"
        )
        self._ensure_file()

    def _ensure_file(self):
        """确保 JSON 文件存在"""
        if not os.path.exists(self.file_path):
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def _load_all(self) -> list:
        """读取所有记忆"""
        with open(self.file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_all(self, records: list):
        """覆写所有记忆"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    def save(
        self,
        content: str,
        memory_type: str = "semantic",
        importance: float = 0.5,
        source: str = "user",
    ) -> dict:
        """保存一条记忆

        Args:
            content: 记忆内容
            memory_type: working(临时) / semantic(长期)
            importance: 0.0~1.0 重要程度
            source: 信息来源 (user/assistant/system)
        """
        record = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "content": content,
            "type": memory_type,
            "importance": importance,
            "source": source,
            "created_at": datetime.now().isoformat(),
        }
        records = self._load_all()
        records.append(record)
        self._save_all(records)
        return record

    def _extract_keywords(self, text: str) -> set[str]:
        """提取关键词（同时处理中英文）"""
        import unicodedata

        stop_words = {
            "的", "了", "是", "在", "和", "也", "就", "都", "而",
            "与", "着", "或", "不", "会", "能", "要",
            "the", "a", "an", "is", "are", "was", "were",
            "it", "its", "this", "that", "these", "those",
        }

        def is_chinese(ch):
            try:
                return unicodedata.name(ch).startswith("CJK")
            except:
                return False

        text = text.lower()
        # 英文/数字词
        en_words = re.findall(r"[a-z0-9]+", text)
        # 中文字符
        cn_chars = [ch for ch in text if is_chinese(ch)]

        all_words = en_words + cn_chars
        return {w for w in all_words if w not in stop_words and len(w) > 0}

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """搜索记忆（关键词匹配）"""
        query_keywords = self._extract_keywords(query)

        records = self._load_all()

        # 计算每条记忆的匹配分数
        scored = []
        for r in records:
            content_keywords = self._extract_keywords(r["content"])
            overlap = query_keywords & content_keywords
            if overlap:
                # 加权：importance 越高排名越靠前
                score = len(overlap) * (0.5 + r["importance"] * 0.5)
                scored.append((score, r))

        # 按分数从高到低排序
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]

    def summary(self) -> str:
        """生成记忆摘要"""
        records = self._load_all()
        if not records:
            return "📭 暂无记忆"

        total = len(records)
        by_type = {}
        for r in records:
            t = r["type"]
            by_type[t] = by_type.get(t, 0) + 1

        lines = [
            f"📊 记忆统计",
            f"  总共: {total} 条",
        ]
        for t, c in by_type.items():
            lines.append(f"  {t}: {c} 条")

        # 显示最近3条
        lines.append(f"\n📝 最近记忆:")
        for r in records[-3:]:
            lines.append(f"  • {r['content'][:60]}...")

        return "\n".join(lines)

    def get_tool_description(self) -> str:
        """给 LLM 看的工具描述（供 ReActAgent 使用）"""
        return """记忆工具 — 记住用户信息和学习历史
用法: Memory[save: 内容, 类型]
  或: Memory[search: 关键词]
类型: semantic(事实/偏好), working(临时)

示例: Memory[save: 用户喜欢Python, semantic]
示例: Memory[search: Python]"""
