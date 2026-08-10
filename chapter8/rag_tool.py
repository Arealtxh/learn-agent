"""
rag_tool.py — 为 HelloAgents 框架添加 RAG（检索增强生成）能力

核心思想：
  在问 LLM 之前，先查本地知识库，把相关段落塞进 prompt

工作流程：
  1. add_text() — 添加知识到知识库（JSON 存储）
  2. search() — 根据问题搜索最相关的知识段落
  3. build_context() — 把搜索结果拼成 prompt 上下文

对比高级 RAG（Qdrant + 向量嵌入）：
  这里先用 TF-IDF 风格的简单关键词匹配
  理解了核心概念后，再升级到向量搜索
"""
import json
import os
import re
from datetime import datetime
from typing import Optional
from collections import Counter
import math


class RAGTool:
    """简单的 RAG 工具 — 基于关键词匹配的知识检索

    用法：
        rag = RAGTool(kb_path="knowledge_base.json")
        rag.add_text("Python 是一种解释型语言...", doc_id="python_intro")
        results = rag.search("Python怎么用")
        context = rag.build_context("Python怎么用")
    """

    def __init__(self, kb_path: str = None):
        self.kb_path = kb_path or os.path.join(
            os.path.dirname(__file__), "knowledge_base.json"
        )
        self._ensure_file()

    def _ensure_file(self):
        """确保知识库 JSON 文件存在"""
        if not os.path.exists(self.kb_path):
            os.makedirs(os.path.dirname(self.kb_path), exist_ok=True)
            with open(self.kb_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def _load_all(self) -> list:
        with open(self.kb_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_all(self, records: list):
        with open(self.kb_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    def _chunk_text(self, text: str, chunk_size: int = 200) -> list[str]:
        """把长文本切分成小块（简单按句子切）

        Args:
            text: 原始文本
            chunk_size: 每块大约多少字符
        """
        # 先把文本按句号、问号、感叹号拆成句子
        sentences = re.split(r"(?<=[。！？.!?])\s*", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current = ""
        for s in sentences:
            if len(current) + len(s) < chunk_size:
                current += s
            else:
                if current:
                    chunks.append(current)
                current = s
        if current:
            chunks.append(current)
        return chunks

    def add_text(
        self,
        text: str,
        doc_id: str = None,
        source: str = "manual",
        auto_chunk: bool = True,
    ) -> dict:
        """添加知识文本

        Args:
            text: 知识内容
            doc_id: 文档ID（自动生成）
            source: 来源标记
            auto_chunk: 是否自动分块
        """
        doc_id = doc_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        if auto_chunk and len(text) > 300:
            chunks = self._chunk_text(text)
        else:
            chunks = [text]

        records = self._load_all()
        saved = []
        for i, chunk in enumerate(chunks):
            record = {
                "chunk_id": f"{doc_id}_chunk_{i}",
                "doc_id": doc_id,
                "content": chunk,
                "source": source,
                "created_at": datetime.now().isoformat(),
            }
            records.append(record)
            saved.append(record)

        self._save_all(records)
        return {
            "doc_id": doc_id,
            "chunks": len(saved),
            "message": f"已添加 {len(saved)} 个知识块",
        }

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """搜索最相关的知识段落（TF-IDF 风格关键词匹配）

        这里用简单的 TF-IDF 权重计算：
          TF = 关键词在段落中出现的次数
          IDF = log(总段落数 / 包含该关键词的段落数)

        让常见词（如"是"、"的"）权重降低
        让罕见词（如"装饰器"、"协程"）权重升高
        """
        import math

        records = self._load_all()
        if not records:
            return []

        # 预处理查询
        query_keywords = self._extract_keywords(query)

        # 计算 IDF
        n_docs = len(records)
        idf = {}
        for kw in query_keywords:
            docs_with_kw = sum(1 for r in records if kw in r["content"].lower())
            idf[kw] = math.log((n_docs + 1) / (docs_with_kw + 1)) + 1

        # 计算每条记录的 TF-IDF 分数
        scored = []
        for r in records:
            content_lower = r["content"].lower()
            score = 0
            for kw in query_keywords:
                tf = content_lower.count(kw)
                score += tf * idf.get(kw, 0)
            if score > 0:
                scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    def _extract_keywords(self, text: str) -> list[str]:
        """提取关键词

        同时处理中英文：
          - 英文：按空格分词
          - 中文：按字符拆分（简单但有效）
        
        注意：这里没有用 jieba 分词，因为要保持零依赖
        以后可以升级为 jieba 或 LLM 分词
        """
        import unicodedata

        stop_words = {
            "的", "了", "是", "在", "和", "也", "就", "都", "而", "及",
            "与", "着", "或", "一个", "没有", "我们", "你们", "他们",
            "这个", "那个", "什么", "怎么", "如何", "为什么", "因为",
            "所以", "但是", "如果", "虽然", "可以", "可能", "需要",
            "the", "a", "an", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may",
            "might", "shall", "can", "need", "dare", "ought",
            "this", "that", "these", "those", "it", "its", "不",
            "会", "能", "要", "对", "上", "下", "中", "来", "去",
        }

        def is_chinese(ch):
            """判断字符是否为中文"""
            try:
                return unicodedata.name(ch).startswith("CJK")
            except:
                return False

        def split_chinese(text):
            """把中文句子拆成单个字符（作为词单元）"""
            result = []
            for ch in text:
                if is_chinese(ch):
                    result.append(ch)
            return result

        text = text.lower()
        # 提取英文/数字词
        en_words = re.findall(r"[a-z0-9]+", text)
        # 提取中文字符
        cn_chars = split_chinese(text)

        all_words = en_words + cn_chars
        return [w for w in all_words if w not in stop_words and len(w) > 0]

    def build_context(self, query: str, top_k: int = 3) -> str:
        """搜索并组装成 RAG 上下文

        这是 RAG 的核心方法：
          1. 搜索最相关的知识段落
          2. 拼成"根据以下资料回答问题"的格式
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return ""

        parts = ["根据以下资料回答问题：\n"]
        for i, r in enumerate(results, 1):
            parts.append(f"[资料{i}]\n{r['content']}\n")

        return "\n".join(parts)

    def add_file(self, file_path: str, doc_id: str = None):
        """从文件添加知识（支持 .txt 和 .md）"""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        doc_id = doc_id or os.path.basename(file_path)
        return self.add_text(text, doc_id=doc_id, source=file_path)

    def get_tool_description(self) -> str:
        """给 LLM 看的工具描述（供 ReActAgent 使用）"""
        return """知识库工具 — 基于本地知识库回答问题
用法: RAG[query: 你的问题]
示例: RAG[query: Python装饰器是什么]

注意：知识库里的资料会作为回答的参考上下文"""
