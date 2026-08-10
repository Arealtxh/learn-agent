"""
第9章: NoteTool — 结构化笔记工具 (精简版)

第8章的 MemoryTool 处理"对话记忆"
第9章的 NoteTool 处理"项目笔记"——更结构化、更人类友好

格式: Markdown + YAML 前置元数据
  ---
  title: 重构进展
  type: task_state
  tags: [refactoring, phase1]
  ---
  # 正文内容...

和 MemoryTool 的区别:
  第8章 MemoryTool → 面向 LLM 阅读的向量/关键词记忆
  第9章 NoteTool   → 面向人类阅读的 Markdown 笔记
"""

import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any


class NoteTool:
    """
    结构化笔记工具

    用法:
        notes = NoteTool(workspace="./project_notes")
        notes.create("本周计划", "## 目标\n完成数据迁移", note_type="task_state", tags=["week1"])
        notes.search("数据迁移")
        notes.summary()
    """

    VALID_TYPES = {"task_state", "conclusion", "blocker", "action", "reference", "general"}

    def __init__(self, workspace: str = "./notes"):
        self.workspace = workspace
        self.index_path = os.path.join(workspace, "notes_index.json")
        self.index: Dict[str, Dict] = {}
        self._ensure_workspace()

    def _ensure_workspace(self):
        """确保工作目录和索引文件存在"""
        os.makedirs(self.workspace, exist_ok=True)
        if os.path.exists(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as f:
                self.index = json.load(f)
        else:
            self.index = {}
            self._save_index()

    def _save_index(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    def _next_id(self) -> str:
        """生成唯一笔记 ID"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"note_{ts}_{len(self.index)}"

    # ----- 核心操作 -----

    def create(
        self,
        title: str,
        content: str,
        note_type: str = "general",
        tags: Optional[List[str]] = None,
    ) -> str:
        """创建笔记"""
        if note_type not in self.VALID_TYPES:
            raise ValueError(f"无效笔记类型: {note_type}，可选: {self.VALID_TYPES}")

        note_id = self._next_id()
        now = datetime.now().isoformat()

        meta = {
            "id": note_id,
            "title": title,
            "type": note_type,
            "tags": tags or [],
            "created_at": now,
            "updated_at": now,
        }

        # 构建 Markdown 内容
        md = f"---\ntitle: {title}\ntype: {note_type}\ntags: {json.dumps(tags or [], ensure_ascii=False)}\ncreated_at: {now}\nupdated_at: {now}\n---\n\n{content}"

        file_path = os.path.join(self.workspace, f"{note_id}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md)

        meta["file_path"] = file_path
        self.index[note_id] = meta
        self._save_index()

        print(f"  ✅ 笔记已创建: [{note_type}] {title}  (ID: {note_id})")
        return note_id

    def read(self, note_id: str) -> Optional[Dict[str, Any]]:
        """读取笔记"""
        if note_id not in self.index:
            print(f"  ❌ 笔记不存在: {note_id}")
            return None

        meta = self.index[note_id]
        file_path = meta.get("file_path", os.path.join(self.workspace, f"{note_id}.md"))

        if not os.path.exists(file_path):
            print(f"  ❌ 笔记文件丢失: {file_path}")
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()

        # 分离 YAML 和正文
        parts = raw.split("---\n", 2)
        content = parts[2].strip() if len(parts) >= 3 else raw.strip()

        return {"metadata": meta, "content": content}

    def update(
        self,
        note_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        note_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """更新笔记"""
        if note_id not in self.index:
            print(f"  ❌ 笔记不存在: {note_id}")
            return

        meta = self.index[note_id]
        existing = self.read(note_id)
        if not existing:
            return

        old_content = existing["content"]
        old_meta = existing["metadata"]

        if title:
            meta["title"] = title
        if note_type:
            if note_type not in self.VALID_TYPES:
                raise ValueError(f"无效笔记类型: {note_type}")
            meta["type"] = note_type
        if tags is not None:
            meta["tags"] = tags

        meta["updated_at"] = datetime.now().isoformat()
        new_content = content if content is not None else old_content

        # 重写文件
        md = f"---\ntitle: {meta['title']}\ntype: {meta['type']}\ntags: {json.dumps(meta.get('tags', []), ensure_ascii=False)}\ncreated_at: {meta['created_at']}\nupdated_at: {meta['updated_at']}\n---\n\n{new_content}"

        file_path = meta["file_path"]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md)

        self.index[note_id] = meta
        self._save_index()
        print(f"  ✅ 笔记已更新: {meta['title']}")

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索笔记（在标题和内容中匹配关键词）"""
        q = query.lower()
        results = []

        for note_id, meta in self.index.items():
            note = self.read(note_id)
            if not note:
                continue

            content = note["content"]
            title = meta.get("title", "")

            if q in title.lower() or q in content.lower():
                results.append({
                    "note_id": note_id,
                    "title": title,
                    "type": meta.get("type"),
                    "tags": meta.get("tags", []),
                    "content": content[:200] + ("..." if len(content) > 200 else ""),
                    "updated_at": meta.get("updated_at"),
                })

        results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return results[:limit]

    def list(
        self,
        note_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """列出笔记"""
        results = []
        for note_id, meta in self.index.items():
            if note_type and meta.get("type") != note_type:
                continue
            if tags:
                note_tags = set(meta.get("tags", []))
                if not note_tags.intersection(tags):
                    continue
            results.append(meta)

        results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return results[:limit]

    def delete(self, note_id: str):
        """删除笔记"""
        if note_id not in self.index:
            print(f"  ❌ 笔记不存在: {note_id}")
            return

        meta = self.index[note_id]
        file_path = meta.get("file_path", os.path.join(self.workspace, f"{note_id}.md"))
        if os.path.exists(file_path):
            os.remove(file_path)

        title = meta.get("title", note_id)
        del self.index[note_id]
        self._save_index()
        print(f"  ✅ 笔记已删除: {title}")

    def summary(self) -> Dict[str, Any]:
        """笔记统计摘要"""
        total = len(self.index)
        type_counts = {}
        for meta in self.index.values():
            t = meta.get("type", "general")
            type_counts[t] = type_counts.get(t, 0) + 1

        recent = sorted(self.index.values(), key=lambda x: x.get("updated_at", ""), reverse=True)[:5]

        return {
            "total_notes": total,
            "type_distribution": type_counts,
            "recent_notes": [{"id": n["id"], "title": n.get("title"), "type": n.get("type")} for n in recent],
        }


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    import tempfile
    import shutil

    print("=" * 55)
    print("  NoteTool 演示 — 结构化笔记管理")
    print("=" * 55)

    # 创建临时目录
    tmpdir = tempfile.mkdtemp(prefix="notetool_demo_")
    notes = NoteTool(workspace=tmpdir)

    # 创建笔记
    print("\n📝 创建笔记...")
    notes.create(
        "重构项目 - 第一阶段",
        "## 完成情况\n已完成数据模型层重构，测试覆盖率85%。\n\n## 下一步\n重构业务逻辑层",
        note_type="task_state",
        tags=["refactoring", "phase1"],
    )

    notes.create(
        "依赖版本冲突问题",
        "## 问题描述\n发现某些第三方库版本不兼容。\n\n影响范围: 业务逻辑层3个模块",
        note_type="blocker",
        tags=["dependency", "urgent"],
    )

    notes.create(
        "本周计划",
        "## 目标\n1. 解决依赖冲突\n2. 开始业务逻辑重构\n3. 提升测试覆盖",
        note_type="action",
        tags=["week1"],
    )

    # 搜索
    print("\n🔍 搜索 '依赖'...")
    results = notes.search("依赖")
    for r in results:
        print(f"  [{r['type']}] {r['title']}")

    # 摘要
    print("\n📊 笔记摘要:")
    s = notes.summary()
    print(f"  总数: {s['total_notes']}")
    print(f"  类型分布: {s['type_distribution']}")
    print(f"  最近笔记:")
    for n in s['recent_notes']:
        print(f"    · [{n['type']}] {n['title']}")

    # 清理
    shutil.rmtree(tmpdir)
