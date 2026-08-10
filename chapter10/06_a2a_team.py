#!/usr/bin/env python3
"""第10章 A2A demo 2：多 Agent 协作流水线（研究员→撰写员→编辑，教程 10.3.3）"""
import threading
import time
import re

from hello_agents.protocols import A2AServer, A2AClient

# ============================================================
# 1. 创建三个 Agent 服务
# ============================================================
researcher = A2AServer(name="researcher", description="研究员")

@researcher.skill("research")
def do_research(text: str) -> str:
    """处理研究请求"""
    match = re.search(r'research\s+(.+)', text, re.IGNORECASE)
    topic = match.group(1).strip() if match else text
    return str({"topic": topic, "findings": f"{topic}的研究结果"})


writer = A2AServer(name="writer", description="撰写员")

@writer.skill("write")
def write_article(text: str) -> str:
    """撰写文章"""
    match = re.search(r'write\s+(.+)', text, re.IGNORECASE)
    content = match.group(1).strip() if match else text
    try:
        data = eval(content)
        topic = data.get("topic", "未知主题")
        findings = data.get("findings", "无研究结果")
    except Exception:
        topic = "未知主题"
        findings = content
    return f"# {topic}\n\n基于研究：{findings}\n\n文章内容..."


editor = A2AServer(name="editor", description="编辑")

@editor.skill("edit")
def edit_article(text: str) -> str:
    """编辑文章"""
    match = re.search(r'edit\s+(.+)', text, re.IGNORECASE)
    article = match.group(1).strip() if match else text
    result = {
        "article": article + "\n\n[已编辑优化]",
        "feedback": "文章质量良好",
        "approved": True
    }
    return str(result)


# ============================================================
# 2. 启动所有服务（各自独立端口）
# ============================================================
print("🚀 启动 3 个 Agent 服务...")
threading.Thread(target=lambda: researcher.run(port=5000), daemon=True).start()
threading.Thread(target=lambda: writer.run(port=5001), daemon=True).start()
threading.Thread(target=lambda: editor.run(port=5002), daemon=True).start()
time.sleep(3)  # 等待服务启动

# ============================================================
# 3. 客户端连接
# ============================================================
researcher_client = A2AClient("http://localhost:5000")
writer_client = A2AClient("http://localhost:5001")
editor_client = A2AClient("http://localhost:5002")

# ============================================================
# 4. 协作流程：研究 → 撰写 → 编辑
# ============================================================
def create_content(topic):
    print(f"\n📝 任务: 生产关于「{topic}」的内容")

    print("\n[步骤1] 研究员 Agent 研究...")
    research = researcher_client.execute_skill("research", f"research {topic}")
    research_data = research.get('result', '')
    print(f"  结果: {research_data}")

    print("\n[步骤2] 撰写员 Agent 写稿...")
    article = writer_client.execute_skill("write", f"write {research_data}")
    article_content = article.get('result', '')
    print(f"  结果: {article_content[:80]}...")

    print("\n[步骤3] 编辑 Agent 审稿...")
    final = editor_client.execute_skill("edit", f"edit {article_content}")
    print(f"  结果: {final.get('result', '')}")

    return final.get('result', '')


result = create_content("AI在医疗领域的应用")
print(f"\n✅ 最终产出：\n{result}")
