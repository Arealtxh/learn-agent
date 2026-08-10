"""
hello_agents/core/message.py

消息系统 — 把第4章手写的 message dict 变成标准化的 Message 类。

核心作用：
  让 Agent 和 LLM 之间的数据格式统一、可扩展。 
  
对比第4章：
  第4章：用裸字典 {"role": "user", "content": "你好"}
  第7章：用 Message 对象，自带时间戳和元数据，还能扩展额外字段

为什么用类不用字典？
  1. 字典没法加额外信息（比如想加个时间戳，要改所有代码）
  2. 字典没有类型检查（写错 role="useer" 也不报错）
  3. 类可以封装方法（比如 to_dict() 转成 API 格式）
"""

from datetime import datetime
from typing import Optional, Dict, Any, Literal


# ── 角色类型 ──
# Literal 意思是：role 只能是这四个值之一
# 如果你写 role="useer"（拼写错误），IDE 和类型检查工具会报红
# 第4章的字典没有这个保护
MessageRole = Literal["user", "assistant", "system", "tool"]


class Message:
    """统一消息类

    每个 Message 对象代表一条对话消息。
    内部用对象管理，发请求时用 to_dict() 转成 OpenAI API 认识的字典。

    用法：
        msg = Message("你好", "user")
        print(msg.content)       # "你好"
        print(msg.role)          # "user"
        print(msg.timestamp)     # 自动记录创建时间
        print(msg.to_dict())     # {"role": "user", "content": "你好"}
    """

    def __init__(
        self,
        content: str,                     # 消息正文（比如用户说的话）
        role: MessageRole,                # 角色：user/assistant/system/tool
        metadata: Optional[Dict[str, Any]] = None,  # 附加信息（可选）
    ):
        # ── 核心字段 ──
        self.content = content            # 消息内容（字符串）
        self.role = role                  # 谁说的（user=用户, assistant=AI, system=系统指令, tool=工具返回）

        # ── 自动记录的字段 ──
        self.timestamp = datetime.now()   # ⏰ 自动记录消息创建时间，不需要手动传

        # ── 可扩展字段 ──
        # metadata 是一个字典，以后想加什么额外信息都放这里
        # 比如：{"source": "web_search", "confidence": 0.9}
        # 注意：如果没传 metadata，默认给空字典 {}，避免后面写代码时要判断 None
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, str]:
        """转成 OpenAI API 兼容的字典格式

        为什么要有这个方法？
          Agent 内部用 Message 对象管理消息（方便扩展）
          OpenAI API 只认字典格式（API 要求）
          to_dict() 就是"对象 → API格式"的桥梁

        返回示例：
          {"role": "user", "content": "你好"}
        """
        return {"role": self.role, "content": self.content}

    def __str__(self) -> str:
        """给 print() 看的时候显示什么"""
        return f"[{self.role}] {self.content[:50]}..."

    def __repr__(self) -> str:
        """给调试器看的时候显示什么"""
        return f"Message(role={self.role!r}, content={self.content[:30]!r})"
