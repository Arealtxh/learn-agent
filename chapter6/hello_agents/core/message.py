"""
hello_agents/core/message.py

消息系统 — 把第4章手写的 message dict 变成标准化的 Message 类。
核心作用：让 Agent 和 LLM 之间的数据格式统一、可扩展。
"""

from datetime import datetime
from typing import Optional, Dict, Any, Literal


# 角色类型：严格限制为 OpenAI API 支持的四种
MessageRole = Literal["user", "assistant", "system", "tool"]


class Message:
    """统一消息类

    第4章你手写的是 {role, content} 字典
    这章升级为 Message 类，增加时间戳和元数据，方便后续扩展
    """

    def __init__(
        self,
        content: str,
        role: MessageRole,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.content = content
        self.role = role
        self.timestamp = datetime.now()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, str]:
        """转成 OpenAI API 兼容的字典格式"""
        return {"role": self.role, "content": self.content}

    def __str__(self) -> str:
        return f"[{self.role}] {self.content[:50]}..."

    def __repr__(self) -> str:
        return f"Message(role={self.role!r}, content={self.content[:30]!r})"
