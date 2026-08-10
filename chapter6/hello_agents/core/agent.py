"""
hello_agents/core/agent.py

Agent 抽象基类 — 所有 Agent 的"模板"。
定义了一个 Agent 应该有什么属性和方法，但不实现具体逻辑。

类比：建筑蓝图 — 规定了「要有门、要有窗」，但不规定门是什么颜色。
具体怎么建交给子类（SimpleAgent, ReActAgent, ...）
"""

from abc import ABC, abstractmethod
from typing import Optional, List

from .message import Message
from .llm import HelloAgentsLLM


class Agent(ABC):
    """Agent 基类（抽象类 — 不能直接实例化）"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self._history: List[Message] = []  # 对话历史

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行 Agent — 每个子类必须实现这个方法"""
        pass

    # ── 历史管理（所有 Agent 共享） ──

    def add_message(self, message: Message):
        self._history.append(message)

    def clear_history(self):
        self._history.clear()

    def get_history(self) -> List[Message]:
        return self._history.copy()

    def _build_messages(self, input_text: str) -> list[dict]:
        """把 system prompt + 历史 + 当前输入 拼成 OpenAI 格式

        这是所有 Agent 都需要的公共方法，写在基类里避免重复
        """
        messages = []

        # 1. system prompt
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # 2. 历史消息
        for msg in self._history:
            messages.append(msg.to_dict())

        # 3. 当前用户输入
        messages.append({"role": "user", "content": input_text})

        return messages

    def __str__(self) -> str:
        return f"Agent(name={self.name})"
