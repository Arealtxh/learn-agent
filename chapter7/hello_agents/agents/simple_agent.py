"""
hello_agents/agents/simple_agent.py

SimpleAgent — 最简单的 Agent，只做对话，没有工具调用。

从第4章的视角看：这就是 llm_client.think() 的最小封装。
从框架的视角看：它是 Agent 基类的第一个具体实现。
"""

from typing import Optional

from ..core.agent import Agent
from ..core.message import Message
from ..core.llm import HelloAgentsLLM


class SimpleAgent(Agent):
    """简单对话 Agent — 没有 ReAct 循环，没有工具，只有 LLM 对话"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
    ):
        super().__init__(name, llm, system_prompt)

    def run(self, input_text: str, **kwargs) -> str:
        """
        执行一次对话

        流程：
            1. 拼 messages（system + 历史 + 当前输入）
            2. 调 LLM
            3. 保存到历史
            4. 返回回复
        """
        # 1. 构建消息
        messages = self._build_messages(input_text)

        # 2. 调 LLM
        response = self.llm.invoke(messages, **kwargs)

        # 3. 记录历史
        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=response, role="assistant"))

        return response
