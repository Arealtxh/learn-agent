"""
hello_agents/agents/react_agent.py

ReActAgent — 从第4章 react_agent.py 升级到框架版本。

核心变化：
  第4章：独立脚本，硬编码 prompt，手动拼 history 字符串
  这章：继承 Agent 基类，用 Message 管理历史，支持自定义 prompt

ReAct 循环不变：Thought → Action → Observation → Thought → ...
"""

import re
from typing import Optional, Dict, Callable

from ..core.agent import Agent
from ..core.message import Message
from ..core.llm import HelloAgentsLLM


class ToolRegistry:
    """工具注册中心 — 第4章的 tool_executor.py 的框架化版本

    功能：
      - 注册工具（名字 → 描述 + 函数）
      - 根据名字获取工具
      - 生成给 LLM 看的工具描述文本
    """
    def __init__(self):
        self._tools: Dict[str, Dict] = {}

    def register(self, name: str, description: str, fn: Callable):
        self._tools[name] = {"description": description, "fn": fn}

    def get(self, name: str) -> Optional[Callable]:
        tool = self._tools.get(name)
        return tool["fn"] if tool else None

    def get_tool_descriptions(self) -> str:
        """生成给 LLM 看的工具列表文本"""
        lines = ["可用工具："]
        for name, info in self._tools.items():
            lines.append(f"  - {name}: {info['description']}")
        lines.append("使用格式: 工具名[参数]   例如: Search[今日新闻]")
        lines.append('想结束时使用: Finish[答案]')
        return "\n".join(lines)


# 默认 ReAct Prompt 模板 — 比第4章更精致
REACT_SYSTEM_PROMPT = """你是一个智能助手，通过"思考→行动→观察"的循环来解决问题。

工作流程：
1. Thought: 分析当前情况，决定下一步做什么
2. Action: 执行一个具体行动（调用工具或给出最终答案）
3. 观察工具返回的结果，然后进入下一轮思考

{tool_descriptions}

每次输出格式：
Thought: 你的分析...
Action: 工具名[参数]
  或
Action: Finish[最终答案]"""


class ReActAgent(Agent):
    """
    ReAct 范式 Agent — 边想边做

    和第4章 react_agent.py 的区别：
    - 继承了 Agent 基类，复用 _build_messages() 和 history 管理
    - 用 ToolRegistry 代替 tool_executor.py
    - 支持自定义 system_prompt
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_steps: int = 5,
    ):
        super().__init__(name, llm, system_prompt)
        self.tools = tool_registry or ToolRegistry()
        self.max_steps = max_steps

    def run(self, input_text: str, **kwargs) -> str:
        """执行 ReAct 循环"""
        # 构建完整的 system prompt（含工具描述）
        tool_desc = self.tools.get_tool_descriptions()
        system_prompt = self.system_prompt or REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_desc
        )

        history = ""  # 存储本轮循环的 Thought/Action/Observation

        for step in range(1, self.max_steps + 1):
            print(f"\n── 第{step}轮 ReAct ──")

            # 拼 prompt（参考第4章 main.py 的拼法）
            prompt = f"""问题: {input_text}

{system_prompt}

历史执行记录:
{history}

请根据以上信息，输出你的思考和行动："""

            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages, **kwargs)
            print(f"  LLM: {response[:200]}")

            # 解析 Thought
            thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", response, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else ""

            # 解析 Action
            action_match = re.search(r"Action:\s*(.*?)$", response, re.MULTILINE)
            if not action_match:
                history += f"Thought: {thought}\nAction: 解析失败\nObservation: 无法解析输出\n"
                continue

            action_text = action_match.group(1).strip()

            # 检查是不是 Finish
            finish_match = re.match(r"Finish\[(.*)\]", action_text)
            if finish_match:
                answer = finish_match.group(1)
                print(f"  ✅ Agent 决定结束")
                # 保存到历史
                self.add_message(Message(content=input_text, role="user"))
                self.add_message(Message(content=answer, role="assistant"))
                return answer

            # 解析工具调用
            tool_match = re.match(r"(\w+)\[(.*)\]", action_text)
            if not tool_match:
                history += f"Thought: {thought}\nAction: {action_text}\nObservation: 格式错误，无法解析工具\n"
                continue

            tool_name, tool_input = tool_match.group(1), tool_match.group(2)
            tool_fn = self.tools.get(tool_name)

            if not tool_fn:
                obs = f"错误: 未知工具 '{tool_name}'"
                print(f"  ❌ {obs}")
            else:
                try:
                    obs = tool_fn(tool_input)
                    print(f"  🔧 调用 {tool_name} → 成功")
                except Exception as e:
                    obs = f"工具执行错误: {e}"
                    print(f"  ❌ 工具出错: {e}")

            history += f"Thought: {thought}\nAction: {action_text}\nObservation: {obs}\n"

        # 超出最大步数
        fallback = f"（经过{self.max_steps}轮仍未得到最终答案）"
        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=fallback, role="assistant"))
        return fallback
