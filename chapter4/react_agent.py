"""
ReActAgent — 思考-行动-观察 循环的核心实现
"""
import re

REACT_PROMPT_TEMPLATE = """你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `ToolName[工具输入]`: 调用一个可用工具。
- `Finish[最终答案]`: 当你认为已经获得最终答案时。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""


class ReActAgent:
    def __init__(self, llm_client, tool_executor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        """运行 ReAct 智能体"""
        self.history = []
        print(f"\n{'='*50}")
        print(f"❓ 问题: {question}")
        print(f"{'='*50}\n")

        for step in range(1, self.max_steps + 1):
            print(f"\n--- 第 {step} 步 ---")

            # 1. 构造提示词
            tools_desc = self.tool_executor.get_available_tools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2. 调用 LLM 思考
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.think(messages)
            if not response:
                print("❌ LLM 返回为空，终止")
                return None

            # 3. 解析 Thought 和 Action
            thought, action = self._parse_output(response)

            if thought:
                print(f"🤔 思考: {thought}")

            if not action:
                print("⚠️ 未解析到 Action，终止")
                return None

            # 4. 如果是 Finish，结束
            if action.startswith("Finish"):
                final = re.match(r"Finish\[(.*)\]", action)
                if final:
                    answer = final.group(1)
                    print(f"\n🎉 最终答案: {answer}")
                    return answer
                break

            # 5. 解析并执行工具调用
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                observation = f"错误: 无法解析 Action '{action}'"
                print(f"⚠️ {observation}")
                self.history.append(f"Action: {action}")
                self.history.append(f"Observation: {observation}")
                continue

            print(f"🎬 行动: {tool_name}[\"{tool_input}\"]")

            tool_fn = self.tool_executor.get_tool(tool_name)
            if tool_fn:
                observation = tool_fn(tool_input)
            else:
                observation = f"错误: 未定义工具 '{tool_name}'"

            print(f"👀 观察: {observation[:200]}{'...' if len(observation) > 200 else ''}")

            # 6. 记录到历史，进入下一轮
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        print(f"\n⏰ 达到最大步数 ({self.max_steps})，终止")
        return None

    def _parse_output(self, text: str):
        """从 LLM 输出中提取 Thought 和 Action"""
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """从 Action 字符串中提取工具名和参数"""
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None
