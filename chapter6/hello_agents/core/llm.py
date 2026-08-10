"""
hello_agents/core/llm.py

LLM 调用封装 — 从第4章的 llm_client.py 升级。
新增能力：
  - 支持任意 OpenAI 兼容 API（只需改 model + base_url）
  - 支持流式输出
  - 支持自动从 .env 加载配置
"""

import os
from typing import Optional, List, Dict, Any, Union
from openai import OpenAI


class HelloAgentsLLM:
    """统一的 LLM 调用接口

    用法：
        llm = HelloAgentsLLM()                    # 自动从 .env 读取
        llm = HelloAgentsLLM(model="gpt-4")        # 手动指定

    相比第4章的新增功能：
    - 构造函数自动加载 .env
    - 支持流式输出
    - 统一的 invoke 接口（不管模型是谁家的）
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
    ):
        # 优先用传参，其次从环境变量读
        self.model = model or os.getenv("LLM_MODEL_ID", "deepseek-v4-flash")
        api_key = api_key or os.getenv("LLM_API_KEY")
        base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.temperature = temperature

    def invoke(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> str:
        """调用 LLM，返回回复文本

        messages 格式：OpenAI API 标准格式
            [{"role": "system", "content": "..."},
             {"role": "user", "content": "..."}]
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            stream=stream,
        )

        if stream:
            collected = []
            for chunk in response:
                if chunk.choices[0].delta.content:
                    collected.append(chunk.choices[0].delta.content)
            return "".join(collected)
        else:
            return response.choices[0].message.content or ""

    def __str__(self) -> str:
        return f"HelloAgentsLLM(model={self.model})"
