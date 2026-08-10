"""
hello_agents/core/llm.py

LLM 调用封装 — 从第4章的 llm_client.py 升级。

这个类的作用：
  把"调 LLM"这个操作封装起来，让调用方只需要写一行：
    llm.invoke(messages)

对比第4章：
  第4章每次调 LLM 都要写：
    client = OpenAI(api_key=..., base_url=...)
    response = client.chat.completions.create(model=..., messages=...)
    answer = response.choices[0].message.content

  第7章只需要：
    llm = HelloAgentsLLM()          # 自动读 .env
    answer = llm.invoke(messages)    # 调 LLM

新增能力：
  - 自动从 .env 加载配置（API key、模型名、地址）
  - 支持流式输出（stream=True）
  - 换模型只需改参数，不用改代码
"""

import os
from typing import Optional, List, Dict, Any, Union
from openai import OpenAI


class HelloAgentsLLM:
    """统一的 LLM 调用接口

    用法：
        # 最简用法（自动从 .env 读取配置）
        llm = HelloAgentsLLM()
        answer = llm.invoke([{"role": "user", "content": "你好"}])

        # 手动指定模型
        llm = HelloAgentsLLM(model="gpt-4")

        # 流式输出
        answer = llm.invoke(messages, stream=True)

    核心思路：
      把所有 LLM API 的通用参数（model, api_key, base_url, temperature）
      放在构造函数里一次性设置，后面调用 invoke() 就不用重复传了。
    """

    def __init__(
        self,
        model: Optional[str] = None,          # 模型名，不传就从 .env 读
        api_key: Optional[str] = None,         # API key，不传就从 .env 读
        base_url: Optional[str] = None,        # API 地址，不传就从 .env 读
        temperature: float = 0.3,              # 温度参数（默认 0.3，值越大回答越随机）
    ):
        # ── 参数优先级：传参 > 环境变量 > 默认值 ──
        # 这样设计的好处：
        #   1. 用户可以不传任何参数（全从 .env 自动读）
        #   2. 用户想临时换模型时，可以直接传参覆盖
        #   3. 默认值保证即使 .env 没配也不会报错
        self.model = model or os.getenv("LLM_MODEL_ID", "deepseek-v4-flash")
        api_key = api_key or os.getenv("LLM_API_KEY")
        base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")

        # ── 创建 OpenAI 客户端 ──
        # OpenAI() 是官方的 SDK，支持所有 OpenAI 兼容的 API
        # 包括 DeepSeek、OpenRouter、Claude 等
        # 只要改 base_url 和 api_key，就能切换不同的模型提供商
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.temperature = temperature

    def invoke(
        self,
        messages: List[Dict[str, str]],       # 消息列表，OpenAI API 格式
        temperature: Optional[float] = None,   # 可选：临时覆盖温度参数
        stream: bool = False,                  # 可选：是否流式输出
    ) -> str:
        """调用 LLM，返回回复文本

        参数 messages 的格式（和第4章完全一样）：
            [
                {"role": "system", "content": "你是一个助手"},
                {"role": "user", "content": "你好"}
            ]

        返回值：
            纯文本字符串（LLM 的回答）

        流式输出（stream=True）：
            普通模式：等 LLM 全部生成完再一次性返回
            流式模式：LLM 一边生成一边返回 chunk，invoke 帮你拼成完整字符串
            调用方不需要知道流式的存在，invoke 返回的永远是完整字符串
        """
        # ── 调用 OpenAI API ──
        # 这是实际发 HTTP 请求到 LLM 的地方
        # model, temperature 来自构造函数
        # messages 来自参数
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            stream=stream,
        )

        # ── 处理返回结果 ──
        if stream:
            # 流式模式：API 返回的是一个个 chunk
            # 每个 chunk 里只有一小段文本
            # 我们需要把它们拼起来
            collected = []
            for chunk in response:
                if chunk.choices[0].delta.content:
                    collected.append(chunk.choices[0].delta.content)
            return "".join(collected)
        else:
            # 非流式模式：API 一次性返回完整结果
            # response.choices[0].message.content 就是 LLM 的回答文本
            # or "" 是为了防止返回 None（API 有时会返回 None）
            return response.choices[0].message.content or ""

    def __str__(self) -> str:
        """print(llm) 时显示的内容"""
        return f"HelloAgentsLLM(model={self.model})"
