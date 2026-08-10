"""
hello_agents/core/agent.py

Agent 抽象基类 — 所有 Agent 的"模板"。

这个文件是第7章最核心的一个文件，它定义了：
  1. 一个 Agent 应该有什么属性（name, llm, system_prompt, _history）
  2. 一个 Agent 应该有什么方法（run, add_message, _build_messages, ...）
  3. 哪些东西是所有 Agent 共享的（写在基类里）
  4. 哪些东西是每个 Agent 独有的（声明为抽象方法，让子类实现）

类比：建筑蓝图
  蓝图规定了"要有门、要有窗"，但不会告诉你门是什么颜色。
  你不能直接拿蓝图住人（不能 agent = Agent(...)）。
  你必须按照蓝图建房子（写一个子类继承它）。

对比第4章：
  第4章：每个脚本都手写拼 prompt、管理历史、调 LLM
  第7章：这些公共逻辑都写在基类里，子类只管"自己的特色逻辑"
"""

# ── 第1行：导入 ABC 工具 ──
# ABC = Abstract Base Class（抽象基类），Python 标准库提供的工具
# abstractmethod 是一个"装饰器"，用来标记"子类必须实现的方法"
from abc import ABC, abstractmethod

# Optional 表示"这个参数可以是 None"
# List 表示"这个参数是列表"
from typing import Optional, List

# ── 导入我们自己写的模块 ──
# 注意 import 语句里的点：
#   .message 表示"从当前目录的 message.py 导入"
#   .llm 表示"从当前目录的 llm.py 导入"
# 因为 agent.py 和 message.py、llm.py 都在 core/ 文件夹下
from .message import Message
from .llm import HelloAgentsLLM


class Agent(ABC):
    """
    Agent 基类（抽象类 — 不能直接实例化）

    这个类规定了"什么叫一个 Agent"：
      - 有一个名字（name）
      - 有一个 LLM（llm）
      - 有一段系统指令（system_prompt）
      - 有一段对话历史（_history）
      - 能运行（run）
      - 能管理历史（add_message、clear_history、get_history）
      - 能拼 prompt（_build_messages）

    使用方式：
      你不能直接写 agent = Agent(...)  # ❌ 会报错
      你必须写 agent = SimpleAgent(...) # ✅ 继承自 Agent

    在第4章，这些功能分散在 main.py、react_agent.py 等文件里。
    在第7章，它们集中在这里，所有子类共享。
    """

    # ════════════════════════════════════════
    # 构造函数 __init__
    # ════════════════════════════════════════
    # Python 的构造函数：当你写 agent = Agent(name="助手", llm=xxx) 时，
    # Python 会自动调用 __init__ 方法，把参数传进来
    #
    # 参数中的类型注解（name: str, llm: HelloAgentsLLM）：
    #   - name: str         意思是"name 应该是字符串"
    #   - llm: HelloAgentsLLM 意思是"llm 应该是 HelloAgentsLLM 类型"
    #   - Optional[str]     意思是"可以是字符串，也可以是 None"
    # 这些注解不会影响代码运行，但 IDE 和类型检查工具会帮你发现错误
    #
    # 注意 self：
    #   在 Python 的类方法中，self 总是第一个参数
    #   self = "这个对象自己"
    #   self.name = name  = "把这个对象的 name 属性设为参数 name 的值"
    def __init__(
        self,
        name: str,                           # Agent 的名字（比如"聊天助手"、"搜索助手"）
        llm: HelloAgentsLLM,                 # 用哪个 LLM？（从外面传进来，不是自己创建的）
        system_prompt: Optional[str] = None,  # 系统指令（可选，不传就是 None）
    ):
        """
        初始化 Agent

        参数：
          name:           Agent 的名字。以后有多个 Agent 时方便区分
                          比如"聊天助手"用 SimpleAgent，"搜索助手"用 ReActAgent
          llm:            用什么 LLM。注意这个 LLM 是"从外面传进来的"，
                          不是 Agent 自己创建的。这叫"依赖注入"——
                          LLM 怎么配置是外面的事，Agent 只管用它
          system_prompt:  系统指令。给 LLM 设定角色和行为规则
                          比如"你是一个友好的助手"或"你是一个 Python 专家"
                          不传的话就没有系统指令（简单场景）
        """
        # ── self.name = name ──
        # 把参数 name 的值保存到对象的 name 属性上
        # 以后在其他方法里可以用 self.name 拿到这个名字
        self.name = name

        # ── self.llm = llm ──
        # 保存 LLM 实例。注意不是 Agent 自己创建 LLM，
        # 而是外面创建好传进来（这叫"依赖注入"）
        #
        # 为什么这么设计？
        #   如果 Agent 自己创建 LLM，那换 LLM 就要改 Agent 的代码
        #   如果从外面传，换 LLM 只需要在创建 Agent 时换参数
        #   agent = SimpleAgent(name="助手", llm=llm_a)  # 用一个 LLM
        #   agent = SimpleAgent(name="助手", llm=llm_b)  # 换一个 LLM，不改 Agent 代码
        self.llm = llm

        # ── self.system_prompt = system_prompt ──
        # 保存系统指令。如果不传，system_prompt 就是 None
        # 在 _build_messages() 中会判断：如果是 None，就不加 system prompt
        self.system_prompt = system_prompt

        # ── self._history ──
        # 对话历史列表。变量名前加 _ 是 Python 约定，表示
        # "这个变量是内部使用的，别在外面直接改它"
        #
        # 外部代码只能通过 add_message()、get_history()、clear_history()
        # 这三个公开方法来操作_history
        #
        # 类型注解 List[Message] 表示"这是一个列表，里面每个元素都是 Message 对象"
        self._history: List[Message] = []

    # ════════════════════════════════════════
    # 抽象方法 run
    # ════════════════════════════════════════
    # @abstractmethod 是"抽象方法"装饰器
    # 意思是：这个方法在基类里只声明，不实现
    # 子类必须自己实现这个方法，否则子类也无法实例化
    #
    # 为什么基类不实现 run()？
    #   因为每个 Agent "运行"的方式不一样：
    #     SimpleAgent 的 run：调一次 LLM 就返回
    #     ReActAgent 的 run：循环 Thought→Action→Observation 多轮
    #     以后可能还有别的 Agent，各有各的 run 方式
    #
    # 但所有 Agent 都必须有一个 run 方法——这是基类规定的
    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """
        运行 Agent — 每个子类必须实现这个方法

        参数：
          input_text: 用户的输入文本（比如"中国最长的河流是什么？"）
          **kwargs:   额外的关键字参数（比如 stream=True）
                      这个语法叫"关键字参数收集"——调用时传了哪些额外的参数，
                      都会被收集到 kwargs 这个字典里

        返回值：
          str: Agent 的回复文本

        子类实现示例（SimpleAgent）：
          def run(self, input_text, **kwargs):
              messages = self._build_messages(input_text)
              response = self.llm.invoke(messages, **kwargs)
              self.add_message(Message(input_text, "user"))
              self.add_message(Message(response, "assistant"))
              return response
        """
        # pass 是 Python 的"空语句"
        # 表示"这个函数暂时什么都不做"
        # 因为抽象方法本身就不应该做任何事——它只是一个"规定"
        pass

    # ════════════════════════════════════════
    # 公开方法：历史管理
    # ════════════════════════════════════════
    # 这三个方法是"所有 Agent 共享"的：
    #   SimpleAgent 用它们记录对话
    #   ReActAgent 也用它们记录对话
    #   以后的任何 Agent 也用它们记录对话
    # 所以放在基类里，写一次，所有子类都有

    def add_message(self, message: Message):
        """
        追加一条消息到对话历史

        参数：
          message: Message 对象（自己写的消息类）

        示例：
          agent.add_message(Message("你好", "user"))
          agent.add_message(Message("你好！", "assistant"))

        对比第4章：
          第4章：history.append({"role": "user", "content": "你好"})
          第7章：agent.add_message(Message("你好", "user"))
          区别：第7章有类型检查（写错 role 会报错），而且 Message 自带时间戳
        """
        # 把 Message 对象追加到 _history 列表的末尾
        self._history.append(message)

    def clear_history(self):
        """
        清空对话历史

        什么时候需要清空？
          - 开始一个新的话题
          - Agent 的记忆太多，上下文窗口不够了
          - 想让 Agent"忘记"之前的对话

        示例：
          agent.clear_history()  # 所有历史记录被清空
          print(agent.get_history())  # []
        """
        # list.clear() 是 Python 列表的内置方法
        # 会清空列表中的所有元素，但保留列表对象本身
        self._history.clear()

    def get_history(self) -> List[Message]:
        """
        获取对话历史的副本

        返回值：
          一个 Message 列表，是 _history 的副本（copy）
          注意是副本不是原列表，防止外部代码误操作修改内部数据

        为什么要返回副本（copy）而不是原列表？
          如果返回原列表，外部代码可以这样：
            history = agent.get_history()
            history.clear()          # 外部把 Agent 的历史清空了！
          如果是副本：
            history = agent.get_history()
            history.clear()          # 清空的是副本，原列表不受影响

        这种"保护性复制"是面向对象编程的常见模式。
        """
        # list.copy() 返回列表的浅拷贝
        # "浅拷贝"的意思是：新列表中的 Message 对象和原列表是同一个对象
        # 但列表本身是新的，所以增删元素不会影响原列表
        return self._history.copy()

    # ════════════════════════════════════════
    # 内部方法：拼 prompt
    # ════════════════════════════════════════
    # 方法名前加 _ 表示"这是内部方法，别在外面调用"
    # 这个方法做的事：
    #   把 system prompt + 历史消息 + 当前输入
    #   拼成 OpenAI API 认识的字典列表格式

    def _build_messages(self, input_text: str) -> list[dict]:
        """
        把 system prompt + 历史 + 当前输入 拼成 OpenAI 格式

        这是所有 Agent 都需要的公共方法，写在基类里避免重复。

        返回格式（OpenAI API 格式）：
          [
              {"role": "system", "content": "你是一个友好的助手"},
              {"role": "user", "content": "中国最长的河流是什么？"},
              {"role": "assistant", "content": "长江是中国最长的河流"},
              {"role": "user", "content": "它的长度是多少公里？"},
          ]

        参数：
          input_text: 用户当前的输入文本

        返回值：
          一个字典列表，每个字典有 role 和 content 两个键

        对比第4章：
          第4章：每个脚本都手写这段拼接逻辑
          第7章：写在基类里，一个方法搞定
        """
        # ── 创建空列表，准备放消息 ──
        # 这个列表最后会传给 LLM 的 API
        messages = []

        # ── 第1步：加 system prompt ──
        # 如果设置了 system_prompt，把它放在消息列表的最前面
        # system prompt 在 OpenAI API 中是一个特殊的消息角色
        # 它告诉 LLM "你是什么角色"、"应该怎么回答问题"
        #
        # 注意：if self.system_prompt 检查的是 system_prompt 不是 None
        # 也不是空字符串。如果 system_prompt 是 None 或 ""，就不加
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # ── 第2步：加历史消息 ──
        # 遍历 _history 列表中的每一个 Message 对象
        # 用 to_dict() 方法把 Message 对象转成字典
        # （因为 OpenAI API 只认识字典，不认识 Message 对象）
        for msg in self._history:
            messages.append(msg.to_dict())

        # ── 第3步：加当前用户输入 ──
        # 用户的当前输入放在最后
        # 这样 LLM 看到的就是：
        #   1. 你是谁（system prompt）
        #   2. 之前说过什么（历史）
        #   3. 现在要问什么（当前输入）
        messages.append({"role": "user", "content": input_text})

        # ── 返回拼好的消息列表 ──
        # 这个列表可以直接传给 llm.invoke(messages)
        return messages

    # ════════════════════════════════════════
    # 魔法方法 __str__
    # ════════════════════════════════════════
    # __str__ 是 Python 的"字符串表示"方法
    # 当你 print(agent) 时，Python 会自动调用这个方法
    # 返回的字符串就是 print 显示的内容

    def __str__(self) -> str:
        """
        返回 Agent 的字符串表示

        当你写 print(agent) 时，会看到类似这样的输出：
          Agent(name=聊天助手)

        作用：方便调试和日志记录
        """
        return f"Agent(name={self.name})"
