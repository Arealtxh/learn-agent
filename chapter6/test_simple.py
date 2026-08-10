"""
测试框架：SimpleAgent
"""
import sys
sys.path.insert(0, "/home/txh/learn-agent/chapter6")

from dotenv import load_dotenv
load_dotenv("/home/txh/learn-agent/chapter4/.env")

from hello_agents import SimpleAgent, HelloAgentsLLM

llm = HelloAgentsLLM()
agent = SimpleAgent(
    name="聊天助手",
    llm=llm,
    system_prompt="你是一个友好的助手，请用中文简短回答。",
)

for q in ["中国最长的河流是什么？", "它的长度是多少公里？"]:
    print(f"问: {q}")
    ans = agent.run(q)
    print(f"答: {ans[:150]}")
    print()

print(f"历史消息数: {len(agent.get_history())}")
