#!/usr/bin/env python3
"""第10章 ANP demo：服务发现 + 负载均衡 + 任务调度（教程 10.4.2 / 10.4.3）"""
import os
import random
from dotenv import load_dotenv

load_dotenv("/home/txh/learn-agent/chapter4/.env")

from hello_agents.protocols import ANPDiscovery, register_service, discover_service

# ============================================================
# Part 1: 服务注册（10 个计算节点）
# ============================================================
print("=" * 60)
print("Part 1: 注册 10 个计算节点")
print("=" * 60)
discovery = ANPDiscovery()

for i in range(10):
    register_service(
        discovery=discovery,
        service_id=f"compute_node_{i}",
        service_name=f"计算节点{i}",
        service_type="compute",
        capabilities=["data_processing", "ml_training"],
        endpoint=f"http://node{i}:8000",
        metadata={
            "load": random.uniform(0.1, 0.9),
            "cpu_cores": random.choice([4, 8, 16]),
            "memory_gb": random.choice([16, 32, 64]),
            "gpu": random.choice([True, False])
        }
    )

print(f"✅ 注册了 {len(discovery.list_all_services())} 个计算节点")

# ============================================================
# Part 2: 服务发现 + 负载均衡（纯代码版）
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 按类型发现服务，选择负载最低的节点")
print("=" * 60)
compute_services = discover_service(discovery, service_type="compute")
best = min(compute_services, key=lambda s: s.metadata.get("load", 1.0))
print(f"发现 {len(compute_services)} 个 compute 服务")
print(f"负载最低: {best.service_name} (负载: {best.metadata['load']:.2f}, "
      f"CPU: {best.metadata['cpu_cores']}核, GPU: {best.metadata['gpu']})")

# 模拟 5 个请求的负载均衡（每次选最闲的，然后给它加负载）
print("\n模拟 5 个请求依次分配（每次挑负载最低）:")
servers = discovery.discover_services(service_type="compute")
for i in range(5):
    target = min(servers, key=lambda s: s.metadata.get("load", 1.0))
    print(f"  请求 {i+1} -> {target.service_name} (负载: {target.metadata['load']:.2f})")
    target.metadata["load"] += 0.1

# ============================================================
# Part 3: LLM 任务调度（SimpleAgent + ANPTool）
# ============================================================
print("\n" + "=" * 60)
print("Part 3: LLM 任务调度（SimpleAgent + ANPTool）")
print("=" * 60)
try:
    from hello_agents import SimpleAgent, HelloAgentsLLM
    from hello_agents.tools.builtin import ANPTool

    llm = HelloAgentsLLM()
    scheduler = SimpleAgent(
        name="任务调度器",
        llm=llm,
        system_prompt="""你是一个智能任务调度器，负责：
1. 分析任务需求
2. 选择最合适的计算节点
3. 分配任务

你有工具 service_discovery，可以查询计算节点。调用方法：
[TOOL_CALL:service_discovery:action=discover_services]
返回结果是所有节点的列表（含 service_name、负载 load、cpu_cores、memory_gb、gpu 等信息）。

选择节点时考虑：负载、CPU核心数、内存、GPU等因素。"""
    )

    anp_tool = ANPTool(
        name="service_discovery",
        description="服务发现工具，可以查找和选择计算节点",
        discovery=discovery
    )
    scheduler.add_tool(anp_tool)

    print("\n任务: 训练一个大型深度学习模型，需要GPU支持")
    response = scheduler.run("""
    请为以下任务选择最合适的计算节点：
    训练一个大型深度学习模型，需要GPU支持

    要求：
    1. 列出所有可用节点
    2. 分析每个节点的特点
    3. 选择最合适的节点
    4. 说明选择理由
    """)
    print(f"\n🤖 调度结果:\n{response}")
except Exception as e:
    print(f"⚠️ LLM 调度部分跳过: {e}")
