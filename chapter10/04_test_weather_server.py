#!/usr/bin/env python3
"""测试天气查询 MCP 服务器（教程 10.5.1 测试部分）"""
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hello_agents.protocols.mcp.client import MCPClient


async def test_weather_server():
    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_weather_mcp_server.py")
    client = MCPClient(["python", server_script])

    try:
        async with client:
            # 测试1: 获取服务器信息
            info = json.loads(await client.call_tool("get_server_info", {}))
            print(f"服务器: {info['name']} v{info['version']}")

            # 测试2: 列出支持的城市
            cities = json.loads(await client.call_tool("list_supported_cities", {}))
            print(f"支持城市: {cities['count']} 个")

            # 测试3: 查询北京天气
            weather = json.loads(await client.call_tool("get_weather", {"city": "北京"}))
            if "error" not in weather:
                print(f"\n北京天气: {weather['temperature']}°C, {weather['condition']}, "
                      f"湿度{weather['humidity']}%, 风速{weather['wind_speed']}m/s")
            else:
                print(f"北京查询失败: {weather.get('error')}")

            # 测试4: 查询杭州天气
            weather = json.loads(await client.call_tool("get_weather", {"city": "杭州"}))
            if "error" not in weather:
                print(f"杭州天气: {weather['temperature']}°C, {weather['condition']}")
            else:
                print(f"杭州查询失败: {weather.get('error')}")

            print("\n✅ 所有测试完成！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    asyncio.run(test_weather_server())
