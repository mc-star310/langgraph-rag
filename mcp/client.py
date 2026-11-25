"""
mcp/client.py —— 可选 MCP 工具接入
==================================
通过 langchain-mcp-adapters 把外部 MCP server 的工具接进 Agent。
默认关闭（MCP_ENABLED=false）。开启后，Agent 能调用更多外部能力（如文件系统、搜索等），
而无需改任何业务代码 —— 这正是现代 Agent 开发的"即插即用"范式。

示例：设 MCP_FS_ENABLED=true 会启动一个 filesystem MCP server，
让 Agent 能读写 MCP_FS_PATH 目录下的文件。
"""

import os
from langchain_mcp_adapters.client import MultiServerMCPClient


async def load_mcp_tools():
    if os.getenv("MCP_ENABLED", "false").lower() != "true":
        return []

    servers = {}
    if os.getenv("MCP_FS_ENABLED", "false").lower() == "true":
        servers["filesystem"] = {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                os.getenv("MCP_FS_PATH", "./corpus"),
            ],
            "transport": "stdio",
        }
    if not servers:
        return []

    client = MultiServerMCPClient(servers)
    return await client.get_tools()
