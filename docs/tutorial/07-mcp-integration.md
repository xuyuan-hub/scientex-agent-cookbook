# Step 07: MCP 集成（Model Context Protocol）

## 目标

让 agent 能够调用外部 MCP 服务器上的工具（而非只能调用内置工具）。

## 前置条件

- 完成 [06-conversation-persistence.md](06-conversation-persistence.md)
- 熟悉工具调用机制 [04-tool-calling.md](04-tool-calling.md)

## 设计思路

### 什么是 MCP

Model Context Protocol (MCP) 是 Anthropic 提出的标准协议，让 LLM 应用可以通过标准接口访问外部工具和数据源。

```
没有 MCP：
  Agent → 内置工具（数量有限，在代码中写死）

有了 MCP：
  Agent → MCP Client → MCP Server 1 (PubMed, 文献搜索)
                      → MCP Server 2 (UniProt, 蛋白质数据)
                      → MCP Server 3 (ClinVar, 遗传变异)
                      → ...
```

### 传输方式

MCP 支持两种传输：

1. **stdio**：启动子进程，通过 stdin/stdout 通信
   - 适合本地工具，如 `python pubmed_server.py`
2. **SSE (HTTP)**：通过 Server-Sent Events 通信
   - 适合远程工具，如 `https://tools.example.com/mcp`

### 我们使用官方 `mcp` SDK

`pyproject.toml` 已经有 `mcp>=1.9` 依赖，不需要从头实现客户端。
SDK 提供：

```python
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp import ClientSession, StdioServerParameters
```

### 集成方式

```
Agent Loop
  └── ToolRegistry
        ├── built-in tools (read_file, calculate, ...)
        └── MCP tools (动态注册)
              ├── pubmed_search  (from MCP Server A)
              ├── pubmed_fetch   (from MCP Server A)
              ├── uniprot_search (from MCP Server B)
              └── ...
```

MCP 服务器的工具在启动时动态发现并注册到 ToolRegistry，对 Agent 来说和内置工具没有区别。

## 实现

### 1. 创建 src/scientex_agent/mcp_connector.py

```python
"""MCP connector using the official mcp SDK."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from typing import Any


async def connect_stdio_server(
    command: str,
    args: list[str] | None = None,
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> tuple[Any, Any]:  # (read_stream, write_stream)
    """Connect to an MCP server via stdio.

    Args:
        command: Executable path or command name.
        args: Command-line arguments.
        env: Environment variables to pass.
        cwd: Working directory.

    Returns:
        (read_stream, write_stream) tuple for ClientSession.
    """
    from mcp.client.stdio import stdio_client
    from mcp import StdioServerParameters

    params = StdioServerParameters(
        command=command,
        args=args or [],
        env=env,
        cwd=cwd,
    )
    # stdio_client returns an async context manager
    return await stdio_client(params).__aenter__()


async def connect_sse_server(url: str) -> tuple[Any, Any]:
    """Connect to an MCP server via SSE.

    Args:
        url: SSE endpoint URL.

    Returns:
        (read_stream, write_stream) tuple for ClientSession.
    """
    from mcp.client.sse import sse_client

    return await sse_client(url).__aenter__()


class MCPServerConnection:
    """Manages a single MCP server connection and its tools.

    Usage::

        async with MCPServerConnection(
            command="python", args=["pubmed_server.py"]
        ) as conn:
            tools = await conn.list_tools()
            for tool in tools:
                print(tool.name, tool.description)

            result = await conn.call_tool("pubmed_search", {"query": "covid"})
    """

    def __init__(
        self,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        sse_url: str | None = None,
        server_name: str = "",
    ) -> None:
        if not command and not sse_url:
            raise ValueError("Must provide either command (stdio) or sse_url")
        self.command = command
        self.args = args or []
        self.sse_url = sse_url
        self.server_name = server_name or command or sse_url or "unnamed"
        self._session: Any = None
        self._read: Any = None
        self._write: Any = None
        self._tools: list[MCPServerTool] = []

    async def __aenter__(self) -> "MCPServerConnection":
        from mcp import ClientSession

        if self.command:
            self._read, self._write = await connect_stdio_server(
                self.command, self.args
            )
        else:
            self._read, self._write = await connect_sse_server(self.sse_url)

        self._session = ClientSession(self._read, self._write)
        await self._session.__aenter__()
        await self._session.initialize()

        # Discover tools
        result = await self._session.list_tools()
        self._tools = [
            MCPServerTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema or {"type": "object", "properties": {}},
            )
            for tool in (result.tools or [])
        ]

        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._session:
            await self._session.__aexit__(*args)
        # Close streams if needed
        for stream in (self._read, self._write):
            if stream and hasattr(stream, "close"):
                with contextlib.suppress(Exception):
                    stream.close()

    @property
    def tools(self) -> list["MCPServerTool"]:
        return list(self._tools)

    async def list_tools(self) -> list["MCPServerTool"]:
        result = await self._session.list_tools()
        self._tools = [
            MCPServerTool(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema or {"type": "object", "properties": {}},
            )
            for t in (result.tools or [])
        ]
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._session.call_tool(name, arguments)
        # Extract text from content blocks
        parts = []
        for block in (result.content or []):
            if hasattr(block, "text") and block.text:
                parts.append(block.text)
            elif isinstance(block, dict):
                parts.append(str(block))
        return "\n".join(parts)


class MCPServerTool:
    """Metadata about an MCP server tool."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
```

### 2. 同步包装器（用于同步 Agent Loop）

当前 agent loop 是同步的，但 MCP SDK 是异步的。
需要一个同步适配器：

```python
# src/scientex_agent/mcp_sync.py

"""Synchronous wrapper around async MCP connector."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from .tools import ToolRegistry
from .mcp_connector import MCPServerConnection


def _run_async(coro):
    """Run an async coroutine in a new event loop (thread-safe)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class SyncMCPServer:
    """Synchronous MCP server connection.

    Internally manages an async event loop in a background thread.
    """

    def __init__(
        self,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        sse_url: str | None = None,
        server_name: str = "",
    ) -> None:
        self._conn_params = dict(
            command=command, args=args, sse_url=sse_url, server_name=server_name
        )
        self._conn: MCPServerConnection | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> list[dict]:
        """Start the MCP connection and return discovered tools."""
        self._loop = asyncio.new_event_loop()
        self._conn = MCPServerConnection(**self._conn_params)

        # Run __aenter__ in the loop
        self._loop.run_until_complete(self._conn.__aenter__())

        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self._conn.tools
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self._conn is None or self._loop is None:
            raise RuntimeError("MCP server not started")
        return self._loop.run_until_complete(
            self._conn.call_tool(name, arguments)
        )

    def close(self) -> None:
        if self._conn and self._loop:
            self._loop.run_until_complete(self._conn.__aexit__(None, None, None))
        if self._loop:
            self._loop.close()


def register_mcp_tools(
    registry: ToolRegistry,
    server: SyncMCPServer,
    *,
    prefix: str = "",
) -> list[str]:
    """Register all tools from an MCP server into a ToolRegistry.

    Args:
        registry: The ToolRegistry to register into.
        server: A started SyncMCPServer.
        prefix: Namespace prefix (usually the server name).

    Returns:
        List of registered tool names.
    """
    tools = server.start()
    registered = []

    for tool_info in tools:
        tool_name = f"{prefix}_{tool_info['name']}" if prefix else tool_info["name"]
        if registry.has(tool_name):
            continue

        # Capture by value in closure
        _name = tool_info["name"]

        def make_handler(mcp_name: str, mcp: SyncMCPServer):
            def handler(**kwargs):
                result = mcp.call_tool(mcp_name, kwargs)
                return {"ok": True, "content": result}
            return handler

        registry._handlers[tool_name] = make_handler(_name, server)
        registry._schemas.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_info["description"],
                "parameters": tool_info["input_schema"],
            },
        })
        registered.append(tool_name)

    return registered
```

### 3. 示例 MCP 服务器

```python
# examples/simple_mcp_server.py
"""A minimal MCP server using the mcp SDK.

Run: uv run python examples/simple_mcp_server.py
"""

from mcp.server import Server
from mcp.server.stdio import stdio_server

app = Server("simple-calculator")

@app.tool()
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A Python math expression, e.g. '2 + 3 * 4'
    """
    import math
    allowed = {"abs": abs, "round": round, "sqrt": math.sqrt, "pi": math.pi, "e": math.e}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@app.tool()
def echo(message: str) -> str:
    """Echo back the message."""
    return message

if __name__ == "__main__":
    import asyncio
    async def run():
        async with stdio_server() as (read, write):
            await app.run(read, write, app.create_initialization_options())
    asyncio.run(run())
```

## 验证

```python
import asyncio
from scientex_agent.mcp_connector import MCPServerConnection

async def test():
    # 连接到示例 MCP 服务器
    async with MCPServerConnection(
        command="python", args=["examples/simple_mcp_server.py"]
    ) as conn:
        print("Tools discovered:")
        for tool in conn.tools:
            print(f"  - {tool.name}: {tool.description}")

        # 调用工具
        result = await conn.call_tool("calculate", {"expression": "sqrt(144)"})
        print(f"calculate('sqrt(144)') = {result}")

asyncio.run(test())
```

预期输出：
```
Tools discovered:
  - calculate: Evaluate a mathematical expression.
  - echo: Echo back the message.
calculate('sqrt(144)') = 12.0
```

## 深入理解

### MCP 协议层次

```
Application Layer:  你的 agent 应用
    ↕
Tool Layer:         ToolRegistry (统一工具接口)
    ↕
MCP Client:         MCPServerConnection (mcp SDK)
    ↕
Transport:          JSON-RPC 2.0 over stdio/SSE
    ↕
MCP Server:         外部工具服务器 (Python/Node/Go)
```

### 为什么用异步

MCP SDK 是异步的（`async/await`），因为：
1. MCP 服务器可能慢（网络请求、数据库查询）
2. 多个 MCP 服务器可以并发操作
3. SSE 传输天然是异步的

但我们的 agent loop 目前是同步的——所以需要 `SyncMCPServer` 适配器。
后续迁移到 LangGraph（原生异步）后，可以直接使用异步客户端。

### 安全考虑

```
MCP Server
  ├── tools/list  → 发现工具有哪些
  ├── tools/call  → 调用工具
  └── (未来) resources/*, prompts/*
```

应该限制 MCP 服务器可以执行的操作：
- 设置文件系统访问白名单
- 限制网络访问范围
- 工具调用权限分级（只读/读写/特权）

## 当前局限

1. 同步/异步桥接有性能损耗
2. 没有 MCP 服务器的权限管理系统
3. 服务器崩溃时没有自动重启机制

## 下一步

→ [08-skill-system.md](08-skill-system.md)
