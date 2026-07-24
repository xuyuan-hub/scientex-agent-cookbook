# Step 07: MCP 集成（Model Context Protocol）

## 目标

让 Scientex 在保留 Step 04 工具调用体验的前提下，安全地发现并调用外部 MCP Server 的工具。MCP 工具不是“可信的内置函数”：它们有明确的服务器归属、生命周期、超时和审计记录。

## 前置条件

- 完成 [06-conversation-persistence.md](06-conversation-persistence.md)
- 理解 [04-tool-calling.md](04-tool-calling.md) 的 `ToolRegistry`
- 了解 `async` / `await` 的基本用法

## 与 Step 06 的边界

Step 06 保存 Project、Frame 和 Message；本章只增加一种工具来源。MCP 配置属于应用配置，MCP 调用记录属于 Frame 的可追溯事件，不能把远程服务器返回的内容直接当作系统指令或数据库命令。

## 设计思路

### MCP 在 Scientex 中的位置

```
                     ┌─────────────────────────┐
                     │ Agent / ToolRegistry    │
                     └────────────┬────────────┘
                                  │ 同一套工具契约
            ┌─────────────────────┴─────────────────────┐
            │                                           │
     内置 Python 工具                              MCPToolAdapter
                                                        │
                           ┌────────────────────────────┴────────────┐
                           │ MCPManager：连接、发现、超时、关闭       │
                           ├─────────────────────┬────────────────────┤
                        stdio（本地）       Streamable HTTP（远程）
```

MCP 的价值是标准化发现和调用，不是绕过权限模型。工具名称使用
`{server_id}__{tool_name}` 命名空间，例如 `pubmed__search`；这能避免不同 Server 的 `search`
互相覆盖，也让日志可以追溯到提供方。

### 选择传输方式

| 场景 | 传输 | 规则 |
|---|---|---|
| 开发机上受信任的脚本 | `stdio` | 命令、参数和工作目录必须来自配置，不从模型文本拼接 |
| 已部署的服务 | Streamable HTTP | 仅允许 HTTPS、显式 allowlist、连接和调用均有超时 |
| 旧服务兼容 | SSE | 只作为迁移选项，不再作为新的远程部署默认值 |

当前 MCP Python SDK 将 Streamable HTTP 作为生产部署的推荐传输；因此本教程的远程客户端以它为主。MCP
连接必须由应用 lifespan 管理，不能在每次 tool call 时新建进程或泄漏异步 context manager。

### 先定义配置和领域对象

配置是系统边界，先校验，再建立连接。不要允许 LLM 自己给出 `command`、`cwd` 或远程 URL。

```python
# src/scientex_agent/mcp_models.py
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{1,62}$")]
    transport: Literal["stdio", "streamable_http"]
    command: str | None = None
    args: tuple[str, ...] = ()
    url: AnyHttpUrl | None = None
    timeout_seconds: Annotated[float, Field(gt=0, le=120)] = 30
    enabled: bool = True

    @model_validator(mode="after")
    def validate_transport(self) -> "MCPServerConfig":
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio server requires command")
        if self.transport == "streamable_http" and not self.url:
            raise ValueError("streamable_http server requires url")
        return self


class MCPToolDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True)

    qualified_name: str
    server_id: str
    name: str
    description: str
    input_schema: dict
```

`MCPServerConfig` 应从受版本控制的 `mcp_servers.json` 或部署配置载入。密钥引用环境变量名称或系统密钥链中的别名，绝不写进配置文件、提示词或日志。

## 实现

### 1. 添加依赖并保存配置

```toml
# pyproject.toml
[project]
dependencies = [
    # ...Step 06 的依赖
    "mcp>=1.9",
    "pydantic>=2.0",
]
```

```json
// mcp_servers.json（示例；不放入真实 token）
{
  "servers": [
    {
      "id": "pubmed",
      "transport": "stdio",
      "command": "uvx",
      "args": ["--from", "scientex-pubmed-mcp", "pubmed-mcp"],
      "timeout_seconds": 20
    }
  ]
}
```

只在明确审查过命令来源后启用 `stdio` Server。对于远程 Server，启动时同时校验 URL 的 scheme 和主机 allowlist。

### 2. 创建 `MCPManager`

下面的对象只暴露本项目需要的三个动作：启动、列出工具、调用工具。`AsyncExitStack` 持有 SDK 返回的
context manager，从而保证服务关闭时 session 和子进程一并关闭。

```python
# src/scientex_agent/mcp_manager.py
from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Mapping
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from .mcp_models import MCPServerConfig, MCPToolDescriptor


class MCPManager:
    """Own all MCP sessions for one running Scientex application."""

    def __init__(self, configs: list[MCPServerConfig]) -> None:
        self._configs = {item.id: item for item in configs if item.enabled}
        self._stack = contextlib.AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, MCPToolDescriptor] = {}

    async def start(self) -> None:
        for config in self._configs.values():
            if config.transport == "stdio":
                streams = await self._stack.enter_async_context(
                    stdio_client(StdioServerParameters(
                        command=config.command or "",
                        args=list(config.args),
                    ))
                )
            else:
                streams = await self._stack.enter_async_context(
                    streamablehttp_client(str(config.url))
                )

            read_stream, write_stream, *_ = streams
            session = await self._stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await asyncio.wait_for(session.initialize(), config.timeout_seconds)
            self._sessions[config.id] = session
            await self._discover(config, session)

    async def aclose(self) -> None:
        await self._stack.aclose()
        self._sessions.clear()
        self._tools.clear()

    def list_tools(self) -> list[MCPToolDescriptor]:
        return list(self._tools.values())

    async def call_tool(
        self, qualified_name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        descriptor = self._tools[qualified_name]
        session = self._sessions[descriptor.server_id]
        timeout = self._configs[descriptor.server_id].timeout_seconds
        result = await asyncio.wait_for(
            session.call_tool(descriptor.name, dict(arguments)), timeout
        )
        return {
            "server_id": descriptor.server_id,
            "is_error": bool(getattr(result, "isError", False)),
            "content": _content_to_jsonable(result.content),
        }

    async def _discover(self, config: MCPServerConfig, session: ClientSession) -> None:
        response = await asyncio.wait_for(session.list_tools(), config.timeout_seconds)
        for tool in response.tools:
            qualified_name = f"{config.id}__{tool.name}"
            if qualified_name in self._tools:
                raise RuntimeError(f"duplicate MCP tool: {qualified_name}")
            self._tools[qualified_name] = MCPToolDescriptor(
                qualified_name=qualified_name,
                server_id=config.id,
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema or {"type": "object"},
            )


def _content_to_jsonable(content: list[Any]) -> list[dict[str, Any]]:
    """Preserve text/image/resource block shape instead of flattening it to text."""
    return [item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in content]
```

SDK minor versions occasionally change the exact tuple returned by a transport helper. Pin the resolved version in
`uv.lock` and put the transport adaptation in this one module; the rest of Scientex must not depend on SDK internals.

### 3. 适配为异步工具

Step 04 的 registry 原本只接受同步函数。这里增加 `aexecute()`：同步内置工具通过 `asyncio.to_thread`
执行，MCP 工具直接 await。不要在 FastAPI 或 LangGraph 的 event loop 中调用 `asyncio.run()`。

```python
# src/scientex_agent/tools.py（新增的协议和适配器，省略已有 registry）
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol


class AsyncTool(Protocol):
    name: str
    description: str
    input_schema: Mapping[str, Any]

    def invoke(self, arguments: Mapping[str, Any]) -> Awaitable[dict[str, Any]]: ...


class MCPToolAdapter:
    def __init__(self, manager: MCPManager, descriptor: MCPToolDescriptor) -> None:
        self.name = descriptor.qualified_name
        self.description = descriptor.description
        self.input_schema = descriptor.input_schema
        self._manager = manager

    async def invoke(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return await self._manager.call_tool(self.name, arguments)
```

`MCPToolAdapter` 的 schema 先由 provider 的 function-calling 校验，再由 MCP Server 再次校验。两层校验都不能代替
权限判断：注册时需要给每个 Server 设定 `read`、`write` 或 `network` 风险等级；在 Step 10 中，高风险调用会进入人工审批节点。

### 4. 在应用生命周期中启动

```python
# src/scientex_agent/app.py（骨架）
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    manager = MCPManager(load_mcp_configs())
    await manager.start()
    app.state.mcp = manager
    try:
        yield
    finally:
        await manager.aclose()
```

CLI 中可用 `asyncio.run(app.start())` 建立顶层事件循环；HTTP 服务则由 FastAPI lifespan 调用。两者都应共享同一个
`MCPManager`，而非为每一个请求重新发现工具。

## 验证

```bash
# 检查配置和连接；应显示带 server_id 前缀的工具名
uv run scientex_agent mcp doctor --config mcp_servers.json

# 只列出已启用、已经通过启动发现的工具
uv run scientex_agent tools list --source mcp

# 为测试 Server 调用一个只读工具
uv run scientex_agent tools call pubmed__search '{"query":"CRISPR", "limit":1}'
```

至少覆盖以下测试：

- 假 MCP session 返回两个同名工具时，限定名不会冲突；
- Server 初始化、调用超时和异常关闭都能释放资源；
- MCP 返回 `is_error` 时，Agent 得到结构化工具结果而不是假装成功；
- 未在配置中的 `command`、非 HTTPS 远程 URL 或未知 tool 名称会被拒绝。

## 深入理解

### 为什么不把 MCP 结果直接拼成字符串

MCP 返回的是内容块，可能是文本、资源引用或图片。保存原始结构和来源，既能让前端正确渲染，也能在后续验证中回答“这段结论来自哪个 Server、哪次调用”。把一切 `str()` 化会丢掉这个证据链。

### 连接失败的降级策略

单个可选 Server 故障不应阻断本地聊天：启动时记录其不可用状态并暴露在 `mcp doctor`；请求中明确告诉模型该工具不可用。标记为必需的 Server 则应让应用启动失败，避免悄悄以不完整能力执行科研任务。

## 当前局限

- 本章只处理工具；MCP 的 resources 和 prompts 先不映射到 Scientex 的技能系统。
- `stdio` 隔离的是协议通道，不隔离进程权限；不可信 Server 必须运行在容器、虚拟机或受限账户中。
- 本地 SQLite 尚未存储完整的 tool-run 审计表；Step 12 会把调用输入、输出和产物关联起来。

## 下一步

下一章把领域操作步骤做成可审查的 `SKILL.md`。技能负责指导 Agent 如何选择和组合工具，MCP 只负责提供工具本身。
