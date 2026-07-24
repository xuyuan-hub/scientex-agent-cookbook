# Step 11: HTTP API 服务（FastAPI + SSE）

## 目标

将 Scientex 的 Project、Frame 与 Agent Run 公开为版本化 HTTP API，并以结构化 SSE 流传递运行事件。API 是 Vue 前端和 CLI 的稳定契约，不是 `ScientexApp` 内部对象的直接镜像。

## 前置条件

- 完成 [10-agent-loop.md](10-agent-loop.md)
- 熟悉 FastAPI、Pydantic 和异步生成器

## 设计思路

### API 边界

```
Vue / CLI
   │  JSON request + SSE response
   ▼
/api/v1 FastAPI router
   │  Pydantic 校验、认证、HTTP 错误映射
   ▼
Application service
   │  业务规则、事务、RunEvent
   ▼
MetadataStore / AgentGraph / ArtifactStore
```

路由只做 HTTP 关心的事：校验、身份、状态码和序列化。它不直接拼 SQL、不构造 provider client，也不吞掉所有异常后返回字符串。

### 路由约定

| 方法 | 路径 | 作用 |
|---|---|---|
| `GET` | `/api/v1/health` | 进程健康与版本 |
| `GET, POST` | `/api/v1/projects` | 项目列表、创建项目 |
| `GET, PATCH` | `/api/v1/projects/{project_id}` | 读取、更新项目 |
| `GET, POST` | `/api/v1/projects/{project_id}/frames` | Frame 列表、创建 Frame |
| `GET` | `/api/v1/frames/{frame_id}/messages` | 已持久化消息 |
| `POST` | `/api/v1/frames/{frame_id}/runs` | 创建 Agent Run，返回 JSON 终态 |
| `POST` | `/api/v1/frames/{frame_id}/runs/stream` | 创建并流式返回 Agent Run |
| `POST` | `/api/v1/runs/{run_id}/resume` | 提交审批或恢复被中断 Run |
| `GET` | `/api/v1/tools`, `/api/v1/skills` | 只读能力目录 |

路径使用复数资源名，所有新端点位于 `/api/v1`。前端不能根据数据库字段猜测行为；请求与响应都有独立 Pydantic 模型。

### SSE 是事件协议，不是 token 字符串

浏览器需要在同一条请求中看到 token、工具进度、审批和终态。每个数据帧遵循标准 SSE 字段：

```
id: 17
event: message.delta
data: {"run_id":"run_123","seq":17,"data":{"text":"结果是"}}

id: 18
event: tool.started
data: {"run_id":"run_123","seq":18,"data":{"name":"pubmed__search"}}

id: 19
event: run.completed
data: {"run_id":"run_123","seq":19,"data":{"message_id":"msg_456"}}
```

`id` / `seq` 让客户端检测乱序或缺失；`event` 为类型，`data` 为 JSON。由于 `EventSource` 只能 GET，聊天使用 `fetch()` 读取 POST 响应流；Step 14 会实现一个完整的 SSE parser。

## 实现

### 1. 添加服务依赖

```toml
# pyproject.toml
[project]
dependencies = [
    # ...
    "fastapi>=0.110",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.0",
]
```

### 2. 定义 API schema

```python
# src/scientex_agent/api_models.py
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class CreateProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1, max_length=120)]
    description: Annotated[str, Field(max_length=4_000)] = ""
    context: Annotated[str, Field(max_length=16_000)] = ""


class CreateFrameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(max_length=120)] = ""
    parent_frame_id: str | None = None


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: Annotated[str, Field(min_length=1, max_length=100_000)]
    skill_ids: tuple[str, ...] = ()
    idempotency_key: Annotated[str, Field(min_length=16, max_length=128)]


class ResumeRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved_tool_ids: tuple[str, ...] = ()


class ProblemDetail(BaseModel):
    """RFC 9457-style error body; do not expose stack traces."""
    type: str
    title: str
    status: int
    detail: str
    instance: str
```

客户端为每次“发送消息”生成 UUID 形式的 `idempotency_key`。服务端用 `(actor_id, route, key)` 去重，以防网络重试生成两次 Agent Run。

### 3. 以 lifespan 创建应用依赖

```python
# src/scientex_agent/api_server.py
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .api_models import CreateRunRequest, ProblemDetail


@asynccontextmanager
async def lifespan(api: FastAPI):
    services = await build_application_services()
    api.state.services = services
    try:
        yield
    finally:
        await services.aclose()


def create_api() -> FastAPI:
    api = FastAPI(
        title="Scientex API",
        version=__version__,
        lifespan=lifespan,
        openapi_url="/api/v1/openapi.json",
        docs_url="/api/v1/docs",
    )
    install_exception_handlers(api)
    install_routes(api)
    return api
```

应用对象在启动时建立数据库、MCP manager、graph 和配置；请求只从 `request.app.state.services` 取得它们。这样测试可注入 fake services，生产也不会在每个请求重新初始化 MCP 会话。

### 4. 发送标准 SSE 事件

```python
def encode_sse(event: "RunEvent") -> str:
    payload = event.model_dump_json()
    return (
        f"id: {event.seq}\n"
        f"event: {event.type}\n"
        f"data: {payload}\n\n"
    )


# 以下 route 定义位于 install_routes(api) 内部。
@api.post("/api/v1/frames/{frame_id}/runs/stream")
async def stream_run(
    frame_id: str,
    body: CreateRunRequest,
    request: Request,
) -> StreamingResponse:
    services = request.app.state.services

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event in services.run_stream(
                frame_id=frame_id,
                content=body.content,
                skill_ids=body.skill_ids,
                idempotency_key=body.idempotency_key,
                actor=request.state.actor,
            ):
                if await request.is_disconnected():
                    await services.cancel_delivery(event.run_id)
                    return
                yield encode_sse(event)
        except DomainError as error:
            yield encode_sse(error.to_run_event())
        except Exception:
            request.app.state.logger.exception("run stream failed")
            yield "event: run.failed\ndata: {\"code\":\"internal_error\"}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
```

不要在 SSE generator 中把任意异常的 `str(error)` 返回给浏览器；错误细节常包含路径、令牌或 provider 响应。客户端断开只停止**投递**，是否取消实际 Run 由产品策略决定：只读检索可以继续完成并保存，昂贵或有副作用的 Run 应请求取消。

### 5. 一致的错误映射

```python
def install_exception_handlers(api: FastAPI) -> None:
    @api.exception_handler(DomainError)
    async def domain_error(request: Request, exc: DomainError):
        return JSONResponse(
            status_code=exc.status_code,
            content=ProblemDetail(
                type=f"https://scientex.dev/problems/{exc.code}",
                title=exc.public_title,
                status=exc.status_code,
                detail=exc.public_detail,
                instance=str(request.url.path),
            ).model_dump(),
            media_type="application/problem+json",
        )

    @api.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        request.app.state.logger.exception("unhandled request error")
        return JSONResponse(
            status_code=500,
            content={"type": "https://scientex.dev/problems/internal",
                     "title": "Internal server error", "status": 500,
                     "detail": "The request could not be completed.",
                     "instance": str(request.url.path)},
            media_type="application/problem+json",
        )
```

输入校验由 FastAPI/Pydantic 产生 422；领域中不存在的 Project/Frame 产生 404；审批冲突或过期恢复产生 409；未认证或无权限产生 401/403。所有错误都带 trace id，完整 stack trace 只进入已脱敏日志。

### 6. 开发时处理跨域，生产同源部署

```python
# 仅开发配置开启；允许来源必须是明确列表
from fastapi.middleware.cors import CORSMiddleware

if settings.dev_frontend_origin:
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[str(settings.dev_frontend_origin)],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["Content-Type", "X-Request-Id"],
    )
```

Step 14 的 Vite 开发服务器通过代理访问 `/api`；生产环境由 FastAPI 或反向代理在同一 origin 提供 `web/dist`。不要在生产中使用 `allow_origins=["*"]` 加 cookie/credential。

## 验证

```bash
# 启动本地服务
uv run uvicorn scientex_agent.api_server:create_api --factory --reload

# 健康检查与 OpenAPI 文档
curl http://127.0.0.1:8765/api/v1/health
open http://127.0.0.1:8765/api/v1/docs

# 用 curl 观察 SSE；真实客户端还应验证 event / id / data
curl -N -X POST http://127.0.0.1:8765/api/v1/frames/FRAME_ID/runs/stream \
  -H 'Content-Type: application/json' \
  -d '{"content":"解释 DNA 提取","idempotency_key":"7cfbd33c-3e6e-4f44-a130-9ebf4f93215b"}'
```

API 测试使用 `httpx.AsyncClient` 和临时 SQLite。至少覆盖 schema 错误、未知 Frame、重复 idempotency key、SSE event 顺序、审批恢复、断开后的取消策略，以及 domain exception 不泄漏内部异常文本。

## 深入理解

### SSE 与 WebSocket

此处通信是服务器单向的连续事件，用户输入和审批恢复仍是普通 HTTP POST；SSE 更符合代理、日志和断线处理模型。若未来需要多人协作光标或双向低延迟控制，再为那个明确场景增加 WebSocket，而不是为了 token streaming 预先引入它。

### OpenAPI 是前端契约的起点

FastAPI 自动生成的 OpenAPI 应在 CI 中导出并作 diff。Vue 侧可以从它生成 TypeScript 类型，或至少将 `api_models.py` 的响应示例固定为契约测试，避免前后端各自复制字段定义。

## 当前局限

- API 先假设单用户本地使用；身份、会话、CSRF 和服务端授权在 Step 16 根据部署模式完善。
- Run event 只支持当前连接实时消费，尚未提供按 `Last-Event-ID` 回放。
- Artifact 路由、导出和版本控制将在下一章加入。

## 下一步

下一章把 Project 变成可审计的工作空间：版本化 Artifact、可复现导出和与 Agent Run 关联的证据链。
