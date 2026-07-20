# Step 11: HTTP API 服务（FastAPI + SSE）

## 目标

将 Agent 包装为 HTTP API 服务，支持 RESTful 接口和 SSE 流式输出。

## 前置条件

- 完成 [10-agent-loop.md](10-agent-loop.md)

## 设计思路

### 为什么用 FastAPI

```
FastAPI 的优势：
- 原生异步（async/await）→ 与 LangGraph 的异步流式完美配合
- 自动生成 OpenAPI 文档 → /docs 端点即可查看
- SSE 支持 → StreamingResponse 原生支持
- 类型安全 → Pydantic 模型自动校验
```

### API 设计

```
GET  /health                             健康检查
GET  /projects                           列出项目
POST /projects                           创建项目
GET  /projects/{id}/frames               列出 Frame
POST /projects/{id}/frames               创建 Frame
POST /frames/{id}/chat                   发送消息（非流式）
POST /frames/{id}/chat/stream            发送消息（SSE 流式）
GET  /frames/{id}/messages               获取历史消息
GET  /tools                              列出可用工具
GET  /skills                             列出可用技能
```

### SSE 流式格式

```
data: {"token": "Hello"}

data: {"token": ", "}

data: {"token": "world"}

data: {"tool_start": {"name": "calculate", "input": {"expression": "1+1"}}}

data: {"tool_end": {"name": "calculate", "output": "2"}}

data: {"token": "The"}

data: {"token": " answer"}

data: {"token": " is"}

data: {"token": " 2"}

data: {"done": true}
```

## 实现

### 1. 添加依赖

```toml
[project]
dependencies = [
    ...
    "fastapi>=0.110",
    "uvicorn>=0.29",
]
```

### 2. 创建 src/scientex_agent/api_server.py

```python
"""FastAPI HTTP API for Scientex."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .app import ScientexApp


# === Request/Response Models ===

class CreateProjectRequest(BaseModel):
    name: str
    description: str = ""
    context: str = ""


class CreateFrameRequest(BaseModel):
    name: str = ""


class ChatRequest(BaseModel):
    content: str


# === App Factory ===

def create_app(scientex: ScientexApp) -> FastAPI:
    """Build the FastAPI application.

    Args:
        scientex: Initialized ScientexApp instance.

    Returns:
        FastAPI app ready to be served by uvicorn.
    """
    api = FastAPI(
        title="Scientex API",
        version="0.1.0",
        description="Local scientific agent platform API",
    )

    # === Health ===

    @api.get("/health")
    async def health() -> dict:
        return {"ok": True}

    # === Projects ===

    @api.get("/projects")
    async def list_projects() -> list[dict]:
        projects = scientex.list_projects()
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "context": p.context,
                "created_at": p.created_at,
                "updated_at": p.updated_at,
            }
            for p in projects
        ]

    @api.post("/projects")
    async def create_project(req: CreateProjectRequest) -> dict:
        project = scientex.create_project(
            name=req.name,
            description=req.description,
            context=req.context,
        )
        return {
            "id": project.id,
            "name": project.name,
            "description": project.description,
        }

    @api.get("/projects/{project_id}")
    async def get_project(project_id: str) -> dict:
        project = scientex.metadata.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "context": project.context,
        }

    # === Frames ===

    @api.get("/projects/{project_id}/frames")
    async def list_frames(project_id: str) -> list[dict]:
        frames = scientex.list_frames(project_id)
        return [
            {
                "id": f.id,
                "project_id": f.project_id,
                "name": f.name,
                "status": f.status,
                "created_at": f.created_at,
            }
            for f in frames
        ]

    @api.post("/projects/{project_id}/frames")
    async def create_frame(project_id: str, req: CreateFrameRequest) -> dict:
        frame = scientex.create_frame(project_id=project_id, name=req.name)
        return {
            "id": frame.id,
            "project_id": frame.project_id,
            "name": frame.name,
            "status": frame.status,
        }

    # === Chat ===

    @api.post("/frames/{frame_id}/chat")
    async def frame_chat(frame_id: str, req: ChatRequest) -> dict:
        """Send a message and get the full response."""
        try:
            result = await scientex.chat(frame_id=frame_id, content=req.content)
            return result
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @api.post("/frames/{frame_id}/chat/stream")
    async def frame_chat_stream(frame_id: str, req: ChatRequest):
        """Send a message and stream the response via SSE."""

        async def event_generator():
            try:
                async for event in scientex.chat_stream(
                    frame_id=frame_id, content=req.content
                ):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except KeyError as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    # === Messages ===

    @api.get("/frames/{frame_id}/messages")
    async def list_messages(frame_id: str) -> list[dict]:
        messages = scientex.metadata.list_messages(frame_id)
        return [
            {
                "idx": m.idx,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at,
            }
            for m in messages
        ]

    # === Tools ===

    @api.get("/tools")
    async def list_tools() -> list[dict]:
        registry = scientex._build_tool_registry()
        schemas = registry.schemas()
        return [
            {"name": s["function"]["name"], "description": s["function"]["description"]}
            for s in schemas
        ]

    # === Skills ===

    @api.get("/skills")
    async def list_skills() -> list[dict]:
        return scientex.skills.list_skills()

    @api.get("/skills/{name}")
    async def read_skill(name: str) -> dict:
        try:
            body = scientex.skills.read_skill(name)
            return {"name": name, "body": body}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Skill not found: {name}")

    # === Static Files (frontend) ===

    web_dir = Path(__file__).parent / "web"
    if web_dir.exists():
        api.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")

    return api


# === Server Launcher ===

def run_server(
    data_dir: str | Path = "~/.scientex_agent",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Start the Scientex HTTP server.

    This is the main entry point for `scientex_agent run-server`.
    """
    app = ScientexApp(Path(data_dir))
    app.initialize()

    api = create_app(app)

    uvicorn.run(api, host=host, port=port, log_level="info")
```

### 3. 更新 CLI

```python
# src/scientex_agent/cli.py

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="scientex_agent")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    # ... 之前的 chat 子命令 ...

    # run-server 子命令
    server_parser = subparsers.add_parser("run-server", help="Start the HTTP API server")
    server_parser.add_argument("--data-dir", default="~/.scientex_agent")
    server_parser.add_argument("--host", default="127.0.0.1")
    server_parser.add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)

    if args.command == "run-server":
        from .api_server import run_server
        run_server(data_dir=args.data_dir, host=args.host, port=args.port)
        return 0

    # ... 其他子命令 ...
```

## 验证

### 启动服务

```bash
uv run scientex_agent run-server --data-dir /tmp/scientex-test
# 输出: INFO: Uvicorn running on http://127.0.0.1:8765
```

### 测试 API

```bash
# 健康检查
curl http://127.0.0.1:8765/health

# 查看 API 文档
open http://127.0.0.1:8765/docs

# 创建项目
curl -X POST http://127.0.0.1:8765/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "Test", "description": "A test project"}'

# 创建 Frame（用返回的 project id）
curl -X POST http://127.0.0.1:8765/projects/abc123/frames \
  -H "Content-Type: application/json" \
  -d '{"name": "Chat 1"}'

# 发送消息（非流式）
curl -X POST http://127.0.0.1:8765/frames/def456/chat \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello!"}'

# 流式聊天
curl -N -X POST http://127.0.0.1:8765/frames/def456/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"content": "Count from 1 to 5"}'
```

### SSE 响应示例

```
data: {"token": "Sure"}

data: {"token": "!"}

data: {"token": " Here"}

data: {"token": " you"}

data: {"token": " go"}

data: {"done": true}
```

## 深入理解

### 为什么 SSE 而不是 WebSocket

| | SSE | WebSocket |
|---|---|---|
| 方向 | 单向（server → client） | 双向 |
| 协议 | HTTP（标准） | 升级后的 TCP |
| 重连 | 浏览器自动重连 | 需手动实现 |
| 代理兼容 | ✅ 标准 HTTP 代理 | ❌ 可能需要特殊配置 |
| 复杂度 | 简单 | 复杂 |

对于 AI 流式输出场景（server → client 流式推文本），SSE 是天然适配的。

### FastAPI 的 StreamingResponse

```python
async def event_generator():
    for i in range(5):
        yield f"data: {json.dumps({'token': str(i)})}\n\n"
        await asyncio.sleep(0.5)

return StreamingResponse(event_generator(), media_type="text/event-stream")
```

`StreamingResponse` 接受一个异步生成器，每个 `yield` 都立即发送到客户端。
不需要等待生成器完成。

### CORS 配置（如果需要从其他域名访问）

```python
from fastapi.middleware.cors import CORSMiddleware

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 当前局限

1. 没有认证/授权
2. 没有请求限流
3. 没有 HTTPS（本地服务一般不需要）

## 下一步

→ [12-project-management.md](12-project-management.md)
