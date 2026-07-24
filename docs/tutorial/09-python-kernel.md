# Step 09: Python 执行内核

## 目标

提供可复用、可取消、可审计的 Python 计算运行时，用于数据整理和科学分析。它支持会话级变量状态，但不会把“子进程”误称为安全沙箱：处理不可信代码时，仍需由容器、虚拟机或受限账户提供真正的隔离。

## 前置条件

- 完成 [08-skill-system.md](08-skill-system.md)
- 了解 [07-mcp-integration.md](07-mcp-integration.md) 的异步资源管理

## 与 Step 08 的边界

Skill 可以说明分析的步骤，Python Kernel 负责执行明确的代码片段。Kernel 不读取任意宿主机文件、不继承全部环境变量，也不让代码随意调用应用内部对象；与项目交互只能经过窄且可审计的 capability API。

## 设计思路

### 两种运行配置

| 配置 | 用途 | 隔离与限制 |
|---|---|---|
| `trusted-local` | 开发者或本机可信分析 | 独立子进程、最小环境、超时、输出上限 |
| `isolated` | 用户提供或远程执行的代码 | 容器/微虚拟机、只读镜像、非 root、网络默认关闭、CPU/内存/磁盘配额 |

Step 09 实现运行时协议和 `trusted-local`；`isolated` 是部署层的适配接口。仅靠删除 `open()` 或黑名单 import 不能安全执行 Python，不能把它作为安全方案。

### JSON Lines 协议

长驻进程减少导入开销，也让变量可以在同一 Frame 内延续。host 和 worker 只用一行一个 JSON 的双向协议通信：

```
host → {"id":"run_01","type":"execute","code":"x = 2\nx * 21","timeout_ms":10000}
worker → {"id":"run_01","type":"stdout","text":"42\n"}
worker → {"id":"run_01","type":"result","value_repr":"42","mime":"text/plain"}
worker → {"id":"run_01","type":"complete","ok":true}

host → {"id":"run_02","type":"cancel"}
worker → {"id":"run_02","type":"complete","ok":false,"error":{"code":"cancelled"}}
```

协议必须有 `id`，才能把并发请求、取消和日志正确关联。事件是结构化数据，不要混在 stdout 中用特殊字符串解析。

### 能力而非对象注入

旧式设计常把一个可调用的 `host` Python 对象塞给用户代码。这会扩大可见 API，并难以追踪权限。本教程改为内核事件：

```
代码 → emit_capability_request("artifact.read", {"artifact_id": "..."})
     → host 校验项目、用户和工具策略
     → host 返回具有限制的 JSON 结果
```

第一版只开放只读、明确声明的能力；写入产物、网络访问和执行外部程序由 Step 10 的审批策略控制。

## 实现

### 1. 定义运行时消息模型

```python
# src/scientex_agent/kernel_models.py
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    type: Literal["execute"] = "execute"
    code: Annotated[str, Field(min_length=1, max_length=100_000)]
    timeout_ms: Annotated[int, Field(ge=100, le=120_000)] = 10_000


class KernelEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["stdout", "stderr", "result", "error", "complete", "capability_request"]
    text: str | None = None
    value_repr: str | None = None
    mime: str | None = None
    payload: dict[str, Any] | None = None
    ok: bool | None = None
```

Pydantic 在 host 和 worker 两端校验消息。输出、code 和 timeout 都有上限，防止单个请求意外耗尽内存或把 Web SSE 塞满。

### 2. 创建异步 host

```python
# src/scientex_agent/kernel.py
from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from collections.abc import AsyncIterator

from .kernel_models import ExecuteRequest, KernelEvent


class PythonKernel:
    """Long-lived worker bound to one Frame; not a security sandbox."""

    def __init__(self, *, worker_module: str = "scientex_agent.kernel_worker") -> None:
        self._worker_module = worker_module
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._process and self._process.returncode is None:
            return
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
        }
        self._process = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-m", self._worker_module,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )

    async def execute(self, request: ExecuteRequest) -> AsyncIterator[KernelEvent]:
        await self.start()
        assert self._process and self._process.stdin and self._process.stdout
        async with self._lock:  # 一个解释器一次只跑一个 cell
            self._process.stdin.write(
                (request.model_dump_json() + "\n").encode("utf-8")
            )
            await self._process.stdin.drain()
            try:
                async with asyncio.timeout(request.timeout_ms / 1000):
                    async for event in self._read_until_complete(request.id):
                        yield event
            except TimeoutError:
                await self.stop()
                yield KernelEvent(
                    id=request.id, type="complete", ok=False,
                    payload={"code": "timeout"},
                )

    async def _read_until_complete(self, request_id: str) -> AsyncIterator[KernelEvent]:
        assert self._process and self._process.stdout
        while line := await self._process.stdout.readline():
            event = KernelEvent.model_validate_json(line)
            if event.id != request_id:
                continue
            yield event
            if event.type == "complete":
                return
        raise RuntimeError("kernel worker terminated before complete event")

    async def stop(self) -> None:
        if not self._process or self._process.returncode is not None:
            return
        self._process.terminate()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._process.wait(), timeout=2)
        if self._process.returncode is None:
            self._process.kill()
            await self._process.wait()
```

每个 Frame 使用一个 kernel，Frame 结束或超时后销毁。每次执行都复用该 Frame 的 namespace，不能跨项目复用解释器状态。

### 3. Worker 只实现协议，不承担隔离

```python
# src/scientex_agent/kernel_worker.py（核心循环，省略富显示支持）
from __future__ import annotations

import ast
import contextlib
import io
import json
import sys
import traceback
from typing import Any

namespace: dict[str, Any] = {"__name__": "__scientex_kernel__"}


def emit(event: dict[str, Any]) -> None:
    sys.__stdout__.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


def execute(request: dict[str, Any]) -> None:
    request_id = request["id"]
    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        module = ast.parse(request["code"], mode="exec")
        tail: ast.expr | None = None
        if module.body and isinstance(module.body[-1], ast.Expr):
            tail = module.body.pop().value

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(module, "<scientex-cell>", "exec"), namespace)
            value = eval(compile(ast.Expression(tail), "<scientex-cell>", "eval"), namespace) if tail else None

        if stdout.getvalue():
            emit({"id": request_id, "type": "stdout", "text": stdout.getvalue()[:65_536]})
        if stderr.getvalue():
            emit({"id": request_id, "type": "stderr", "text": stderr.getvalue()[:65_536]})
        if tail is not None:
            emit({"id": request_id, "type": "result", "value_repr": repr(value)[:16_384],
                  "mime": "text/plain"})
        emit({"id": request_id, "type": "complete", "ok": True})
    except Exception as error:
        emit({"id": request_id, "type": "error", "payload": {
            "code": "execution_error", "message": str(error),
            "traceback": traceback.format_exc(limit=20),
        }})
        emit({"id": request_id, "type": "complete", "ok": False})


for line in sys.stdin:
    execute(json.loads(line))
```

真实 worker 还应实现取消消息、输出计数而不是简单截断、图像 MIME 输出和 capability RPC。关键是把这些增强留在协议层，避免前端或 Agent 依赖 Python 的打印格式。

### 4. 将执行记录为产物候选

每个 kernel event 都带 `run_id`、`frame_id`、开始/结束时间、kernel image/version 和 code SHA-256。代码、stdout、stderr、生成文件先写入临时运行目录；Step 12 再将被用户保留的结果升级为版本化 Artifact。

## 验证

```bash
# 启动一个可信本地内核并执行一段确定性代码
uv run scientex_agent kernel run --code 'import math; math.factorial(6)'

# 同一 Frame 下第二段代码能读到第一段变量
uv run scientex_agent kernel repl --frame demo
>>> x = 21
>>> x * 2

# 无穷循环必须在 timeout 后返回失败，内核随后可重启
uv run scientex_agent kernel run --timeout-ms 200 --code 'while True: pass'
```

测试应覆盖：尾表达式、stdout/stderr、语法错误、超时后的进程回收、Frame 间 namespace 隔离，以及超过输出上限时的明确事件。对 `isolated` 运行器，另写集成测试确认网络、写入目录和资源限制确实由运行环境强制执行。

## 深入理解

### 为什么子进程不是安全边界

子进程仍与主进程拥有同一用户权限，可能读取用户可读的文件、访问网络或消耗机器资源。它用于故障隔离和生命周期管理；面对敌对代码，必须使用操作系统权限、容器或微虚拟机，并将宿主目录以最小、只读方式挂载。

### 为什么保留状态但不无限保留

Notebook 式分析需要 `x` 在下一格可用，但长期状态会导致不可复现和内存膨胀。一个 Frame 的 kernel 生命周期、可见 package 版本、输入 Artifact 和 code hash 都应记录；用户需要复现时，使用干净运行器重放这些输入。

## 当前局限

- `trusted-local` 不是不可信代码沙箱，且还没有容器运行器实现。
- 第一版一次只执行一个 cell，不支持并发 cell 或 notebook 协作。
- 还没有把 DataFrame、图像和文件自动转换为 Artifact；这属于 Step 12 的产物层。

## 下一步

下一章把 Provider、内置工具、MCP、Skill 和 Kernel 放入一个可恢复的 Agent 工作流。执行前的策略检查和人工审批会在编排层完成。
