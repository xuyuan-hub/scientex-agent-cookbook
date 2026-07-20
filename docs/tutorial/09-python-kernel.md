# Step 09: Python 执行内核

## 目标

让 agent 能执行 Python 代码——在沙箱中运行数据分析、生成图表、处理数据。

## 前置条件

- 完成 [08-skill-system.md](08-skill-system.md)
- 理解工具调用 [04-tool-calling.md](04-tool-calling.md)

## 设计思路

### 两种内核模式

| | In-Process | Subprocess |
|---|---|---|
| 安全性 | ❌ 低（可访问宿主内存） | ✅ 高（独立进程） |
| 性能 | ✅ 快（无进程启动开销） | ❌ 慢（需启动子进程） |
| 状态保持 | ✅ 跨执行共享 namespace | ✅ 通过 JSON 协议维持 |
| 适用场景 | 开发/调试 | 生产环境 |

**默认使用 Subprocess 内核**（安全性优先）。

### Subprocess 通信协议

```
父进程 (agent)                      子进程 (kernel worker)
    │                                      │
    │── {"source": "x=1+1\\nprint(x)"} ──→│
    │                                      │ exec(...)
    │←── {"stdout": "2\\n", "stderr": ""}─│
    │                                      │
    │── {"source": "print(x*2)"} ────────→│
    │                                      │ exec(...)
    │←── {"stdout": "4\\n", "stderr": ""}─│
```

每行是一个 JSON 对象，用 `\n` 分隔。

### host 回调

内核中的代码可以通过 `host` 模块回调 agent 的能力：

```python
# 在内核中执行
host.llm("Summarize the results")          # 调用 LLM
host.read_file("data.csv")                 # 读取工作区文件
host.artifact_path("abc123")               # 获取产物路径
```

这通过子进程向父进程发送 `{"type": "host_call", ...}` 消息实现。

## 实现

### 1. 创建 src/scientex_agent/kernel.py

```python
"""Python execution kernel — in-process and subprocess variants."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class KernelResult:
    stdout: str = ""
    stderr: str = ""
    exit_status: str = "ok"   # "ok" | "error" | "timeout"
    error: str | None = None


class PythonKernel:
    """In-process Python kernel. Executes code in a persistent namespace.

    NOT safe for untrusted code. Use SubprocessPythonKernel for production.
    """

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)
        self.namespace: dict[str, Any] = {
            "__builtins__": __builtins__,
            "workspace": str(self.workspace),
        }

    def execute(self, source: str) -> KernelResult:
        stdout_parts = []
        stderr_parts = []

        def _write(text: str):
            stdout_parts.append(text)

        try:
            code = compile(source, "<kernel>", "exec")
            old_stdout_write = getattr(sys.stdout, "write", None)
            # Simple capture: redirect sys.stdout
            import io
            buf = io.StringIO()
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = buf
            sys.stderr = buf
            try:
                exec(code, self.namespace)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            return KernelResult(stdout=buf.getvalue())
        except Exception as e:
            return KernelResult(stderr=str(e), exit_status="error", error=str(e))

    def close(self) -> None:
        self.namespace.clear()


class SubprocessPythonKernel:
    """Subprocess-based Python kernel with JSON-line protocol.

    Safe for production. The kernel runs in an isolated child process.
    """

    def __init__(self, workspace: str | Path, *, timeout: float = 60.0) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.host_call_handler: Callable | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Launch the kernel worker subprocess."""
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _KERNEL_WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(self.workspace),
        )

    def execute(self, source: str) -> KernelResult:
        """Send code to the kernel and wait for the result."""
        if self._proc is None:
            self.start()

        with self._lock:
            assert self._proc and self._proc.stdin
            payload = json.dumps({"source": source, "filename": "<kernel>"})
            self._proc.stdin.write(payload + "\n")
            self._proc.stdin.flush()

            # Read response line
            line = self._proc.stdout.readline()
            if not line:
                return KernelResult(exit_status="error", error="Kernel process died")

            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                return KernelResult(exit_status="error", error=f"Bad kernel response: {line[:200]}")

            # Handle host calls (nested RPC)
            while result.get("type") == "host_call":
                handler_result = self._handle_host_call(result)
                self._proc.stdin.write(json.dumps(handler_result) + "\n")
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
                result = json.loads(line)

            return KernelResult(
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", ""),
                exit_status=result.get("exit_status", "ok"),
                error=result.get("error"),
            )

    def _handle_host_call(self, request: dict) -> dict:
        """Handle a host.* call from the kernel worker."""
        if self.host_call_handler is None:
            return {"type": "host_response", "ok": False, "error": "No host handler configured"}

        method = request.get("method", "")
        args = request.get("args", [])
        kwargs = request.get("kwargs", {})

        try:
            result = self.host_call_handler(method, args, kwargs)
            return {"type": "host_response", "ok": True, "result": result}
        except Exception as e:
            return {"type": "host_response", "ok": False, "error": str(e)}

    def stop(self) -> None:
        if self._proc:
            try:
                self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None

    def close(self) -> None:
        self.stop()


# === Kernel Worker Script ===

_KERNEL_WORKER_SCRIPT = r'''
import json
import sys
from pathlib import Path

namespace: dict = {"__builtins__": __builtins__}

# Host proxy — intercepts host.* calls
class _HostProxy:
    def __getattr__(self, method):
        def _call(*args, **kwargs):
            req = json.dumps({
                "type": "host_call",
                "method": method,
                "args": list(args),
                "kwargs": kwargs,
            })
            sys.stdout.write(req + "\n")
            sys.stdout.flush()
            resp_line = sys.stdin.readline()
            resp = json.loads(resp_line)
            if resp.get("ok"):
                return resp.get("result")
            raise RuntimeError(resp.get("error", "host call failed"))
        return _call

host = _HostProxy()
namespace["host"] = host

# Main loop: read JSON-line commands, execute, write results
for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line:
        continue
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        continue

    source = request.get("source", "")
    filename = request.get("filename", "<kernel>")

    try:
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = buf
        sys.stderr = buf
        try:
            exec(compile(source, filename, "exec"), namespace)
            response = {"stdout": buf.getvalue(), "stderr": "", "exit_status": "ok"}
        except Exception as e:
            response = {"stdout": buf.getvalue(), "stderr": str(e), "exit_status": "error", "error": str(e)}
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    except Exception as e:
        response = {"stdout": "", "stderr": str(e), "exit_status": "error", "error": str(e)}

    sys.__stdout__.write(json.dumps(response) + "\n")
    sys.__stdout__.flush()
'''


# === Sidecar Loader (inject skill helper code) ===

@dataclass(frozen=True)
class SidecarLoadResult:
    names: list[str] = field(default_factory=list)
    error: str | None = None


def validate_python_sidecar(source: str) -> SidecarLoadResult:
    """Validate that a Python sidecar only contains safe constructs.

    Allowed at module scope: function defs, imports, literal assignments.
    Forbidden: decorators, wildcard imports, non-literal defaults, class defs.
    """
    try:
        tree = ast.parse(source)
        defined_names = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                if node.decorator_list:
                    return SidecarLoadResult(error=f"Function '{node.name}' has decorators (not allowed)")
                # Check default values are literals
                for default in node.args.defaults:
                    if not isinstance(default, ast.Constant):
                        return SidecarLoadResult(
                            error=f"Function '{node.name}' has non-literal default (not allowed)"
                        )
                defined_names.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    defined_names.append(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        if alias.name == "*":
                            return SidecarLoadResult(error="Wildcard import not allowed")
                        defined_names.append(alias.asname or alias.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_names.append(target.id)
            elif isinstance(node, ast.ClassDef):
                return SidecarLoadResult(error=f"Class '{node.name}' not allowed in sidecar")
            # Allow ast.Expr (docstrings), ast.AnnAssign, ast.AugAssign
            elif isinstance(node, (ast.Expr, ast.AnnAssign, ast.AugAssign)):
                pass
            else:
                return SidecarLoadResult(
                    error=f"Unsupported construct: {type(node).__name__}"
                )
        return SidecarLoadResult(names=defined_names)
    except SyntaxError as e:
        return SidecarLoadResult(error=f"Syntax error: {e}")
```

### 2. 注册执行工具

```python
# src/scientex_agent/tools.py 中添加

def register_kernel_tools(registry: ToolRegistry, kernel: SubprocessPythonKernel) -> None:

    @registry.register(
        name="execute_python",
        description="Execute Python code in a persistent kernel. "
                    "Variables persist between calls. "
                    "Use for data analysis, visualization, calculations.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Can be multi-line."
                },
            },
            "required": ["code"],
        },
    )
    def execute_python(code: str) -> dict:
        result = kernel.execute(code)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_status": result.exit_status,
            "error": result.error,
        }
```

## 验证

```python
from scientex_agent.kernel import SubprocessPythonKernel
from pathlib import Path

kernel = SubprocessPythonKernel(Path("/tmp/kernel-test"))

# 基本执行
result = kernel.execute("x = sum(range(100))\nprint(f'sum = {x}')")
assert "sum = 4950" in result.stdout
assert result.exit_status == "ok"

# 状态保持
result = kernel.execute("print(f'x * 2 = {x * 2}')")
assert "x * 2 = 9900" in result.stdout

# 错误处理
result = kernel.execute("1/0")
assert result.exit_status == "error"
assert "division by zero" in result.stderr

kernel.close()
```

## 深入理解

### 为什么用子进程

1. **隔离**：用户代码崩溃不会影响 agent 主进程
2. **安全性**：可以限制文件系统访问、网络访问
3. **资源限制**：可以用 `resource` 模块限制 CPU/内存
4. **可清理**：子进程可以被 kill，资源会被 OS 回收

### host 回调的安全考虑

```python
# 在内核中，用户可以调用
host.llm("Generate malicious content")    # → 通过 host_call_handler 进行
host.read_file("/etc/passwd")            # → 应该在 handler 层做路径检查
```

所有 host 回调都应该：
1. 记录日志（谁调用了什么）
2. 验证参数（路径检查、权限检查）
3. 限制频率（防止滥用）

### 技能侧车注入

技能可以包含 `kernel.py` 文件，在 kernel 启动时自动加载：

```python
# skills/data-analysis/kernel.py
import pandas as pd
import numpy as np

def load_dataset(path: str):
    """Load a CSV or Excel file."""
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported format: {path}")
```

注入方式：在每次 `execute()` 时 prepend 侧车代码到 source 前面。

## 当前局限

1. 没有超时控制（死循环会永远挂起）
2. 没有内存限制
3. 文件系统访问无沙箱
4. 没有多内核管理（一个 frame 一个内核）

## 下一步

→ [10-agent-loop.md](10-agent-loop.md)
