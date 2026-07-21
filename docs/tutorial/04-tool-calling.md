# Step 04: 工具调用（Function Calling）

## 目标

让 LLM 能够"做事"而不只是"说话"——调用预定义的函数来执行实际操作。

## 前置条件

- 完成 [03-streaming.md](03-streaming.md)

## 设计思路

### 什么是 Function Calling

LLM 本身只能生成文本。Function Calling 让 LLM 决定**何时调用哪个函数**以及**传什么参数**。

```
用户: "现在几点了？"
LLM (without tools): "抱歉，我无法获取实时信息。"

用户: "现在几点了？"
LLM (with tools):  → 决定调用 get_current_time()
                   → 收到结果: "2026-07-20T15:30:00+08:00"
                   → 回复: "现在是 2026 年 7 月 20 日下午 3:30。"
```

### 工作流程

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  用户输入  │ →  │  LLM 判断 │ →  │  执行工具  │
│           │     │  是否调用  │     │  获取结果  │
└──────────┘     │  工具？    │     └──────────┘
                 └─────┬─────┘          │
                       │                │
                  text │          tool_result
                       │                │
                 ┌─────▼─────┐     ┌────▼─────┐
                 │  直接回复  │     │ 回传 LLM  │
                 │  给用户    │     │ 生成最终  │
                 └───────────┘     │   回复    │
                                   └──────────┘
```

### 关键数据结构

```python
# 1. 工具定义（告诉 LLM 有哪些工具可用）
tool_schema = {
    "type": "function",
    "function": {
        "name": "caculate",
        "description": "Evaluate a mathematical expression. Supports +,-,*,/,** and common math functions.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Python math expression, e.g. '2 + 3 * 4' or 'sqrt(16)'"
                }
            },
            "required": ["expression"]
        }
    }
}

# 2. LLM 返回的 tool call（告诉我们要执行什么）
# response.choices[0].message.tool_calls = [
#     ToolCall(id="call_123", name="caculate", arguments='{"expression":"3**10"}')
# ]

# 3. 我们把执行结果回传给 LLM（tool role message）
# {"role": "tool", "tool_call_id": "call_123", "content": "{\"ok\": true, \"result\": 59049}"}
```

### 多轮工具调用的挑战

LLM 可能在一轮中调用多个工具，或者多轮连续调用：

```
用户: "搜索 covid 相关论文，然后翻译第一篇的标题"
  → Round 1: LLM calls search_papers("covid")
  → Round 2: LLM calls translate(title_of_first_paper)
  → Round 3: LLM generates final response
```

这需要一个**循环**来处理：每次 LLM 返回 tool calls → 执行工具 → 把结果回传 → 继续循环，直到 LLM 不再调用工具。

## 实现

### 1. 创建 src/scientex_agent/tools.py

```python
"""Tool definitions and execution for function calling."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


# === Tool Schema Definition ===

def tool_schema(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Create an OpenAI-compatible tool schema.

    Args:
        name: Tool name. Use snake_case. Must match the registered handler.
        description: What the tool does. LLM uses this to decide when to call it.
        parameters: JSON Schema for the arguments.

    Returns:
        A tool definition dict ready to pass to the LLM.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


# === Tool Registry ===

class ToolRegistry:
    """A simple registry mapping tool names to handler functions.

    Usage::

        registry = ToolRegistry()

        @registry.register("add", "Add two numbers", {...})
        def add(a: int, b: int) -> int:
            return a + b

        # Get OpenAI-compatible schemas
        schemas = registry.schemas()

        # Execute a tool call from the LLM
        result = registry.execute("add", {"a": 1, "b": 2})
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Callable] = {}
        self._schemas: list[dict[str, Any]] = []

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ):
        """Decorator to register a function as a tool.

        The decorated function becomes the handler. Its name must match `name`.
        """

        def decorator(func: Callable) -> Callable:
            self._handlers[name] = func
            self._schemas.append(tool_schema(name, description, parameters))
            return func

        return decorator

    def schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool schemas for all registered tools."""
        return list(self._schemas)

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return the result.

        Result is a dict suitable for a 'tool' role message content.
        """
        handler = self._handlers.get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}

        try:
            result = handler(**arguments)
            return {"ok": True, "result": result}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def has(self, name: str) -> bool:
        return name in self._handlers


# === Built-in Tools ===

def registry_default_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """Register a set of basic utility tools and return the registry.

    If *registry* is ``None`` a fresh :class:`ToolRegistry` is created.
    """
    if registry is None:
        registry = ToolRegistry()

    @registry.register(
        name="get_current_time",
        description="Get the current date and time in a given timezone.",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone, e.g. 'Asia/Shanghai', 'America/New_York'"
                }
            },
            "required": ["timezone"],
        },
    )
    def get_current_time(timezone:str) -> str:
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo

        try:
            tz = ZoneInfo(timezone)
            now = datetime.now(tz)
            return now.strftime("%Y-%m-%d %H:%M:%S%Z")
        except Exception:
            # Fallback to UTC
            return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%Z")

    @registry.register(
        name="caculate",
        description="Evaluate a mathematical expression. Supports +,-,*,/,** and common math functions.",
        parameters={
            "type":"object",
            "properties":{
                "expression":{
                    "type":"string",
                    "description":"A Python math expression, e.g. '2 + 3 * 4' or 'sqrt(16)'"
                },
            },
            "required":["expression"]
        },
    )
    def caculate(expression:str) -> float |str:
        import math

        # Safe eval: only allow math functions, numbers, and basic operators
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e,
        }
        try:
            result = eval(expression,{"__builtin__":{}},allowed_names)
            return result
        except Exception as e:
            return f"Error: {e}"
    return registry
```

> 📄 最新代码：[code/04-tool-calling/src/scientex_agent/tools.py](../../code/04-tool-calling/src/scientex_agent/tools.py)

#### 代码详解

**1. `from __future__ import annotations` 的作用**

这是 Python 的一个特殊导入，让类型注解延迟求值。好处是：

- 可以在类型注解中引用还没定义的类名（前向引用），不用加引号
- 提升模块加载性能（注解不会立即求值）

```python
# 没有 __future__ 时，引用自身类名需要加引号：
def send(self) -> "ChatSession": ...

# 有了 __future__ 后，直接写：
def send(self) -> ChatSession: ...
```

**2. `tool_schema` 函数**

这个函数很简单，就是把工具的名称、描述、参数打包成 OpenAI API 需要的格式：

```python
# 输入
tool_schema("caculate", "计算表达式", {"type": "object", ...})

# 输出（OpenAI API 需要的格式）
{
    "type": "function",
    "function": {
        "name": "caculate",
        "description": "计算表达式",
        "parameters": {...}
    }
}
```

**3. `ToolRegistry` 类的装饰器模式（难点）**

`register` 方法使用了一个**装饰器工厂**（decorator factory）模式，这是 Python 中比较高级的用法。让我们拆解：

```python
def register(self, name, description, parameters):
    # 这是一个"装饰器工厂"——它返回一个装饰器
  
    def decorator(func):
        # 这是真正的装饰器
        self._handlers[name] = func          # 保存函数本身
        self._schemas.append(...)            # 保存工具的 schema
        return func                          # 返回原函数（不修改它）
  
    return decorator  # 返回装饰器
```

**执行流程图解：**

```python
@registry.register("caculate", "计算表达式", {...})
def caculate(expression: str):
    return eval(expression)
```

等价于：

```python
# 第 1 步：调用 register，返回 decorator 函数
decorator = registry.register("caculate", "计算表达式", {...})

# 第 2 步：用 decorator 装饰 caculate 函数
caculate = decorator(caculate)
```

在 `decorator(caculate)` 执行时：

1. 把 `caculate` 函数存到 `self._handlers["caculate"]`
2. 把工具的 schema 存到 `self._schemas` 列表
3. 返回原来的 `caculate` 函数（函数本身不变）

这样，`caculate` 函数就被"注册"了——我们可以在需要时通过名字找到它并调用。

**4. `execute` 方法中的 `**arguments`**

```python
result = handler(**arguments)
```

`**` 是 Python 的"解包"操作符。假设：

```python
handler = caculate  # 函数定义：def caculate(expression: str)
arguments = {"expression": "2 + 3"}

# handler(**arguments) 等价于：
handler(expression="2 + 3")
# 也就是：
caculate(expression="2 + 3")
```

`**arguments` 把字典的键值对"解包"成关键字参数传给函数。这样我们就能用字典动态地调用任意函数。

**5. `caculate` 函数中的安全 eval**

```python
eval(expression, {"__builtin__": {}}, allowed_names)
```

`eval()` 可以执行字符串形式的 Python 表达式，但直接用它很危险（用户可能执行恶意代码）。这里用三个参数限制它：

| 参数                    | 作用                                                   |
| ----------------------- | ------------------------------------------------------ |
| `expression`          | 要执行的表达式字符串                                   |
| `{"__builtin__": {}}` | 禁用所有内置函数（如`open`, `exec`），防止危险操作 |
| `allowed_names`       | 只允许使用白名单中的函数（如`sqrt`, `sin`）        |

这样，用户只能做数学计算，不能执行危险操作。

### 2. 创建 src/scientex_agent/agent_loop.py

```python
"""Simple tool-calling agent loop."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

from openai import OpenAI

from .tools import ToolRegistry


def _execute_tool_calls(
    response: Any,
    registry: ToolRegistry,
) -> list[dict[str, Any]]:
    """Execute all tool calls in a response and return tool result messages."""
    tool_messages = []
    for tc in response.choices[0].message.tool_calls or []:
        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        result = registry.execute(tc.function.name, args)
        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result, ensure_ascii=False),
        })
    return tool_messages


def chat_with_tools(
    client: OpenAI,
    model: str,
    messages: list[dict],
    registry: ToolRegistry,
    *,
    max_rounds: int = 10,
    **kwargs,
) -> str:
    """Non-streaming chat with tool calling.

    Args:
        client: OpenAI client.
        model: Model name.
        messages: Initial messages (system + history + new user message).
        registry: ToolRegistry with registered tools.
        max_rounds: Maximum LLM → tool → LLM rounds.
        **kwargs: Passed to client.chat.completions.create.

    Returns:
        Final assistant text response.
    """
    schemas = registry.schemas()

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=schemas or None,
            **kwargs,
        )

        choice = response.choices[0]
        msg = choice.message

        # No tool calls → done
        if not msg.tool_calls:
            return msg.content or ""

        # Store assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute tools and append results
        tool_results = _execute_tool_calls(response, registry)
        messages.extend(tool_results)

    return messages[-1].get("content", "") if messages else ""


def chat_with_tools_stream(
    client: OpenAI,
    model: str,
    messages: list[dict],
    registry: ToolRegistry,
    *,
    max_rounds: int = 10,
    **kwargs,
) -> Generator[str, None, None]:
    """Streaming chat with tool calling.

    Yields text tokens. Tool calls are executed internally (not yielded).
    """
    schemas = registry.schemas()

    for _ in range(max_rounds):
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=schemas or None,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )

        # Accumulate streaming response
        full_content = ""
        tool_calls_data: dict[int, dict] = {}

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Text
            if delta.content:
                full_content += delta.content
                yield delta.content

            # Tool calls (accumulate across chunks)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {
                            "id": tc_delta.id or "",
                            "function": {"name": "", "arguments": ""},
                            "type": "function",
                        }
                    if tc_delta.id:
                        tool_calls_data[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_data[idx]["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_data[idx]["function"]["arguments"] += tc_delta.function.arguments

        # If no tool calls, we're done
        if not tool_calls_data:
            return

        # Store assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": full_content or None,
            "tool_calls": list(tool_calls_data.values()),
        })

        # Execute tools and append results
        for tc_data in tool_calls_data.values():
            args = json.loads(tc_data["function"]["arguments"]) if tc_data["function"]["arguments"] else {}
            result = registry.execute(tc_data["function"]["name"], args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc_data["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })

    return
```

> 📄 最新代码：[code/04-tool-calling/src/scientex_agent/agent_loop.py](../../code/04-tool-calling/src/scientex_agent/agent_loop.py)

#### 代码详解

**1. `_execute_tool_calls` 函数**

这个函数负责批量执行 LLM 返回的所有工具调用：

```python
for tc in response.choices[0].message.tool_calls or []:
```

- `response.choices[0].message.tool_calls` 是 LLM 返回的工具调用列表
- `or []` 是为了防止 `tool_calls` 为 `None` 时报错（如果 LLM 没有调用工具）

```python
args = json.loads(tc.function.arguments) if tc.function.arguments else {}
```

- LLM 返回的参数是 JSON 字符串（如 `'{"expression": "2+3"}'`）
- `json.loads()` 把它解析成 Python 字典（如 `{"expression": "2+3"}`）

**2. `chat_with_tools` 函数的核心循环**

```python
for _ in range(max_rounds):
    response = client.chat.completions.create(...)
  
    if not msg.tool_calls:
        return msg.content  # LLM 没有调用工具 → 返回最终回复
  
    # 有工具调用 → 执行工具 → 把结果加入 messages → 继续循环
    messages.append(assistant_msg_with_tool_calls)
    messages.extend(tool_results)
```

这个循环实现了：

1. 发送消息给 LLM
2. 如果 LLM 返回文本（没有调用工具）→ 结束
3. 如果 LLM 调用工具 → 执行工具 → 把结果追加到消息历史 → 再次调用 LLM
4. 最多循环 `max_rounds` 次（防止无限循环）

**3. 函数签名中的 `*` 和 `**kwargs`**

```python
def chat_with_tools(
    client: OpenAI,
    model: str,
    messages: list[dict],
    registry: ToolRegistry,
    *,                    # ← 这个星号
    max_rounds: int = 10,
    **kwargs,             # ← 这个双星号
) -> str:
```

| 符号         | 含义                                        |
| ------------ | ------------------------------------------- |
| `*`        | 后面的参数必须用关键字传参（不能按位置传）  |
| `**kwargs` | 接收任意额外的关键字参数，会透传给 API 调用 |

```python
# 调用示例
chat_with_tools(
    client, "gpt-4", messages, registry,
    max_rounds=5,           # 必须用关键字（因为有 *）
    temperature=0.7,        # 通过 **kwargs 传给 API
)
```

**4. 流式工具调用的难点：聚合 chunk**

流式模式下，一个工具调用的数据会分散在多个 chunk 中：

```
chunk 1: tool_calls[0].function.name = "get_"
chunk 2: tool_calls[0].function.name = "current_"
chunk 3: tool_calls[0].function.name = "time"
chunk 4: tool_calls[0].function.arguments = '{"timezone":"A'
chunk 5: tool_calls[0].function.arguments = 'sia/Shanghai"}'
```

代码用 `tool_calls_data` 字典来聚合：

```python
tool_calls_data: dict[int, dict] = {}

for chunk in stream:
    if delta.tool_calls:
        for tc_delta in delta.tool_calls:
            idx = tc_delta.index  # 第几个工具调用
        
            # 第一次见到这个 index，初始化
            if idx not in tool_calls_data:
                tool_calls_data[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        
            # 累加数据（字符串拼接）
            if tc_delta.function.name:
                tool_calls_data[idx]["function"]["name"] += tc_delta.function.name
            if tc_delta.function.arguments:
                tool_calls_data[idx]["function"]["arguments"] += tc_delta.function.arguments
```

最终 `tool_calls_data` 会变成：

```python
{
    0: {
        "id": "call_abc123",
        "type": "function",
        "function": {
            "name": "get_current_time",
            "arguments": '{"timezone":"Asia/Shanghai"}'
        }
    }
}
```

**5. `Generator[str, None, None]` 类型注解**

```python
def chat_with_tools_stream(...) -> Generator[str, None, None]:
```

这是生成器函数的返回类型：

- 第一个 `str`：`yield` 产出的值的类型
- 第二个 `None`：`send()` 发送的值的类型（通常为 None）
- 第三个 `None`：`return` 返回的值的类型（通常为 None）

### 3. 更新 ChatSession（统一管理对话）

为了让 `cli.py` 不需要直接和 `agent_loop` 打交道，我们在 `ChatSession` 中封装了工具调用方法。这样所有对话逻辑（普通对话、流式对话、工具调用）都由 `ChatSession` 统一管理。

```python
# 在 llm_client.py 中更新 ChatSession：

class ChatSession:
    # ... 之前的 send(), send_stream() 方法 ...

    def __init__(self, model: str | None = None, client: OpenAI | None = None) -> None:
        self.client = client or get_client()
        self.model = model or get_default_model()
        self.registry = registry_default_tools()
        self.messages: list[dict[str, str]] = []

    def send_with_tools(
        self,
        content: str,
        **kwargs,
    ) -> str:
        """Send a message with tool calling support."""
        from .agent_loop import chat_with_tools

        self.messages.append({"role": "user", "content": content})

        reply = chat_with_tools(
            self.client, self.model, self.messages, self.registry, **kwargs
        )

        # Check if reply is already in messages (from tool loop)
        if not any(
            m.get("role") == "assistant" and m.get("content") == reply
            for m in self.messages
        ):
            self.messages.append({"role": "assistant", "content": reply})

        return reply or ""

    def send_with_tools_stream(
        self,
        content: str,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Streaming version with tool calling."""
        from .agent_loop import chat_with_tools_stream

        self.messages.append({"role": "user", "content": content})

        full_reply = ""
        for token in chat_with_tools_stream(
            self.client, self.model, self.messages, self.registry, **kwargs
        ):
            full_reply += token
            yield token

        # Check and append if needed
        if not any(
            m.get("role") == "assistant" and m.get("content") == full_reply
            for m in self.messages
        ):
            self.messages.append({"role": "assistant", "content": full_reply})
```

> 📄 最新代码：[code/04-tool-calling/src/scientex_agent/llm_client.py](../../code/04-tool-calling/src/scientex_agent/llm_client.py)

#### 架构说明

```
cli.py（只和 ChatSession 交互）
  │
  └── ChatSession（统一管理对话）
        ├── send()                 # 普通对话
        ├── send_stream()          # 流式对话
        ├── send_with_tools()      # 工具调用 ← 本章新增
        └── send_with_tools_stream()  # 流式工具调用 ← 本章新增
              │
              └── agent_loop.py（内部实现，cli.py 不直接感知）
                    ├── chat_with_tools()
                    └── chat_with_tools_stream()
```

**好处：**

- `cli.py` 只需要导入 `ChatSession`，不需要知道 `agent_loop` 的存在
- 工具调用和普通对话的接口一致（都通过 `session` 对象）
- 未来添加新功能（如 MCP）只需扩展 `ChatSession`

## 验证

工具在 `ChatSession.__init__` 里通过 `registry_default_tools()` 自动注册，CLI 不需要额外开关：

### 1. 通过 CLI 交互式模式测试

```bash
uv run scientex_agent chat --interactive
```

进入后输入：

```
> 3 的 10 次方是多少？
```

预期：LLM 调用 `caculate("3**10")` → 得到 59049 → 回复包含结果

```
> 现在北京时间几点？
```

预期：LLM 调用 `get_current_time("Asia/Shanghai")` → 回复当前时间

```
> /exit
```

### 2. 指定 system prompt

```bash
uv run scientex_agent chat --interactive \
  --system "你是一个助手。可以用 caculate 工具计算，用 get_current_time 获取时间。"
```

### 3. 关闭流式输出

```bash
uv run scientex_agent chat --interactive --no-stream
```

## 深入理解

### 工具描述的编写技巧

```python
# ❌ 不好：太模糊
"description": "获取时间"

# ✅ 好：明确、有例子
"description": "Get the current date and time in a given timezone. "
               "Use IANA timezone names like 'Asia/Shanghai' or 'America/New_York'."

# ❌ 不好：参数太宽泛
"parameters": {"type": "object", "properties": {"query": {"type": "string"}}}

# ✅ 好：参数有约束和描述
"parameters": {
    "type": "object",
    "properties": {
        "timezone": {
            "type": "string",
            "description": "IANA timezone name (e.g. 'Asia/Shanghai')",
            "enum": ["Asia/Shanghai", "America/New_York", "Europe/London", ...]
        }
    },
    "required": ["timezone"]
}
```

### OpenAI SDK 支持并行工具调用

当 LLM 需要同时调用多个独立工具时（如同时查北京和纽约的时间），它会在一次响应中返回多个 `tool_calls`。SDK 原生支持这一点。

### 流式工具调用的特殊处理

```
chunk 1:  delta.tool_calls = [{"index": 0, "id": "call_abc", "function": {"name": "get_"}}]
chunk 2:  delta.tool_calls = [{"index": 0, "function": {"name": "current_"}}]
chunk 3:  delta.tool_calls = [{"index": 0, "function": {"name": "time"}}]
chunk 4:  delta.tool_calls = [{"index": 0, "function": {"arguments": '{"timezone":"A'}}]
chunk 5:  delta.tool_calls = [{"index": 0, "function": {"arguments": 'sia/Shanghai"}'}}]
```

需要通过 `index` 来聚合不同 chunk 中的 tool call 数据。这就是 `chat_with_tools_stream` 中 `tool_calls_data` 字典的作用。

## 当前局限

1. 工具执行是同步的——慢工具会阻塞对话
2. 工具结果原样返回给 LLM——没有结果过滤/截断
3. 只能调用 OpenAI 兼容 API
4. 工具注册和会话状态紧密耦合

## 下一步

→ [05-multi-provider.md](05-multi-provider.md)

## 参考资料

- [OpenAI Function Calling 官方文档](https://platform.openai.com/docs/guides/function-calling)
- [DeepSeek Tool Calls 官方文档](https://api-docs.deepseek.com/guides/tool_calls)
