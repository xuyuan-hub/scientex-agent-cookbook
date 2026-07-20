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
        "name": "get_current_time",
        "description": "Get the current time in a given timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name, e.g. 'Asia/Shanghai'"
                }
            },
            "required": ["timezone"]
        }
    }
}

# 2. LLM 返回的 tool call（告诉我们要执行什么）
# response.choices[0].message.tool_calls = [
#     ToolCall(id="call_123", name="get_current_time", arguments='{"timezone":"Asia/Shanghai"}')
# ]

# 3. 我们把执行结果回传给 LLM（tool role message）
# {"role": "tool", "tool_call_id": "call_123", "content": "2026-07-20T15:30:00+08:00"}
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

def register_default_tools(registry: ToolRegistry) -> None:
    """Register a set of basic utility tools."""

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
    def get_current_time(timezone: str) -> str:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        try:
            tz = ZoneInfo(timezone)
            now = datetime.now(tz)
            return now.strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            # Fallback to UTC
            return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    @registry.register(
        name="calculate",
        description="Evaluate a mathematical expression. Supports +, -, *, /, **, and common math functions.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Python math expression, e.g. '2 + 3 * 4' or 'sqrt(16)'"
                }
            },
            "required": ["expression"],
        },
    )
    def calculate(expression: str) -> float | str:
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
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return result
        except Exception as e:
            return f"Error: {e}"
```

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

### 3. 更新 ChatSession

```python
# 在 llm_client.py 中更新 ChatSession：

class ChatSession:
    # ... 之前的代码 ...

    def send_with_tools(
        self,
        content: str,
        registry: "ToolRegistry",
        **kwargs,
    ) -> str:
        """Send a message with tool calling support."""
        from .agent_loop import chat_with_tools

        self.messages.append({"role": "user", "content": content})

        reply = chat_with_tools(
            self.client, self.model, self.messages, registry, **kwargs
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
        registry: "ToolRegistry",
        **kwargs,
    ) -> Generator[str, None, None]:
        """Streaming version with tool calling."""
        from .agent_loop import chat_with_tools_stream

        self.messages.append({"role": "user", "content": content})

        full_reply = ""
        for token in chat_with_tools_stream(
            self.client, self.model, self.messages, registry, **kwargs
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

## 验证

```python
from scientex_agent.llm_client import ChatSession
from scientex_agent.tools import ToolRegistry, register_default_tools

# Setup
registry = ToolRegistry()
register_default_tools(registry)

session = ChatSession()
session.system("你是一个助手。可以用 calculate 工具计算，用 get_current_time 获取时间。")

# 测试工具调用
reply = session.send_with_tools("3 的 10 次方是多少？", registry)
print(reply)
# 预期：LLM 调用 calculate("3**10") → 结果 59049 → 回复包含 59049

reply = session.send_with_tools("现在北京时间几点？", registry)
print(reply)
# 预期：LLM 调用 get_current_time("Asia/Shanghai") → 回复当前时间
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
