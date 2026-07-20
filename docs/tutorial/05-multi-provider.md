# Step 05: 多提供商支持

## 目标

支持 Anthropic、Gemini、Azure OpenAI 等多种 LLM 提供商，统一接口。

## 前置条件

- 完成 [04-tool-calling.md](04-tool-calling.md)

## 设计思路

### 问题：每个提供商的 API 都不同

```
OpenAI:                 Anthropic:                 Gemini:
POST /v1/chat/          POST /v1/messages          POST /models/{m}:generateContent
Body: {                 Body: {                    Body: {
  "model": "gpt-4o",      "model": "claude-5",       (model in URL)
  "messages": [           "messages": [              "contents": [
    {"role":"user",         {"role":"user",            {"role":"user",
     "content":"Hi"}         "content":"Hi"}            "parts":[{"text":"Hi"}]}
  ],                      ],                        ],
  "tools": [{             "tools": [{               "tools": [{
    "type":"function",      "name":"calc",              "functionDeclarations":[{
    "function":{...}        "input_schema":{...}          "name":"calc",...}]}
  }]                      }]                        }]
}                        }                          }
```

需要定义**统一的内部接口**，然后每个提供商做一个**适配器**。

### 统一接口

```python
class LLMProvider(Protocol):
    """所有 LLM 提供商的统一接口。"""

    name: str                           # "openai", "deepseek", "anthropic" 等

    def list_models(self) -> list[ModelSpec]: ...
    def chat(self, request: ChatRequest) -> ChatResponse: ...
```

所有业务代码只依赖 `LLMProvider` 接口，不关心背后的提供商是谁。

### ChatRequest / ChatResponse（内部统一格式）

```python
@dataclass
class ChatRequest:
    messages: list[ChatMessage]      # 统一的消息格式
    model: str
    system: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict] = field(default_factory=list)

@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None
    stop_reason: str | None = None

@dataclass
class ChatMessage:
    role: str                        # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = None  # for tool messages
    tool_calls: list[ToolCall] = field(default_factory=list)  # for assistant

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

@dataclass
class ModelSpec:
    provider: str
    model: str
    context_window: int
    supports_tools: bool = False
    supports_streaming: bool = True
```

### 架构图

```
                    ┌──────────────────┐
                    │   ChatSession    │ (业务代码)
                    │   AgentLoop      │
                    └────────┬─────────┘
                             │ 依赖 LLMProvider (Protocol)
                    ┌────────┴─────────┐
                    │  ProviderRegistry │
                    └────────┬─────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
   ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
   │ OpenAICompat│   │  Anthropic  │   │   Gemini    │
   │  Provider   │   │  Provider   │   │  Provider   │
   │             │   │             │   │             │
   │ OpenAI SDK  │   │ HTTP+SDK   │   │ HTTP+SDK   │
   └─────────────┘   └─────────────┘   └─────────────┘
```

## 实现

### 1. 创建 src/scientex_agent/provider_types.py

```python
"""Provider-neutral request/response types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """Model metadata."""
    provider: str          # "openai", "anthropic", "deepseek", "gemini"
    model: str             # "gpt-4o", "claude-sonnet-5", etc.
    context_window: int    # max tokens (input + output)
    supports_tools: bool = False
    supports_streaming: bool = True
    supports_json_mode: bool = False
    input_cost_per_mtok: float | None = None   # USD per million input tokens
    output_cost_per_mtok: float | None = None  # USD per million output tokens


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list["ToolCall"] = field(default_factory=list)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatRequest:
    messages: list[ChatMessage]
    model: str
    system: str | None = None
    system_stable: str | None = None    # for Anthropic cache-control
    system_dynamic: str | None = None   # for Anthropic cache-control
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True)
class ChatResponse:
    content: str
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
```

### 2. 创建 src/scientex_agent/provider_registry.py

```python
"""Provider registry and interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .provider_types import ChatRequest, ChatResponse, ModelSpec


@runtime_checkable
class LLMProvider(Protocol):
    """Every provider must implement this."""

    name: str

    def list_models(self) -> Sequence[ModelSpec]: ...
    def chat(self, request: ChatRequest) -> ChatResponse: ...


class ProviderRegistry:
    """Registry of named LLM providers."""

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider) -> None:
        if provider.name in self._providers:
            raise ValueError(f"Provider already registered: {provider.name}")
        self._providers[provider.name] = provider

    def get(self, name: str) -> LLMProvider:
        if name not in self._providers:
            raise KeyError(f"Unknown provider: {name}. "
                           f"Available: {list(self._providers)}")
        return self._providers[name]

    def list_provider_names(self) -> list[str]:
        return list(self._providers)

    def list_models(self) -> list[ModelSpec]:
        result = []
        for provider in self._providers.values():
            result.extend(provider.list_models())
        return result

    def get_model_spec(self, provider_name: str, model_name: str) -> ModelSpec | None:
        provider = self._providers.get(provider_name)
        if provider is None:
            return None
        for spec in provider.list_models():
            if spec.model == model_name:
                return spec
        return None
```

### 3. 创建 src/scientex_agent/providers/openai_compatible.py

```python
"""OpenAI-compatible provider (covers OpenAI, DeepSeek, OpenRouter, Ollama, etc.)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from openai import OpenAI

from ..provider_types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ModelSpec,
    ToolCall,
    Usage,
)
from ..provider_registry import LLMProvider


class OpenAICompatibleProvider:
    """Provider for any OpenAI-compatible API endpoint."""

    name: str
    _client: OpenAI
    _models: list[ModelSpec]

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        models: list[ModelSpec],
        *,
        timeout: float = 120.0,
    ) -> None:
        self.name = name
        self._models = models
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def list_models(self) -> Sequence[ModelSpec]:
        return self._models

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request and return the response."""
        messages = _to_openai_messages(request)

        response = self._client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=request.tools or None,
        )

        choice = response.choices[0]
        content = choice.message.content or ""

        # Parse tool calls
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
            ),
            stop_reason=choice.finish_reason,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )


def _to_openai_messages(request: ChatRequest) -> list[dict]:
    """Convert internal messages to OpenAI format."""
    messages: list[dict[str, Any]] = []

    if request.system:
        messages.append({"role": "system", "content": request.system})

    for msg in request.messages:
        payload: dict[str, Any] = {"role": msg.role, "content": msg.content}
        if msg.tool_call_id:
            payload["tool_call_id"] = msg.tool_call_id
        if msg.name:
            payload["name"] = msg.name
        if msg.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(payload)

    return messages
```

### 4. 创建 src/scientex_agent/providers/anthropic.py

```python
"""Anthropic provider adapter.

Uses the Anthropic HTTP API directly (minimal dependency footprint).
For production, consider `pip install anthropic` for the official SDK.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any
from urllib import request as urllib_request
from urllib.error import URLError

from ..provider_types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ModelSpec,
    ToolCall,
    Usage,
)
from ..provider_registry import LLMProvider


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        models: list[ModelSpec] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._models = models or [
            ModelSpec("anthropic", "claude-sonnet-5", 200000, supports_tools=True),
            ModelSpec("anthropic", "claude-opus-4-8", 200000, supports_tools=True),
        ]
        self._timeout = timeout

    def list_models(self) -> Sequence[ModelSpec]:
        return self._models

    def chat(self, request: ChatRequest) -> ChatResponse:
        url = f"{self._base_url}/v1/messages"
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        body = self._build_body(request)
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        req = urllib_request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib_request.urlopen(req, timeout=self._timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except URLError as e:
            raise RuntimeError(f"Anthropic API error: {e}") from e

        return self._parse_response(result)

    def _build_body(self, request: ChatRequest) -> dict:
        body: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens or 4096,
            "messages": self._convert_messages(request.messages),
        }

        # System prompt
        system_parts = []
        if request.system_stable:
            system_parts.append({
                "type": "text",
                "text": request.system_stable,
                "cache_control": {"type": "ephemeral"},
            })
        if request.system_dynamic:
            system_parts.append({"type": "text", "text": request.system_dynamic})
        if request.system and not system_parts:
            system_parts.append({"type": "text", "text": request.system})
        if system_parts:
            body["system"] = system_parts

        # Tools
        if request.tools:
            body["tools"] = self._convert_tools(request.tools)

        if request.temperature is not None:
            body["temperature"] = request.temperature

        return body

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert to Anthropic messages format.

        Anthropic requires strict user/assistant alternation.
        Tool results are merged into the next user turn.
        """
        result: list[dict] = []
        for msg in messages:
            if msg.role == "system":
                continue  # handled separately

            if msg.role == "user":
                result.append({"role": "user", "content": msg.content})

            elif msg.role == "assistant":
                content: Any = msg.content or ""
                if msg.tool_calls:
                    content = []
                    if msg.content:
                        content.append({"type": "text", "text": msg.content})
                    for tc in msg.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })
                result.append({"role": "assistant", "content": content})

            elif msg.role == "tool":
                # Merge into the next user message or create one
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "",
                    "content": msg.content,
                }
                if result and result[-1]["role"] == "user":
                    if isinstance(result[-1]["content"], list):
                        result[-1]["content"].append(tool_result)
                    else:
                        result[-1]["content"] = [tool_result]
                else:
                    result.append({"role": "user", "content": [tool_result]})

        return result

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI-style tools to Anthropic format."""
        result = []
        for tool in tools:
            func = tool.get("function", {})
            result.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
        return result

    def _parse_response(self, raw: dict) -> ChatResponse:
        """Parse Anthropic response into ChatResponse."""
        content_text = ""
        tool_calls = []

        for block in raw.get("content", []):
            if block["type"] == "text":
                content_text += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block.get("input", {}),
                ))

        usage_data = raw.get("usage", {})
        return ChatResponse(
            content=content_text,
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
                cache_read_tokens=usage_data.get("cache_read_input_tokens", 0),
                cache_write_tokens=usage_data.get("cache_creation_input_tokens", 0),
            ),
            stop_reason=raw.get("stop_reason"),
            raw=raw,
        )
```

### 5. 创建工厂函数

```python
# src/scientex_agent/providers/__init__.py

from ..provider_types import ModelSpec

# 常用模型定义
ANTHROPIC_MODELS = [
    ModelSpec("anthropic", "claude-sonnet-5", 200000, supports_tools=True, supports_streaming=False,
              input_cost_per_mtok=3.0, output_cost_per_mtok=15.0),
    ModelSpec("anthropic", "claude-haiku-4-5", 200000, supports_tools=True, supports_streaming=False,
              input_cost_per_mtok=0.80, output_cost_per_mtok=4.0),
]

OPENAI_MODELS = [
    ModelSpec("openai", "gpt-4o", 128000, supports_tools=True, supports_streaming=True,
              input_cost_per_mtok=2.50, output_cost_per_mtok=10.0),
    ModelSpec("openai", "gpt-4o-mini", 128000, supports_tools=True, supports_streaming=True,
              input_cost_per_mtok=0.15, output_cost_per_mtok=0.60),
]

DEEPSEEK_MODELS = [
    ModelSpec("deepseek", "deepseek-chat", 128000, supports_tools=True, supports_streaming=True,
              input_cost_per_mtok=0.27, output_cost_per_mtok=1.10),
]
```

## 验证

```python
from scientex_agent.providers.openai_compatible import OpenAICompatibleProvider
from scientex_agent.providers import DEEPSEEK_MODELS
from scientex_agent.provider_types import ChatRequest, ChatMessage

# DeepSeek provider
provider = OpenAICompatibleProvider(
    name="deepseek",
    base_url="https://api.deepseek.com/v1",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    models=DEEPSEEK_MODELS,
)

# 通过统一接口调用
response = provider.chat(ChatRequest(
    messages=[ChatMessage(role="user", content="Hello!")],
    model="deepseek-chat",
))
print(response.content)
```

## 当前局限

1. Anthropic 和 Gemini 目前不支持流式
2. Gemini provider 还未实现（结构与 Anthropic 类似）
3. 每个 provider 的错误处理格式不统一

## 下一步

→ [06-conversation-persistence.md](06-conversation-persistence.md)
