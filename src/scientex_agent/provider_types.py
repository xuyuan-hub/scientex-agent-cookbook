"""Provider-neutral messages, requests, and responses.

These types form the boundary between the conversation/agent code and a
provider adapter.  Application code must not depend on an SDK response type.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """A model that can be selected from a provider."""

    provider: str
    model: str
    context_window: int
    supports_tools: bool = False
    supports_streaming: bool = True
    supports_json_mode: bool = False
    input_cost_per_mtok: float | None = None
    output_cost_per_mtok: float | None = None


@dataclass(frozen=True)
class ToolCall:
    """A provider-neutral request to invoke one registered tool."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatMessage:
    """A provider-neutral conversation message."""

    role: str
    content: str = ""
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ChatRequest:
    """A normalized request passed from the agent to a provider."""

    messages: tuple[ChatMessage, ...]
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    # ToolRegistry currently produces OpenAI-shaped schemas.  Each adapter
    # converts that schema at its boundary (Anthropic does so explicitly).
    tools: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
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
    """A normalized final response from a provider."""

    content: str = ""
    usage: Usage = field(default_factory=Usage)
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StreamEvent:
    """One event from a provider stream.

    Text events carry a token fragment.  The final event carries the complete
    normalized response, including any tool calls accumulated by the adapter.
    """

    text: str = ""
    response: ChatResponse | None = None


def message_from_dict(message: Mapping[str, Any]) -> ChatMessage:
    """Convert the legacy dictionary history used by earlier steps."""

    tool_calls: list[ToolCall] = []
    for item in message.get("tool_calls") or []:
        function = item.get("function", {}) if isinstance(item, Mapping) else {}
        raw_arguments = function.get("arguments", {})
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}
        elif isinstance(raw_arguments, Mapping):
            arguments = dict(raw_arguments)
        else:
            arguments = {}
        tool_calls.append(
            ToolCall(
                id=str(item.get("id", "")),
                name=str(function.get("name", "")),
                arguments=arguments,
            )
        )

    content = message.get("content", "")
    return ChatMessage(
        role=str(message["role"]),
        content="" if content is None else str(content),
        tool_call_id=message.get("tool_call_id"),
        name=message.get("name"),
        tool_calls=tuple(tool_calls),
    )


def response_message(response: ChatResponse) -> dict[str, Any]:
    """Convert a normalized response back to the legacy history format."""

    message: dict[str, Any] = {"role": "assistant", "content": response.content}
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in response.tool_calls
        ]
    return message
