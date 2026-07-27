"""Provider-neutral tool-calling loops used by :class:`ChatSession`."""

from __future__ import annotations

import json
from collections.abc import Generator, Iterable
from typing import Any

from .provider_registry import LLMProvider, StreamingLLMProvider
from .provider_types import ChatRequest, ChatResponse, message_from_dict, response_message
from .tools import ToolRegistry


def chat_with_tools(
    provider: LLMProvider,
    model: str,
    messages: list[dict[str, Any]],
    registry: ToolRegistry,
    *,
    max_rounds: int = 10,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Run the same tool loop from Step 04 through a provider adapter."""

    for _ in range(max_rounds):
        response = provider.chat(
            _request(messages, model, registry, temperature=temperature, max_tokens=max_tokens)
        )
        messages.append(response_message(response))
        if not response.tool_calls:
            return response.content
        _append_tool_results(messages, response, registry)
    raise RuntimeError(f"Tool loop exceeded {max_rounds} rounds")


def chat_with_tools_stream(
    provider: LLMProvider,
    model: str,
    messages: list[dict[str, Any]],
    registry: ToolRegistry,
    *,
    max_rounds: int = 10,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Generator[str, None, None]:
    """Stream text while still processing provider-neutral tool calls."""

    for _ in range(max_rounds):
        request = _request(messages, model, registry, temperature=temperature, max_tokens=max_tokens)
        if isinstance(provider, StreamingLLMProvider):
            response = yield from _collect_stream(provider.chat_stream(request))
        else:
            # A custom provider can implement the minimal protocol and still
            # work with Step 04.  It simply returns one complete text chunk.
            response = provider.chat(request)
            if response.content:
                yield response.content

        messages.append(response_message(response))
        if not response.tool_calls:
            return
        _append_tool_results(messages, response, registry)
    raise RuntimeError(f"Tool loop exceeded {max_rounds} rounds")


def _collect_stream(events: Iterable) -> Generator[str, None, ChatResponse]:
    final_response: ChatResponse | None = None
    for event in events:
        if event.text:
            yield event.text
        if event.response is not None:
            final_response = event.response
    if final_response is None:
        raise RuntimeError("Provider stream ended without a final response")
    return final_response


def _request(
    messages: list[dict[str, Any]],
    model: str,
    registry: ToolRegistry,
    *,
    temperature: float | None,
    max_tokens: int | None,
) -> ChatRequest:
    return ChatRequest(
        messages=tuple(message_from_dict(message) for message in messages),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tuple(registry.schemas()),
    )


def _append_tool_results(
    messages: list[dict[str, Any]],
    response: ChatResponse,
    registry: ToolRegistry,
) -> None:
    for call in response.tool_calls:
        result = registry.execute(call.name, call.arguments)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
