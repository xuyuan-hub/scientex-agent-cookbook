"""Adapter for Anthropic's Messages API.

The optional ``anthropic`` package is imported only when this adapter is used,
so DeepSeek/OpenAI learners do not need to install it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from ..provider_types import ChatMessage, ChatRequest, ChatResponse, ModelSpec, StreamEvent, ToolCall, Usage


class AnthropicProvider:
    """Translate normalized requests to the Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, api_key: str, models: Sequence[ModelSpec]) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic support needs the optional SDK. Run `uv add anthropic`."
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._api_key = api_key
        self._models = tuple(models)

    def list_models(self) -> Sequence[ModelSpec]:
        return self._models

    def chat(self, request: ChatRequest) -> ChatResponse:
        return _parse_response(self._client.messages.create(**_request_kwargs(request)))

    def chat_stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        with self._client.messages.stream(**_request_kwargs(request)) as stream:
            for text in stream.text_stream:
                yield StreamEvent(text=text)
            yield StreamEvent(response=_parse_response(stream.get_final_message()))

    def to_langchain_chat_model(self, model: str, *, streaming: bool) -> Any:
        """Create the Step 10 LangChain bridge for this provider."""

        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic LangGraph support needs `uv add langchain-anthropic`."
            ) from exc
        return ChatAnthropic(model=model, api_key=self._api_key, streaming=streaming)


def _request_kwargs(request: ChatRequest) -> dict[str, Any]:
    messages = [message for message in request.messages if message.role != "system"]
    kwargs: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens or 4096,
        "messages": _to_anthropic_messages(messages),
    }
    system_text = "\n\n".join(message.content for message in request.messages if message.role == "system")
    if system_text:
        kwargs["system"] = system_text
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.tools:
        kwargs["tools"] = _to_anthropic_tools(request.tools)
    return kwargs


def _to_anthropic_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id or "",
                "content": message.content,
            }
            if result and result[-1]["role"] == "user":
                previous = result[-1]["content"]
                result[-1]["content"] = previous + [block] if isinstance(previous, list) else [block]
            else:
                result.append({"role": "user", "content": [block]})
            continue

        if message.role == "assistant" and message.tool_calls:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
                for call in message.tool_calls
            )
            result.append({"role": "assistant", "content": blocks})
            continue

        result.append({"role": message.role, "content": message.content})
    return result


def _to_anthropic_tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function", {})
        result.append(
            {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return result


def _parse_response(response: Any) -> ChatResponse:
    content = ""
    calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            content += block.text
        elif block.type == "tool_use":
            calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
    usage = response.usage
    return ChatResponse(
        content=content,
        tool_calls=tuple(calls),
        usage=Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        ),
        stop_reason=response.stop_reason,
        raw={"id": response.id, "model": response.model},
    )
