"""Adapter for OpenAI-compatible chat-completions APIs.

The same adapter works for OpenAI, DeepSeek, OpenRouter, Ollama, vLLM, and
other services that implement the OpenAI chat-completions wire format.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

from openai import OpenAI

from ..provider_types import (
    ChatRequest,
    ChatResponse,
    ModelSpec,
    StreamEvent,
    ToolCall,
    Usage,
)


class OpenAICompatibleProvider:
    """Translate normalized requests to an OpenAI-compatible SDK call."""

    def __init__(
        self,
        name: str,
        base_url: str | None,
        api_key: str | None,
        models: Sequence[ModelSpec],
        *,
        timeout: float = 120.0,
        client: Any | None = None,
    ) -> None:
        self.name = name
        self._models = tuple(models)
        self._api_key = api_key
        self._base_url = base_url
        self._client = client or OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def list_models(self) -> Sequence[ModelSpec]:
        return self._models

    def chat(self, request: ChatRequest) -> ChatResponse:
        response = self._client.chat.completions.create(**_request_kwargs(request))
        return _parse_response(response)

    def chat_stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        """Yield text fragments followed by one final normalized response."""

        stream = self._client.chat.completions.create(
            **_request_kwargs(request),
            stream=True,
            stream_options={"include_usage": True},
        )
        content = ""
        tool_calls: dict[int, dict[str, Any]] = {}
        usage = Usage()
        stop_reason: str | None = None

        for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = Usage(
                    input_tokens=getattr(chunk_usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(chunk_usage, "completion_tokens", 0) or 0,
                )

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            stop_reason = getattr(choice, "finish_reason", None) or stop_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            text = getattr(delta, "content", None)
            if text:
                content += text
                yield StreamEvent(text=text)

            for delta_call in getattr(delta, "tool_calls", None) or []:
                index = getattr(delta_call, "index", 0)
                partial = tool_calls.setdefault(
                    index,
                    {"id": "", "name": "", "arguments": ""},
                )
                call_id = getattr(delta_call, "id", None)
                if call_id:
                    partial["id"] = call_id
                function = getattr(delta_call, "function", None)
                if function is not None:
                    name = getattr(function, "name", None)
                    arguments = getattr(function, "arguments", None)
                    if name:
                        partial["name"] += name
                    if arguments:
                        partial["arguments"] += arguments

        yield StreamEvent(
            response=ChatResponse(
                content=content,
                usage=usage,
                tool_calls=tuple(_tool_call_from_partial(item) for item in tool_calls.values()),
                stop_reason=stop_reason,
            )
        )

    def to_langchain_chat_model(self, model: str, *, streaming: bool) -> Any:
        """Create the Step 10 LangChain bridge without leaking config upward."""

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=self._api_key,
            base_url=self._base_url,
            streaming=streaming,
        )


def _request_kwargs(request: ChatRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": request.model,
        "messages": _to_openai_messages(request),
    }
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.tools:
        kwargs["tools"] = list(request.tools)
    return kwargs


def _to_openai_messages(request: ChatRequest) -> list[dict[str, Any]]:
    """Convert normalized history to OpenAI's wire format."""

    messages: list[dict[str, Any]] = []
    for message in request.messages:
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.name:
            payload["name"] = message.name
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
        messages.append(payload)
    return messages


def _parse_response(response: Any) -> ChatResponse:
    choice = response.choices[0]
    message = choice.message
    calls: list[ToolCall] = []
    for call in getattr(message, "tool_calls", None) or []:
        raw_arguments = call.function.arguments or "{}"
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {}
        calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))

    usage_data = getattr(response, "usage", None)
    return ChatResponse(
        content=getattr(message, "content", None) or "",
        tool_calls=tuple(calls),
        usage=Usage(
            input_tokens=getattr(usage_data, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage_data, "completion_tokens", 0) or 0,
        ),
        stop_reason=getattr(choice, "finish_reason", None),
        raw=response.model_dump() if hasattr(response, "model_dump") else {},
    )


def _tool_call_from_partial(partial: dict[str, Any]) -> ToolCall:
    try:
        arguments = json.loads(partial["arguments"] or "{}")
    except json.JSONDecodeError:
        arguments = {}
    return ToolCall(id=partial["id"], name=partial["name"], arguments=arguments)
