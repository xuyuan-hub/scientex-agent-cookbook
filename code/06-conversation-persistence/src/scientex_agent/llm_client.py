"""Provider-backed chat sessions with history, tools, and streaming."""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any

from openai import OpenAI

from .provider_registry import LLMProvider, StreamingLLMProvider
from .provider_types import ChatRequest, message_from_dict, response_message
from .tools import ToolRegistry, registry_default_tools


def get_client() -> OpenAI:
    """Compatibility helper retained for the earlier tutorial steps."""

    if os.environ.get("DEEPSEEK_API_KEY"):
        return OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    raise RuntimeError("No LLM API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY.")


def get_default_model() -> str:
    """Return the selected provider's configured default model name."""

    return (
        os.environ.get("DEEPSEEK_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("ANTHROPIC_MODEL")
        or "deepseek-v4-pro"
    )


class ChatSession:
    """A conversation session that depends on ``LLMProvider``, not an SDK.

    The public ``messages`` history deliberately remains a list of dictionaries
    so Steps 02, 04, and 06 can keep using the same persistence format.
    """

    def __init__(
        self,
        model: str | None = None,
        client: OpenAI | None = None,
        provider: LLMProvider | None = None,
        provider_name: str | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.provider = provider or _provider_from_legacy_client(client, provider_name)
        self.model = model or _default_model_for(self.provider)
        self.registry = registry or registry_default_tools()
        self.messages: list[dict[str, Any]] = []

    def system(self, content: str) -> None:
        """Set or replace the system prompt at the beginning of history."""

        message = {"role": "system", "content": content}
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0] = message
        else:
            self.messages.insert(0, message)

    def send(
        self,
        content: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send one normal chat message through the selected provider."""

        self.messages.append({"role": "user", "content": content})
        response = self.provider.chat(
            self._request(temperature=temperature, max_tokens=max_tokens)
        )
        self.messages.append(response_message(response))
        return response.content

    def send_stream(
        self,
        content: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """Stream a normal response, with a safe one-chunk fallback."""

        self.messages.append({"role": "user", "content": content})
        request = self._request(temperature=temperature, max_tokens=max_tokens)
        if isinstance(self.provider, StreamingLLMProvider):
            final = None
            for event in self.provider.chat_stream(request):
                if event.text:
                    yield event.text
                if event.response is not None:
                    final = event.response
            if final is None:
                raise RuntimeError("Provider stream ended without a final response")
        else:
            final = self.provider.chat(request)
            if final.content:
                yield final.content
        self.messages.append(response_message(final))

    def send_with_tools(
        self,
        content: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Run Step 04's tool loop through the selected provider adapter."""

        from .agent_loop import chat_with_tools

        self.messages.append({"role": "user", "content": content})
        return chat_with_tools(
            self.provider,
            self.model,
            self.messages,
            self.registry,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def send_with_tools_stream(
        self,
        content: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """Stream Step 04's tool loop through the selected provider adapter."""

        from .agent_loop import chat_with_tools_stream

        self.messages.append({"role": "user", "content": content})
        yield from chat_with_tools_stream(
            self.provider,
            self.model,
            self.messages,
            self.registry,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def last_user_message(self) -> dict[str, Any] | None:
        for message in reversed(self.messages):
            if message["role"] == "user":
                return message
        return None

    def clear(self) -> None:
        """Clear history while preserving the system prompt, if set."""

        system_message = self.messages[0] if self.messages and self.messages[0]["role"] == "system" else None
        self.messages = [system_message] if system_message else []

    def _request(
        self,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatRequest:
        return ChatRequest(
            messages=tuple(message_from_dict(message) for message in self.messages),
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def chat(
    prompt: str,
    *,
    model: str | None = None,
    provider_name: str | None = None,
) -> str:
    """Send one message through the same provider-backed session path."""

    return ChatSession(model=model, provider_name=provider_name).send(prompt)


def _provider_from_legacy_client(
    client: OpenAI | None,
    provider_name: str | None,
) -> LLMProvider:
    if client is None:
        from .providers import get_default_provider

        return get_default_provider(provider_name)

    from .providers.openai_compatible import OpenAICompatibleProvider

    return OpenAICompatibleProvider(
        name=provider_name or "openai-compatible",
        base_url=None,
        api_key=None,
        models=(),
        client=client,
    )


def _default_model_for(provider: LLMProvider) -> str:
    configured = get_default_model()
    models = provider.list_models()
    if not models:
        return configured
    provider_models = {spec.model for spec in models}
    return configured if configured in provider_models else models[0].model
