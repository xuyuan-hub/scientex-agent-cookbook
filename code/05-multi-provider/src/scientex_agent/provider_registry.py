"""Provider protocol and registry used by the conversation layer."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from .provider_types import ChatRequest, ChatResponse, ModelSpec, StreamEvent


@runtime_checkable
class LLMProvider(Protocol):
    """The minimum capability required by normal and tool conversations."""

    name: str

    def list_models(self) -> Sequence[ModelSpec]: ...

    def chat(self, request: ChatRequest) -> ChatResponse: ...


@runtime_checkable
class StreamingLLMProvider(LLMProvider, Protocol):
    """An LLM provider that also preserves Step 03 streaming behaviour."""

    def chat_stream(self, request: ChatRequest) -> Iterator[StreamEvent]: ...


@runtime_checkable
class LangChainLLMProvider(LLMProvider, Protocol):
    """Provider that can supply the framework-specific bridge used in Step 10."""

    def to_langchain_chat_model(self, model: str, *, streaming: bool): ...


class ProviderRegistry:
    """Registry of named provider adapters."""

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider) -> None:
        if provider.name in self._providers:
            raise ValueError(f"Provider already registered: {provider.name}")
        self._providers[provider.name] = provider

    def get(self, name: str) -> LLMProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            available = ", ".join(self._providers) or "(none)"
            raise KeyError(f"Unknown provider: {name}. Available: {available}") from exc

    def list_provider_names(self) -> list[str]:
        return list(self._providers)

    def list_models(self) -> list[ModelSpec]:
        return [model for provider in self._providers.values() for model in provider.list_models()]

    def get_model_spec(self, provider_name: str, model_name: str) -> ModelSpec | None:
        provider = self._providers.get(provider_name)
        if provider is None:
            return None
        return next((spec for spec in provider.list_models() if spec.model == model_name), None)
