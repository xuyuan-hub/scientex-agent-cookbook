"""Configured provider adapters and their model catalogues."""

from __future__ import annotations

import os

from ..provider_registry import LLMProvider, ProviderRegistry
from ..provider_types import ModelSpec
from .openai_compatible import OpenAICompatibleProvider

ANTHROPIC_MODELS = (
    ModelSpec("anthropic", "claude-sonnet-5", 200_000, supports_tools=True),
    ModelSpec("anthropic", "claude-haiku-4-5", 200_000, supports_tools=True),
)

OPENAI_MODELS = (
    ModelSpec("openai", "gpt-5.6-terra", 400_000, supports_tools=True),
)

DEEPSEEK_MODELS = (
    ModelSpec("deepseek", "deepseek-v4-pro", 128_000, supports_tools=True),
)


def build_default_registry() -> ProviderRegistry:
    """Register every provider configured through environment variables."""

    registry = ProviderRegistry()

    if os.environ.get("DEEPSEEK_API_KEY"):
        registry.register(
            OpenAICompatibleProvider(
                name="deepseek",
                base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                api_key=os.environ["DEEPSEEK_API_KEY"],
                models=DEEPSEEK_MODELS,
            )
        )

    if os.environ.get("OPENAI_API_KEY"):
        registry.register(
            OpenAICompatibleProvider(
                name="openai",
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=os.environ["OPENAI_API_KEY"],
                models=OPENAI_MODELS,
            )
        )

    if os.environ.get("ANTHROPIC_API_KEY"):
        from .anthropic import AnthropicProvider

        registry.register(
            AnthropicProvider(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                models=ANTHROPIC_MODELS,
            )
        )

    return registry


def get_default_provider(provider_name: str | None = None) -> LLMProvider:
    """Return a configured provider, respecting an explicit selection first."""

    registry = build_default_registry()
    if provider_name:
        return registry.get(provider_name)

    names = registry.list_provider_names()
    if not names:
        raise RuntimeError(
            "No LLM API key found. Set DEEPSEEK_API_KEY, OPENAI_API_KEY, "
            "or ANTHROPIC_API_KEY."
        )
    return registry.get(names[0])
