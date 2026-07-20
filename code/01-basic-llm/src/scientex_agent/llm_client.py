"""Minimal LLM client using OpenAI SDK.

Supported providers & common models:
  - DeepSeek: deepseek-v4-pro, deepseek-chat
    Docs: https://api-docs.deepseek.com/
  - OpenAI:   gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna
    Docs: https://platform.openai.com/docs/
  - GLM (ZhipuAI): glm-5.2
    Docs: https://docs.bigmodel.cn/
  - Bailian (Alibaba): qwen3.7-max, qwen3.7-plus
    Docs: https://help.aliyun.com/product/2864317.html
"""

from __future__ import annotations

import os

from openai import OpenAI

def get_client() -> OpenAI:
    """Create an OpenAI client from environment variables.

    Priority: DEEPSEEK_* > OPENAI_* (easy to override).
    """
    # DeepSeek – https://api-docs.deepseek.com/
    if os.environ.get("DEEPSEEK_API_KEY"):
        return OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )
    # OpenAI – https://platform.openai.com/docs/
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    raise RuntimeError(
        "No LLM API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY."
    )

def get_default_model() -> str:
    """Return the default model name from env or a reasonable default."""
    return os.environ.get("DEEPSEEK_MODEL") or os.environ.get("OPENAI_MODEL") or "deepseek-v4-pro"

def chat(prompt: str, *, model: str | None = None) -> str:
    """Send a single message and return the response text.

    Args:
        prompt: The user's message.
        model: Model name. Uses default if not specified.

    Returns:
        The assistant's text response.
    """
    client = get_client()
    model_name = model or get_default_model()

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content or ""