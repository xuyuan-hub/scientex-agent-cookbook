# Step 01: 首次 LLM 调用

## 目标

用 OpenAI SDK 发送一条消息给 LLM，拿到回复并打印出来。

## 前置条件

- 完成 [00-project-setup.md](00-project-setup.md)
- 至少有一个 LLM API Key（DeepSeek 推荐的性价比最高，OpenAI 最稳定）

## 设计思路

### 为什么直接上 OpenAI SDK

```python
# ❌ 不要这样手写 HTTP
import requests
resp = requests.post("https://api.openai.com/v1/chat/completions", ...)

# ✅ 直接用官方 SDK
from openai import OpenAI
client = OpenAI()
resp = client.chat.completions.create(model="gpt-4o", messages=[...])
```

理由：
1. 自动处理重试、超时、错误
2. 类型提示完整，IDE 友好
3. 统一的 streaming 接口
4. 很多非 OpenAI 厂商也兼容 OpenAI 接口格式（DeepSeek, OpenRouter, Ollama 等）

### 本次调用的最小可行代码

```
用户输入 text → 构造 messages → client.chat.completions.create() → 打印回复
```

就这么简单。不需要类、不需要配置管理、不需要错误处理——先把路走通。

## 实现

### 1. 新建文件：src/scientex_agent/llm_client.py

```python
"""Minimal LLM client using OpenAI SDK."""

from __future__ import annotations

import os

from openai import OpenAI


def get_client() -> OpenAI:
    """Create an OpenAI client from environment variables.

    Priority: DEEPSEEK_* > OPENAI_* (easy to override).
    """
    # DeepSeek
    if os.environ.get("DEEPSEEK_API_KEY"):
        return OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )
    # OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    raise RuntimeError(
        "No LLM API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY."
    )


def get_default_model() -> str:
    """Return the default model name from env or a reasonable default."""
    return os.environ.get("DEEPSEEK_MODEL") or os.environ.get("OPENAI_MODEL") or "deepseek-chat"


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
```

### 2. 更新 src/scientex_agent/cli.py

```python
"""CLI entry point."""

from __future__ import annotations

import argparse

from . import __version__
from .llm_client import chat


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scientex_agent")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    # chat 子命令
    chat_parser = subparsers.add_parser("chat", help="Send a message to the LLM")
    chat_parser.add_argument("message", nargs="+", help="The message to send")
    chat_parser.add_argument("--model", help="Model to use")

    args = parser.parse_args(argv)

    if args.command == "chat":
        prompt = " ".join(args.message)
        print(f"> {prompt}")
        print()
        response = chat(prompt, model=args.model)
        print(response)
        return 0

    parser.print_help()
    return 1
```

## 验证

### 1. 设置环境变量

```bash
# 创建 .env.local（或在 shell 中 export）
cat > .env.local << 'EOF'
DEEPSEEK_API_KEY=sk-your-real-key-here
DEEPSEEK_MODEL=deepseek-chat
EOF
```

**重要**：`.env.local` 已在 `.gitignore` 中，不会被提交。

### 2. 加载环境变量并运行

```bash
# 方式 1：手动 export
export $(cat .env.local | xargs)
uv run scientex_agent chat "Hello, what is the capital of France?"

# 方式 2：一行搞定
DEEPSEEK_API_KEY=sk-xxx uv run scientex_agent chat "解释一下什么是蛋白质折叠"
```

### 3. 预期输出

```
> Hello, what is the capital of France?

The capital of France is Paris.
```

## 深入理解

### OpenAI SDK 的请求结构

```python
response = client.chat.completions.create(
    model="deepseek-chat",        # (1) 模型名
    messages=[                     # (2) 消息列表
        {"role": "user", "content": "Hello"},
    ],
    temperature=0.7,               # (3) 随机性 0-2
    max_tokens=4096,               # (4) 最大输出 token 数
)
```

`response` 对象结构：
```python
response.choices[0].message.content  # 回复文本
response.choices[0].message.role     # "assistant"
response.usage.prompt_tokens         # 输入 token 数
response.usage.completion_tokens     # 输出 token 数
response.model                       # 实际使用的模型名
```

### 常见模型

| 模型 | 厂商 | 特点 |
|---|---|---|
| `gpt-4o` | OpenAI | 最强综合能力 |
| `gpt-4o-mini` | OpenAI | 快速便宜 |
| `deepseek-chat` | DeepSeek | 性价比高 |
| `claude-sonnet-5` | Anthropic | 长上下文，科学场景好 |

注意：Anthropic 和 Gemini 不兼容 OpenAI 接口，后续步骤会处理。

## 当前局限

1. 每次调用都是"一次性"的——不记得之前说过什么
2. 环境变量需要手动设置，没有自动加载 `.env` 文件
3. 没有错误处理（API key 错了怎么办？网络断了怎么办？）
4. 回复是一口气返回的，不能边生成边看

## 下一步

→ [02-multi-turn-conversation.md](02-multi-turn-conversation.md)
