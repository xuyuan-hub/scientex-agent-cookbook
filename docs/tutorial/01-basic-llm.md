# Step 01: 首次 LLM 调用

## 目标

用 OpenAI SDK 发送一条消息给 LLM，拿到回复并打印出来。

## 前置条件

- 完成 [00-project-setup.md](00-project-setup.md)
- 至少有一个 LLM API Key（DeepSeek 推荐的性价比最高，OpenAI 最稳定）

## 设计思路

### 本次调用的最小可行代码

```
用户输入 text → 构造 messages → client.chat.completions.create() → 打印回复
```

就这么简单。不需要类、不需要配置管理、不需要错误处理——先把路走通。

## 实现

### 1. 添加openai 模块

```bash
uv add openai

# 国内可用清华源
uv add openai --index-url=https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 新建文件：src/scientex_agent/llm_client.py

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
```

> 📄 最新代码：[code/01-basic-llm/src/scientex_agent/llm_client.py](../../code/01-basic-llm/src/scientex_agent/llm_client.py)

### 3. 更新 src/scientex_agent/cli.py

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

> 📄 最新代码：[code/01-basic-llm/src/scientex_agent/cli.py](../../code/01-basic-llm/src/scientex_agent/cli.py)

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

### LLM 返回的完整 response 对象（真实案例）

调用 `client.chat.completions.create()` 后，拿到的是一个很大的对象。如果你直接 `print(response)`，会看到这样的内容：

```python
ChatCompletion(
    id='b564763b-6e2c-48aa-bfc4-195cbab5fbae',
    choices=[
        Choice(
            finish_reason='stop',
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content='你好！😊 很高兴见到你！\n\n我是DeepSeek，你的免费AI助手，可以帮你解答问题、提供建议、处理文档、进行创作等等。有什么我可以帮你的吗？不管是闲聊还是正经问题，都欢迎随时告诉我哦~',
                refusal=None,
                role='assistant',
                annotations=None,
                audio=None,
                function_call=None,
                tool_calls=None,
                reasoning_content='嗯，用户发来一个简单的问候"你好"。\n\n这是一个非常基础的社交开场，没有提出任何具体问题或指令。\n\n我需要给出一个友好、热情的回应，建立良好的互动开端。可以用问候语加上自我介绍，并开放性地邀请用户提出任何需求，这样既回应了问候，又引导对话继续。'
            )
        )
    ],
    created=1784612358,
    model='deepseek-v4-pro',
    object='chat.completion',
    usage=CompletionUsage(
        completion_tokens=159,
        prompt_tokens=5,
        total_tokens=164,
        completion_tokens_details=CompletionTokensDetails(
            reasoning_tokens=106
        )
    )
)
```

**这就是 `response` 的全貌。** 它是一个嵌套的对象，包含了：

| 层级 | 路径 | 含义 |
|------|------|------|
| 顶层 | `response.id` | 本次请求的唯一 ID |
| 顶层 | `response.model` | 实际使用的模型名 |
| 顶层 | `response.created` | 创建时间（Unix 时间戳） |
| choices | `response.choices[0]` | 回复列表（通常只有 1 个） |
| message | `response.choices[0].message.content` | **← 我们最关心的回复文本** |
| message | `response.choices[0].message.role` | 回复角色，固定为 `"assistant"` |
| message | `response.choices[0].message.reasoning_content` | 模型的思考过程（部分模型有） |
| usage | `response.usage.prompt_tokens` | 输入 token 数 |
| usage | `response.usage.completion_tokens` | 输出 token 数 |
| usage | `response.usage.total_tokens` | 总 token 数 |

### 为什么是 `response.choices[0].message.content`？

```
response
  └── choices          # 列表，支持一次返回多个回复（通常只有 1 个）
        └── [0]        # 取第一个回复
            └── message  # 回复的消息对象
                └── content  # 消息的文本内容 ← 这就是我们要的
```

所以代码里写的是：

```python
return response.choices[0].message.content or ""
```

`or ""` 是为了防止 `content` 为 `None` 时返回空字符串。

### 常见模型

| 模型                | 厂商          | 特点                    | 文档                                                                                                                                                                              |
| ------------------- | ------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gpt-5.6-sol`     | OpenAI        | 最新旗舰，复杂推理/编程 | [docs](https://platform.openai.com/docs/)                                                                                                                                          |
| `gpt-5.6-terra`   | OpenAI        | 均衡版，速度与成本适中  |                                                                                                                                                                                   |
| `gpt-5.6-luna`    | OpenAI        | 快速便宜，高吞吐任务    |                                                                                                                                                                                   |
| `deepseek-v4-pro` | DeepSeek      | 最新旗舰，性价比高      | [docs](https://api-docs.deepseek.com/)                                                                                                                                             |
| `deepseek-chat`   | DeepSeek      | 通用对话                |                                                                                                                                                                                   |
| `glm-5.2`         | 智谱 AI (GLM) | 最新旗舰，1M 上下文     | [docs](https://docs.bigmodel.cn/cn/guide/develop/python/introduction)                                                                                                              |
| `qwen3.7-max`     | 阿里百炼      | 旗舰模型，Agent 能力强  | [docs](https://bailian.console.aliyun.com/cn-beijing?utm_content=se_1021228199&gclid=EAIaIQobChMI-P-Z-vDglQMVplUPAh1H-BqUEAAYASAAEgIdbvD_BwE&tab=api#/api/?type=model&url=3016807) |
| `qwen3.7-plus`    | 阿里百炼      | 效果/速度/成本均衡      |                                                                                                                                                                                   |
| `claude-sonnet-5` | Anthropic     | 长上下文，科学场景好    |                                                                                                                                                                                   |

注意：Anthropic 和 Gemini 不兼容 OpenAI 接口，后续步骤会处理。

## 当前局限

1. 每次调用都是"一次性"的——不记得之前说过什么
2. 环境变量需要手动设置，没有自动加载 `.env` 文件
3. 没有错误处理（API key 错了怎么办？网络断了怎么办？）
4. 回复是一口气返回的，不能边生成边看

## 下一步

→ [02-multi-turn-conversation.md](02-multi-turn-conversation.md)
