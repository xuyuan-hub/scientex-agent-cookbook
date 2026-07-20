# Step 02: 多轮对话

## 目标

让 LLM 记住之前的对话上下文，实现真正的"聊天"。

## 前置条件

- 完成 [01-basic-llm.md](01-basic-llm.md)

## 设计思路

### 为什么单次调用"不记得"

```
第一次调用:
  messages = [{"role": "user", "content": "我叫张三"}]
  LLM: "你好张三"

第二次调用:
  messages = [{"role": "user", "content": "我叫什么名字？"}]
  LLM: "我不知道你的名字。"  ← 因为没把历史传过去！
```

### 解决方案：维护消息历史

```python
# 维护一个消息列表
history = [
    {"role": "user", "content": "我叫张三"},
    {"role": "assistant", "content": "你好张三"},
    {"role": "user", "content": "我叫什么名字？"},
]
# 把完整历史传给 LLM
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=history,  # ← 关键：传完整历史
)
```

### 消息角色的含义

| role          | 含义                   | 例子                       |
| ------------- | ---------------------- | -------------------------- |
| `system`    | 系统指令，定义 AI 行为 | "你是一个有帮助的助手"     |
| `user`      | 用户说的话             | "你好"                     |
| `assistant` | AI 的回复              | "你好！有什么可以帮你的？" |
| `tool`      | 工具执行结果           | （后续步骤会用到）         |

### 封装为 ChatSession 类

```
ChatSession
  ├── client: OpenAI       # LLM 客户端
  ├── model: str           # 模型名
  ├── messages: list[dict] # 历史消息
  ├── send(text) → str     # 发送消息，自动维护历史
  └── system(text)         # 设置系统指令
```

## 实现

### 1. 重写 src/scientex_agent/llm_client.py

```python
"""LLM chat session with conversation history."""


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

class ChatSession:
    """A conversation session that maintains message history.

    Usage::

        session = ChatSession()
        session.system("You are a helpful biology tutor.")
        reply = session.send("What is the central dogma?")
        reply2 = session.send("Explain that in simpler terms.")  # remembers context
    """

    def __init__(
            self,
            model: str | None = None,
            client: OpenAI | None = None,
            )-> None:
        self.client = client or get_client()
        self.model = model or get_default_model()
        self.messages: list[dict[str, str]] = []

    def system(self, content:str)->None:
        """Set or replace the system prompt (always at position 0)."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0] = {"role": "system", "content": content}
        else:
            self.messages.insert(0, {"role": "system", "content": content})

    def send(self, content: str,**kwargs)->str:
        """Send a user message and return the assistant reply.

        The message pair (user + assistant) is appended to history.

        Args:
            content: The user's message text.
            **kwargs: Passed through to the API (temperature, max_tokens, etc.).

        Returns:
            The assistant's response text.
        """
        # Append use message
        self.messages.append({"role": "user", "content": content})

        # Call LLM with full history
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            **kwargs,
        )

        # Extract and store reply
        reply = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def last_user_message(self) -> dict | None:
        """Return the most recent user message, if any."""
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return msg
        return None

    def clear(self) -> None:
        """Clear all messages. Preserves system prompt if set."""
        system_msg = self.messages[0] if self.messages and self.messages[0]["role"] == "system" else None
        self.messages = [system_msg] if system_msg else []

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

> 📄 最新代码：[code/02-multi-turn-conversation/src/scientex_agent/llm_client.py](../../code/02-multi-turn-conversation/src/scientex_agent/llm_client.py)

### 2. 更新 CLI

```python
"""CLI entry point."""

from __future__ import annotations

import argparse

from . import __version__
from .llm_client import chat

def _interactive_chat(args) -> int:
    """Interactive chat REPL."""
    from .llm_client import ChatSession

    session = ChatSession(model=args.model)
    if args.system:
        session.system(args.system)

    print("Scientex Chat (type /exit to quit, /clear to reset)")
    print(f"Model: {session.model}")
    print()

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ("/exit", "/quit"):
            break
        if user_input.lower() == "/clear":
            session.clear()
            print("History cleared.")
            continue
        if not user_input.strip():
            continue

        print()
        reply = session.send(user_input)
        print(reply)
        print()
        print(f"[{len(session.messages)} messages in history]")
        print()

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scientex_agent")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    # chat 子命令
    chat_parser = subparsers.add_parser("chat", help="Send a message to the LLM")
    chat_parser.add_argument("message", nargs="*", help="The message to send")
    chat_parser.add_argument("--model", help="Model to use")
    chat_parser.add_argument("--interactive", action="store_true", help="Start an interactive chat session")
    chat_parser.add_argument("--system", help="System prompt for interactive mode")

    args = parser.parse_args(argv)

    if args.command == "chat":
        if args.interactive:
            return _interactive_chat(args)
        prompt = " ".join(args.message)
        print(f"> {prompt}")
        print()
        response = chat(prompt, model=args.model)
        print(response)
        return 0
  
    parser.print_help()
    return 1
```

> 📄 最新代码：[code/02-multi-turn-conversation/src/scientex_agent/cli.py](../../code/02-multi-turn-conversation/src/scientex_agent/cli.py)

## 验证

### 交互式测试

```bash
export $(cat .env.local | xargs)
uv run scientex_agent chat --interactive
```

```
> 我叫张三，我喜欢吃苹果
Scientex Chat (type /exit to quit, /clear to reset)

你好张三！苹果是很健康的水果。有什么想聊的吗？

> 我叫什么名字？我喜欢吃什么？
你叫张三，你喜欢吃苹果！😊

> /exit
```

### 程序化测试

```python
from scientex_agent.llm_client import ChatSession

session = ChatSession()
session.system("用中文回复，尽量简洁。")

reply1 = session.send("1+1=?")
assert "2" in reply1

reply2 = session.send("刚才我问了什么？")
assert "1+1" in reply2.lower()

print(f"Messages: {len(session.messages)}")
# 预期: 4 (system + user1 + assistant1 + user2 + assistant2 = 5? No, 1+1+1+1+1 = 5)
```

## 深入理解

### Token 消耗会随对话增长

```
对话轮次   消息数   大约 token 消耗
1          2        ~200
5          10       ~800
20         40       ~3000
100        200      ~15000  ← 可能超过上下文窗口！
```

这是后续需要解决的"上下文窗口管理"问题。现在先不管——能用就行。

### System Prompt 的最佳实践

```python
# ✅ 好的 system prompt：定义角色和约束
session.system(
    "你是一个生物信息学专家。用中文回复，给出具体步骤。"
    "如果不确定，请明确说明。"
)

# ❌ 不好的 system prompt：太模糊
session.system("你是助手")
```

## 当前局限

1. 不支持流式输出（回复要等完全生成好才显示）
2. 没有工具调用能力
3. 消息历史只存在内存中，重启就丢失
4. 对话太长会超出模型上下文窗口

## 下一步

→ [03-streaming.md](03-streaming.md)
