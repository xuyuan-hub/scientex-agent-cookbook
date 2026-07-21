# Step 03: 流式输出

## 目标

实现 token 级别的实时流式输出——每个字生成出来就立刻显示，而不是等整个回复完成后才返回。

## 前置条件

- 完成 [02-multi-turn-conversation.md](02-multi-turn-conversation.md)

## 设计思路

### 非流式 vs 流式

```
非流式（之前的方式）：
  client.chat.completions.create(...)
  → 等待 3 秒
  → 一次性返回全部文本 "The capital of France is Paris."

流式：
  for chunk in client.chat.completions.create(..., stream=True):
      print(chunk.choices[0].delta.content, end="")
  → "The" (0.1s 后)
  → " capital" (0.2s 后)
  → " of" (0.3s 后)
  → ...
```

### 用生成器封装流式逻辑

```python
def send_stream(self, content: str) -> Generator[str, None, None]:
    """发送消息并流式 yield 每个 token。"""
    self.messages.append({"role": "user", "content": content})

    stream = self.client.chat.completions.create(
        model=self.model,
        messages=self.messages,
        stream=True,           # ← 开启流式
        stream_options={"include_usage": True},  # ← 获取 usage 信息
    )

    full_reply = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            full_reply += delta.content
            yield delta.content  # ← 逐个 yield token

    self.messages.append({"role": "assistant", "content": full_reply})
```

### 封装模式

```
ChatSession
  ├── send(text) → str           # 非流式（已有的）
  └── send_stream(text) → Generator  # 流式（新增）
```

两个方法共享同样的历史管理逻辑，只有返回方式不同。

## 实现

### 更新 src/scientex_agent/llm_client.py

在 `ChatSession` 类中新增 `send_stream` 方法：

```python
from collections.abc import Generator


class ChatSession:
    # ... 之前的代码保持不变 ...

    def send_stream(
        self,
        content: str,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Send a user message and yield the assistant reply token by token.

        Usage::

            for token in session.send_stream("Hello"):
                print(token, end="", flush=True)

        Args:
            content: The user's message.
            **kwargs: Passed through to the API.

        Yields:
            Text tokens as they arrive.
        """
        self.messages.append({"role": "user", "content": content})

        # 合并 stream_options
        stream_kwargs = {
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        stream_kwargs.update(kwargs)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            **stream_kwargs,
        )

        full_reply = ""
        for chunk in response:
            # 有些 chunk 没有 choices（如 usage 信息）
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                full_reply += delta.content
                yield delta.content

        self.messages.append({"role": "assistant", "content": full_reply})

    def send(self, content: str, **kwargs) -> str:
        """Send a user message and return the complete reply.

        This is the non-streaming version. For streaming, use send_stream().
        """
        self.messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            **kwargs,
        )

        reply = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": reply})
        return reply
```

### 更新 CLI 交互模式，支持流式

```python
# src/scientex_agent/cli.py

def _interactive_chat(args) -> int:
    from .llm_client import ChatSession

    session = ChatSession(model=args.model)
    if args.system:
        session.system(args.system)

    print(f"Scientex Chat (Model: {session.model})")
    print("Commands: /exit, /clear, /nostream")
    print()

    use_stream = args.stream

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
            print("History cleared")
            continue
        if not user_input.strip():
            continue

        print()
        if use_stream:
            for token in session.send_stream(user_input):
                print(token, end="", flush=True)
            print()
        else:
            print(session.send(user_input))
        print()
        print(f"[{len(session.messages)} msgs]")
        print()

    return 0
```

## 验证

```bash
uv run scientex_agent chat --interactive

> 用 3 句话介绍机器学习
(逐字出现)
机器学习是人工智能的一个分支...
[6 msgs]
```

你会看到文本**逐字输出**，而不是等待后一次性显示。这就是流式。

## 深入理解

### OpenAI SDK 流式响应的细节

```python
stream = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in stream:
    # chunk 的结构：
    # chunk.choices[0].delta.content   ← 当前 token 的文本，可能为 None
    # chunk.choices[0].finish_reason   ← 结束时为 "stop"
    # chunk.usage                      ← 最后一个 chunk 可能包含 usage
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### 为什么要 `flush=True`

Python 的 `print()` 默认有缓冲。在终端中可能每隔几行才刷新一次。
`flush=True` 强制每个 token 都立即显示。

### 流式对用户体验的影响

```
非流式：用户等 5 秒 → 一次性看到完整回复
流式：   用户 0.1 秒后开始看到文字 → 边看边理解
```

对于长回复，流式体验明显更好。这也是 ChatGPT 等产品使用流式输出的原因。

### stream_options

```python
stream_options={"include_usage": True}
```

此选项让 `deepseek-v4-pro` 在流式响应中附带 token 用量信息。
注意：并非所有模型都支持（有些会报错），后续步骤会处理兼容性。

## 当前局限

1. 没有工具/函数调用——LLM 只能说，不能做
2. 只能调用 OpenAI 兼容接口（Anthropic, Gemini 不支持）
3. 流式错误处理不完善（网络中断时生成器会直接抛异常）
4. 消息历史只在内存中

## 下一步

→ [04-tool-calling.md](04-tool-calling.md)
