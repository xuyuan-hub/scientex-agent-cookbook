# Step 05: 多提供商支持

## 目标

把 Step 04 的单一 OpenAI-compatible 调用，重构为可替换的 provider adapter，且**不丢失**已经完成的能力：

- 普通多轮对话；
- 工具调用循环；
- 流式输出；
- CLI 的 `--provider` 选择。

完成后，`ChatSession` 和 `agent_loop` 不再依赖 `OpenAI` SDK；它们只依赖 `LLMProvider`。
OpenAI、DeepSeek 和 Anthropic 的 SDK 差异被限制在 adapter 内。

## 前置条件

- 完成 [04-tool-calling.md](04-tool-calling.md)
- 第 1、2 节和自动测试不需要 API Key
- 真实 API 冒烟测试需要至少一个 API Key（推荐 DeepSeek）

## 本章完成后，代码如何演进

```
Step 04
  ChatSession ──直接调用──> OpenAI SDK
  agent_loop  ──直接调用──> OpenAI SDK

Step 05
  ChatSession ──> LLMProvider <── OpenAICompatibleProvider
       │                            ├── DeepSeek
       ├──> agent_loop               └── OpenAI
       │
       └──> StreamingLLMProvider <── AnthropicProvider

Step 06
  PersistentChatSession 继承同一个 provider-backed ChatSession
```

重点是**替换依赖方向，而不是复制一套对话程序**。因此，原有的 `messages` 历史仍保留为字典列表；
这让 Step 06 的 SQLite 存储可以继续使用同一份数据格式。

---

## 新增类与函数速览

本章新增或关键改造的代码如下。先知道每个名称负责什么，再阅读后面的实现会更容易。

### `provider_types.py`：统一的数据格式

| 名称 | 类型 | 简要作用 |
| --- | --- | --- |
| `ModelSpec` | 数据类 | 描述模型名称、上下文窗口及是否支持工具、流式等能力。 |
| `ToolCall` | 数据类 | 用统一格式表示模型要求执行的一次工具调用。 |
| `ChatMessage` | 数据类 | 用统一格式表示一条历史消息，不依赖任何厂商 SDK。 |
| `ChatRequest` | 数据类 | 会话层发给 provider 的完整请求，包括历史、模型和工具 schema。 |
| `Usage` | 数据类 | 记录输入、输出及缓存 token；`__add__()` 可合并两次用量。 |
| `ChatResponse` | 数据类 | provider 返回的完整标准化结果，包括文本、工具调用和用量。 |
| `StreamEvent` | 数据类 | 表示一段流式文本，或流结束时的完整 `ChatResponse`。 |
| `message_from_dict()` | 函数 | 将旧的字典历史转换成 `ChatMessage`，作为进入 provider 边界的桥。 |
| `response_message()` | 函数 | 将 `ChatResponse` 转回旧的 assistant 字典，继续写入会话历史。 |

### `provider_registry.py`：能力约定与选择

| 名称 | 类型 | 简要作用 |
| --- | --- | --- |
| `LLMProvider` | `Protocol` | 规定所有 provider 至少要提供 `name`、`list_models()` 和 `chat()`。 |
| `StreamingLLMProvider` | `Protocol` | 在 `LLMProvider` 基础上增加 `chat_stream()`，表示支持流式输出。 |
| `LangChainLLMProvider` | `Protocol` | 为 Step 10 提供转成 LangChain 模型的可选能力。 |
| `ProviderRegistry` | 类 | 按名称注册、查找 provider，并集中列出 provider 与模型。 |
| `register()` / `get()` | 方法 | 注册 provider；或按名称取得 provider，并在失败时提示可用名称。 |
| `list_provider_names()` / `list_models()` | 方法 | 给 CLI 或 UI 列出当前可用的 provider、模型。 |
| `get_model_spec()` | 方法 | 查询某个 provider 下某个模型的能力描述。 |

### `providers/`：厂商适配器与工厂

| 名称 | 类型 | 简要作用 |
| --- | --- | --- |
| `OpenAICompatibleProvider` | 类 | 将中立请求翻译为 OpenAI-compatible SDK 调用；可复用给 OpenAI、DeepSeek 等。 |
| `OpenAICompatibleProvider.chat()` | 方法 | 发起一次普通请求，再将 SDK 响应解析为 `ChatResponse`。 |
| `OpenAICompatibleProvider.chat_stream()` | 方法 | 逐段输出文本，累积拆开的工具参数，并在结束时给出完整响应。 |
| `_request_kwargs()` | 函数 | 把 `ChatRequest` 转为 SDK 所需的关键字参数。 |
| `_to_openai_messages()` | 函数 | 把中立消息转为 OpenAI-compatible 的 `messages` 格式。 |
| `_parse_response()` | 函数 | 把一次完整 SDK 响应转为 `ChatResponse`。 |
| `_tool_call_from_partial()` | 函数 | 把流中累积的半截工具调用还原为 `ToolCall`。 |
| `AnthropicProvider` | 类（选做） | 将相同的中立接口转为 Anthropic Messages API 格式。 |
| `_to_anthropic_messages()` | 函数（选做） | 把 system、工具调用和工具结果转换成 Anthropic 的 content block。 |
| `_to_anthropic_tools()` | 函数（选做） | 将 Step 04 的 OpenAI 形状 tool schema 改成 Anthropic 的 `input_schema`。 |
| `build_default_registry()` | 函数 | 根据已配置的环境变量创建并注册可用 provider。 |
| `get_default_provider()` | 函数 | 优先返回显式指定的 provider；未指定时按默认顺序自动选择。 |

### 会话、工具循环与 CLI：业务层改造

| 名称 | 类型 | 简要作用 |
| --- | --- | --- |
| `ChatSession` | 改造后的类 | 保留原来的会话历史 API，但把所有模型调用改为经 `LLMProvider` 完成。 |
| `ChatSession.send()` | 方法 | 发送普通非流式消息。 |
| `ChatSession.send_stream()` | 方法 | 流式发送普通消息；不支持流式的 provider 自动降级为一次性输出。 |
| `ChatSession.send_with_tools()` | 方法 | 通过 provider 运行 Step 04 的非流式工具循环。 |
| `ChatSession.send_with_tools_stream()` | 方法 | 通过 provider 运行流式工具循环。 |
| `ChatSession._request()` | 方法 | 将当前字典历史和参数组装成不可变的 `ChatRequest`。 |
| `chat()` | 函数 | 单次对话的便捷入口，内部仍复用 `ChatSession`。 |
| `chat_with_tools()` | 函数 | 完成“模型请求 → 执行工具 → 再请求模型”的非流式循环。 |
| `chat_with_tools_stream()` | 函数 | 在上述循环中同时逐 token 转发文本。 |
| `_collect_stream()` | 函数 | 收集 provider 的流事件，并通过生成器返回最终 `ChatResponse`。 |
| `_append_tool_results()` | 函数 | 执行所有工具调用，并把结果作为 `role="tool"` 消息写回历史。 |
| `_interactive_chat()` | 函数 | CLI 的 REPL，负责读取输入、选择流式模式并显示当前 provider。 |

后文的代码详解会重点展开其中涉及格式转换、流式状态管理和生成器返回值的函数。

---

## 1. 定义 provider 之间的共同语言

创建 [provider_types.py](../../code/05-multi-provider/src/scientex_agent/provider_types.py)。其中有五个关键类型：

| 类型 | 作用 |
| --- | --- |
| `ChatMessage` | 不依赖 SDK 的历史消息 |
| `ChatRequest` | 会话层发给 provider 的请求 |
| `ChatResponse` | provider 返回给会话层的完整响应 |
| `ToolCall` | 模型要求执行的工具调用 |
| `StreamEvent` | 流式文本片段，或携带最终完整响应的结束事件 |

`ChatRequest` 使用不可变的 tuple，避免 adapter 在翻译请求时意外改写会话历史：

```python
@dataclass(frozen=True)
class ChatRequest:
    messages: tuple[ChatMessage, ...]
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    tools: tuple[dict[str, Any], ...] = ()
```

Step 04 的 `ToolRegistry` 目前生成 OpenAI 形状的 schema。为了让迁移只新增一个概念，
本章保留这个输出：OpenAI-compatible adapter 直接使用它，Anthropic adapter 在自己的边界处翻译它。
以后若要完全中立，可以再引入 `ToolDefinition(name, description, input_schema)`。

### 1.1 关键脚本：中立消息类型与迁移桥

下面是 [provider_types.py](../../code/05-multi-provider/src/scientex_agent/provider_types.py) 中最关键的部分。前四个
数据类是 provider 边界的输入和输出；最后两个函数负责和 Step 02、04 已有的字典历史互转：

```python
@dataclass(frozen=True)
class ToolCall:
    """A provider-neutral request to invoke one registered tool."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatMessage:
    """A provider-neutral conversation message."""

    role: str
    content: str = ""
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ChatResponse:
    content: str = ""
    usage: Usage = field(default_factory=Usage)
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def message_from_dict(message: Mapping[str, Any]) -> ChatMessage:
    """Convert the legacy dictionary history used by earlier steps."""

    tool_calls: list[ToolCall] = []
    for item in message.get("tool_calls") or []:
        function = item.get("function", {}) if isinstance(item, Mapping) else {}
        raw_arguments = function.get("arguments", {})
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}
        elif isinstance(raw_arguments, Mapping):
            arguments = dict(raw_arguments)
        else:
            arguments = {}
        tool_calls.append(
            ToolCall(
                id=str(item.get("id", "")),
                name=str(function.get("name", "")),
                arguments=arguments,
            )
        )

    content = message.get("content", "")
    return ChatMessage(
        role=str(message["role"]),
        content="" if content is None else str(content),
        tool_call_id=message.get("tool_call_id"),
        name=message.get("name"),
        tool_calls=tuple(tool_calls),
    )


def response_message(response: ChatResponse) -> dict[str, Any]:
    """Convert a normalized response back to the legacy history format."""

    message: dict[str, Any] = {"role": "assistant", "content": response.content}
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in response.tool_calls
        ]
    return message
```

#### 代码详解

**1. 为什么 `frozen=True` 和 tuple 要一起使用？**

`frozen=True` 禁止给数据类字段重新赋值，tuple 则阻止 `messages`、`tool_calls` 被 `append()`。
这避免了 adapter 在把请求翻译成 SDK 格式时，意外污染 `ChatSession.messages`；它仍然是唯一可写的历史来源。

**2. `message_from_dict()` 为什么要兼容两种 `arguments`？**

Step 04 写入历史时，工具参数是 JSON 字符串；FakeProvider 或自定义代码也可能直接放 Python 字典。
因此它按下面的规则归一化：

```text
JSON 字符串  ──json.loads──> dict
Python dict ──dict(...)────> dict 的副本
缺失/格式错误 ───────────────> {}
```

即使供应商返回了损坏的工具参数，历史转换也不会让整个会话崩溃；真正执行工具时会由 `ToolRegistry.execute()`
返回可记录的错误结果。

**3. 为什么还要把 `ChatResponse` 转回字典？**

`response_message()` 保留了 Step 04 的 OpenAI 形状：assistant 的工具调用仍放在 `tool_calls` 中，参数仍为 JSON 字符串。
这样 Step 06 保存 SQLite 时不用为 Step 05 增加第二套存储格式：

```text
旧字典历史 ── message_from_dict() ──> provider 中立请求
provider 响应 ── response_message() ──> 旧字典历史
```

### 1.2 定义协议和 registry

创建 [provider_registry.py](../../code/05-multi-provider/src/scientex_agent/provider_registry.py)：

```python
@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def list_models(self) -> Sequence[ModelSpec]: ...
    def chat(self, request: ChatRequest) -> ChatResponse: ...


@runtime_checkable
class StreamingLLMProvider(LLMProvider, Protocol):
    def chat_stream(self, request: ChatRequest) -> Iterator[StreamEvent]: ...
```

这里分成两个协议很重要：所有 provider 都能完成普通对话和工具循环；实现
`StreamingLLMProvider` 的 provider 还会保留 Step 03 的 token 级输出。

`Protocol` 是**结构化类型**：只要对象有同名且兼容的方法，就可以作为 provider 使用，不要求继承某个基类。
因此测试里的 `FakeProvider` 和未来的 Gemini adapter 都能直接注入 `ChatSession`。

```python
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
            raise KeyError(
                f"Unknown provider: {name}. Available: {available}"
            ) from exc

    def list_provider_names(self) -> list[str]:
        return list(self._providers)
```

`raise ... from exc` 会把原始的 `KeyError` 保留为异常链，同时给用户一条包含可选 provider 名称的错误信息。
这比让底层字典的错误直接冒出来更适合 CLI。

### 1.3 先用 FakeProvider 验证边界

在接触真实 API 前，先阅读 [Python `unittest` 入门](unittest-basics.md)，再运行无网络测试：

```bash
uv run python -B -m unittest discover -s tests -v
```

测试中的 [FakeProvider](../../tests/test_provider_integration.py) 会先要求调用 `add` 工具，收到
工具结果后再回复；[test_adapter.py](../../tests/test_adapter.py) 则用 fake SDK 测试 adapter 的请求、
响应和流式转换。这证明你可以在不连接任何供应商的情况下验证：

1. `ChatSession.send()` 通过 provider 工作；
2. Step 04 的工具循环通过 provider 工作；
3. Step 04 的流式工具循环通过 provider 工作。

`@runtime_checkable` 只会在运行时确认属性存在；方法签名和行为仍要靠这类测试保证。

---

## 2. 实现第一个 adapter：OpenAI-compatible

创建 [openai_compatible.py](../../code/05-multi-provider/src/scientex_agent/providers/openai_compatible.py)。
这个 adapter 可复用于 OpenAI、DeepSeek、OpenRouter、Ollama、vLLM 等 OpenAI-compatible 服务。

它负责三次转换：

```
ChatRequest
    │  _to_openai_messages()
    ▼
OpenAI SDK 的 chat.completions.create()
    │  _parse_response()
    ▼
ChatResponse
```

流式版本也遵循相同边界：adapter 累积 SDK 中分散的 tool-call chunk，持续产生文本 `StreamEvent`，
最后产生一个包含完整 `ChatResponse` 的结束事件。会话层因此不需要了解任何 SDK chunk 格式。

### 2.1 关键脚本：普通请求与响应的双向转换

`OpenAICompatibleProvider` 的普通调用只有两行，但真正的适配工作在私有辅助函数中完成：

```python
class OpenAICompatibleProvider:
    def chat(self, request: ChatRequest) -> ChatResponse:
        response = self._client.chat.completions.create(**_request_kwargs(request))
        return _parse_response(response)


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
```

#### 代码详解

**1. `**_request_kwargs(request)` 是什么？**

`_request_kwargs()` 把中立 `ChatRequest` 翻译成 SDK 接受的字典；`**` 再把字典展开为关键字参数：

```python
kwargs = {"model": "deepseek-v4-pro", "messages": [...]}
client.chat.completions.create(**kwargs)

# 等价于
client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[...],
)
```

只有用户显式传入的 `temperature`、`max_tokens` 和 `tools` 才会放进请求。这能避免把 `None` 传给
不同 SDK 后出现版本相关的行为。

**2. 为什么工具参数在边界处做 JSON 转换？**

内部类型中 `ToolCall.arguments` 是便于调用 Python 函数的 `dict`，但 OpenAI-compatible wire format
要求 `function.arguments` 是字符串。因此发送时 `json.dumps()`，接收时 `json.loads()`；这对称地把
供应商格式限制在 adapter 内。

**3. `getattr(..., None) or []` 的作用**

有些响应没有 `tool_calls` 字段，有些字段值为 `None`。这句把两种情况统一为空列表，因而普通回答和
工具调用回答都能经过同一个解析函数。

### 2.2 复杂函数：流式 tool-call chunk 的累积

在普通响应中，一次就能拿到完整的函数名和 JSON 参数；流式响应把它们拆成多个 chunk，例如：

```text
chunk 1: id="call-1", name="add", arguments='{"a": 2,'
chunk 2: id=None,       name=None,   arguments=' "b": 3}'
```

所以 `chat_stream()` 不能在收到第一个 chunk 时就执行工具，必须按 `index` 累积，直到流结束：

```python
def chat_stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
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
                index, {"id": "", "name": "", "arguments": ""}
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
```

这段函数有两种事件，调用方必须同时处理：

| 事件 | 何时产生 | 调用方应该做什么 |
| --- | --- | --- |
| `StreamEvent(text=...)` | 收到一段普通文本 | 立即打印给用户 |
| `StreamEvent(response=...)` | stream 完成 | 保存完整回答，或检查其中的 `tool_calls` |

最后一个 `response` 事件不能省略：只有它同时包含完整文本、使用量、停止原因和合并完成的工具调用。
`_tool_call_from_partial()` 在此时才解析 JSON，解析失败则交给空参数 `{}`，避免半截 JSON 破坏流式输出。

### 2.3 注册已配置的 provider

[providers/__init__.py](../../code/05-multi-provider/src/scientex_agent/providers/__init__.py) 根据环境变量构建 registry：

| 环境变量 | 注册名称 | 默认 endpoint |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | `deepseek` | `https://api.deepseek.com/v1` |
| `OPENAI_API_KEY` | `openai` | `https://api.openai.com/v1` |
| `ANTHROPIC_API_KEY` | `anthropic` | Anthropic 官方 SDK |

可用 provider 的选择顺序是 DeepSeek、OpenAI、Anthropic。也可以明确指定：

```python
from scientex_agent.providers import get_default_provider

provider = get_default_provider("deepseek")
```

关键实现如下。环境变量没有配置时不会创建 SDK client，因此运行单元测试不会读取 API Key 或访问网络：

```python
def build_default_registry() -> ProviderRegistry:
    """Register every provider configured through environment variables."""

    registry = ProviderRegistry()

    if os.environ.get("DEEPSEEK_API_KEY"):
        registry.register(
            OpenAICompatibleProvider(
                name="deepseek",
                base_url=os.environ.get(
                    "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
                ),
                api_key=os.environ["DEEPSEEK_API_KEY"],
                models=DEEPSEEK_MODELS,
            )
        )

    if os.environ.get("OPENAI_API_KEY"):
        registry.register(
            OpenAICompatibleProvider(
                name="openai",
                base_url=os.environ.get(
                    "OPENAI_BASE_URL", "https://api.openai.com/v1"
                ),
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
```

显式的 `--provider` 最优先；未指定时取注册顺序中的第一个，即 DeepSeek、OpenAI、Anthropic。若指定了
未配置的名称，`registry.get()` 会列出当前实际可用的 provider，而不是静默切换到另一个账号。

---

## 3. 把 adapter 接入已有的 ChatSession

这是本章最关键的一步。更新 [llm_client.py](../../code/05-multi-provider/src/scientex_agent/llm_client.py)，让 `ChatSession`
接受 `provider` 或 `provider_name`：

```python
session = ChatSession(
    provider_name="deepseek",  # 也可以直接传入 FakeProvider
    model="deepseek-v4-pro",
)
```

原来的调用关系：

```python
response = self.client.chat.completions.create(
    model=self.model,
    messages=self.messages,
)
```

改为：

```python
response = self.provider.chat(
    ChatRequest(
        messages=tuple(message_from_dict(message) for message in self.messages),
        model=self.model,
    )
)
```

`message_from_dict()` 与 `response_message()` 是迁移桥梁：会话历史仍然是 Step 02、04、06
一直使用的字典列表，但只有在跨越 provider 边界时才转成中立类型。

### 3.1 关键脚本：`ChatSession` 只依赖 provider

以下代码是 [llm_client.py](../../code/05-multi-provider/src/scientex_agent/llm_client.py) 的核心修改。注意：
`self.messages` 仍保存字典，唯一改变的是调用模型前后的边界转换。

```python
class ChatSession:
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

    def send(
        self,
        content: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.messages.append({"role": "user", "content": content})
        response = self.provider.chat(
            self._request(temperature=temperature, max_tokens=max_tokens)
        )
        self.messages.append(response_message(response))
        return response.content

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
```

调用链现在是：

```text
send("你好")
  ├── 追加 user 字典到 self.messages
  ├── _request()：字典历史 → tuple[ChatMessage]
  ├── provider.chat()：中立请求 → SDK 请求 → 中立响应
  ├── response_message()：中立响应 → assistant 字典
  └── 追加 assistant 字典并返回文本
```

这里同时保留了 `client` 参数，是为了让前面步骤中已经写好的 `ChatSession(client=...)` 不会立刻失效。
`_provider_from_legacy_client()` 会把这个旧 client 包进 `OpenAICompatibleProvider`；新代码应优先传
`provider` 或 `provider_name`。

### 3.2 复杂函数：流式能力检测与降级

`send_stream()` 优先调用 `StreamingLLMProvider.chat_stream()`：

```python
def send_stream(
    self,
    content: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Generator[str, None, None]:
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
```

如果你自己实现的 provider 只满足 `LLMProvider`，会话仍可正常工作，只是会把完整回复作为一个
文本片段返回。这样先保证功能正确，再逐步实现流式能力。

**为什么要在循环结束后才写入历史？** 因为文本和工具调用可能分散在很多个 `StreamEvent` 里；只有最终
`response` 才是可持久化的完整 assistant 消息。若中途就把 token 写入历史，下一个工具轮次可能看到残缺的
`tool_calls` JSON。

`isinstance(self.provider, StreamingLLMProvider)` 之所以可用，是因为协议加了 `@runtime_checkable`。
它只检查 `chat_stream` 是否存在，不会验证签名或语义，所以仍需要 FakeProvider 的自动测试覆盖行为。

---

## 4. 让 Step 04 的工具循环也走 adapter

更新 [agent_loop.py](../../code/05-multi-provider/src/scientex_agent/agent_loop.py)。它现在接收 provider，而不是 `OpenAI` client：

```python
def chat_with_tools(
    provider: LLMProvider,
    model: str,
    messages: list[dict[str, Any]],
    registry: ToolRegistry,
    *,
    max_rounds: int = 10,
) -> str:
    ...
```

每一轮都执行相同的步骤：

1. 将现有历史转换为 `ChatRequest`；
2. 调用 `provider.chat()` 或 `provider.chat_stream()`；
3. 将 `ChatResponse.tool_calls` 写回历史；
4. 通过原有 `ToolRegistry` 执行工具，并把 `tool` 消息追加到历史；
5. 直到模型不再返回工具调用。

因此，工具循环仍只有一份；不同厂商对工具格式的差异只留在 adapter 中。

### 4.1 关键脚本：非流式工具循环

以下是 [agent_loop.py](../../code/05-multi-provider/src/scientex_agent/agent_loop.py) 的实现。它没有导入
`OpenAI`，只认识 `LLMProvider` 和中立的 `ChatResponse`：

```python
def chat_with_tools(
    provider: LLMProvider,
    model: str,
    messages: list[dict[str, Any]],
    registry: ToolRegistry,
    *,
    max_rounds: int = 10,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Run the same tool loop from Step 04 through a provider adapter."""

    for _ in range(max_rounds):
        response = provider.chat(
            _request(messages, model, registry,
                     temperature=temperature, max_tokens=max_tokens)
        )
        messages.append(response_message(response))
        if not response.tool_calls:
            return response.content
        _append_tool_results(messages, response, registry)
    raise RuntimeError(f"Tool loop exceeded {max_rounds} rounds")


def _request(
    messages: list[dict[str, Any]],
    model: str,
    registry: ToolRegistry,
    *,
    temperature: float | None,
    max_tokens: int | None,
) -> ChatRequest:
    return ChatRequest(
        messages=tuple(message_from_dict(message) for message in messages),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tuple(registry.schemas()),
    )


def _append_tool_results(
    messages: list[dict[str, Any]],
    response: ChatResponse,
    registry: ToolRegistry,
) -> None:
    for call in response.tool_calls:
        result = registry.execute(call.name, call.arguments)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
```

#### 代码详解

**1. 为什么先追加 assistant 消息，再执行工具？**

工具结果必须和模型的那一次调用配对。正确的历史顺序是：

```text
user 问题
assistant（包含 tool_calls）
tool（每个 tool_call 的执行结果）
assistant（根据工具结果给出最终回答）
```

如果跳过第二行，OpenAI-compatible 服务无法知道某个 `tool_call_id` 是谁发起的；Anthropic adapter 也无法
把它转换成对应的 `tool_use` block。

**2. 为什么 `max_rounds` 必须存在？**

模型可能反复请求同一个工具，或工具结果一直不能满足模型。`for _ in range(max_rounds)` 是循环保险丝：
默认最多 10 轮，超过就抛出明确异常，而不是让 CLI 一直占用 API 配额。

**3. 工具出错为什么仍写入历史？**

`registry.execute()` 返回的是 `{"ok": False, "error": "..."}`，不会直接吞掉失败。把这段 JSON 作为
`role="tool"` 的内容传回模型，模型就有机会解释错误、调整参数，或建议用户下一步操作。

### 4.2 复杂函数：流式工具循环如何接住 generator 的返回值

流式工具循环必须先把一轮 stream 的文本转发给终端，再拿到最终 `ChatResponse` 来决定是否执行工具：

```python
def chat_with_tools_stream(
    provider: LLMProvider,
    model: str,
    messages: list[dict[str, Any]],
    registry: ToolRegistry,
    *,
    max_rounds: int = 10,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Generator[str, None, None]:
    for _ in range(max_rounds):
        request = _request(messages, model, registry,
                           temperature=temperature, max_tokens=max_tokens)
        if isinstance(provider, StreamingLLMProvider):
            response = yield from _collect_stream(provider.chat_stream(request))
        else:
            response = provider.chat(request)
            if response.content:
                yield response.content

        messages.append(response_message(response))
        if not response.tool_calls:
            return
        _append_tool_results(messages, response, registry)
    raise RuntimeError(f"Tool loop exceeded {max_rounds} rounds")


def _collect_stream(events: Iterable[StreamEvent]) -> Generator[str, None, ChatResponse]:
    final_response: ChatResponse | None = None
    for event in events:
        if event.text:
            yield event.text
        if event.response is not None:
            final_response = event.response
    if final_response is None:
        raise RuntimeError("Provider stream ended without a final response")
    return final_response
```

`yield from` 是这里最容易忽略的语法。它做了两件事：

1. 将 `_collect_stream()` 产生的每个文本 token 原样转发给 CLI；
2. 在子生成器 `return final_response` 后，接收它的返回值并赋给 `response`。

因此，一次流式工具调用的状态流如下：

```text
provider stream ──> StreamEvent(text) ──> CLI 实时打印
                └─> StreamEvent(response) ──> _collect_stream return ChatResponse
                                                ├── 无 tool_calls：结束
                                                └── 有 tool_calls：执行工具，开始下一轮
```

这也是为什么 `_collect_stream()` 的返回类型写成 `Generator[str, None, ChatResponse]`：它既会 yield 字符串，
又会在结束时 return 一个 `ChatResponse`。

---

## 5. 用同一个 CLI 使用全部能力

更新 [cli.py](../../code/05-multi-provider/src/scientex_agent/cli.py)。`--provider` 不再进入另一套简化 REPL，而是传给
同一个 `ChatSession`：

```python
def _interactive_chat(args) -> int:
    session = ChatSession(model=args.model, provider_name=args.provider)
    if args.system:
        session.system(args.system)

    print(f"Scientex Chat (Provider: {session.provider.name}, Model: {session.model})")
    print("Commands: /exit, /clear")
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

        if args.stream:
            for token in session.send_with_tools_stream(user_input):
                print(token, end="", flush=True)
            print()
        else:
            print(session.send_with_tools(user_input))
    return 0


chat_parser.add_argument(
    "--stream",
    dest="stream",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable/disable streaming output (default: enabled). Use --no-stream to disable.",
)
chat_parser.add_argument(
    "--provider",
    help="LLM provider to use (deepseek, openai, anthropic). "
    "Default: auto-detect from environment.",
)
```

`argparse.BooleanOptionalAction` 会自动生成互斥的 `--stream` 和 `--no-stream`，不用自己维护两个 flag。
两种模式最终都调用同一个 `ChatSession`：差别只是一个逐 token 打印，另一个等待完整字符串。因此 provider
切换、工具调用与历史清理没有分叉实现。

```bash
# 默认从环境变量选择 provider
uv run scientex_agent chat --interactive

# 保留工具调用和流式输出，同时明确选择 DeepSeek
uv run scientex_agent chat --provider deepseek --interactive

# 非流式工具循环
uv run scientex_agent chat --provider deepseek --interactive --no-stream
```

交互界面会显示选中的 provider 和 model：

```text
Scientex Chat (Provider: deepseek, Model: deepseek-v4-pro)
Commands: /exit, /clear
```

用 Step 04 的两个问题回归验证：

```text
> 3 的 10 次方是多少？
> 现在北京时间几点？
```

第一个问题应触发 `caculate`，第二个应触发 `get_current_time`。切换 provider 不应改变
`ChatSession`、`agent_loop` 或工具函数的代码。

---

## 6. 选做：Anthropic adapter

只有需要 Anthropic 时才安装官方 SDK：

```bash
uv add anthropic
```

[anthropic.py](../../code/05-multi-provider/src/scientex_agent/providers/anthropic.py) 处理与 OpenAI 不同的三类格式：

| 概念 | OpenAI-compatible | Anthropic |
| --- | --- | --- |
| system prompt | `messages` 中的 `system` | 单独的 `system` 参数 |
| tool call | assistant 的 `tool_calls` | `tool_use` content block |
| tool result | `role="tool"` 消息 | 下一条 user 消息的 `tool_result` block |

设置 `ANTHROPIC_API_KEY` 后，registry 会自动注册它：

```bash
uv run scientex_agent chat --provider anthropic --interactive
```

你不需要修改 `ChatSession` 或工具循环；这正是 adapter 的价值。

---

## 7. 完成检查

按顺序完成这些检查：

1. 无 API Key：

   ```bash
   uv run python -B -m unittest discover -s tests -v
   ```

2. 配置 DeepSeek 或 OpenAI Key 后，普通对话：

   ```bash
   uv run scientex_agent chat --provider deepseek "你好"
   ```

3. 交互式工具调用和流式输出：

   ```bash
   uv run scientex_agent chat --provider deepseek --interactive
   ```

4. 可选：安装 Anthropic SDK 后，重复第 2、3 步并把 `--provider` 改为 `anthropic`。

## 本章边界

- 模型名称、上下文窗口和价格会变化；把模型目录当作示例配置，真实使用前以供应商文档为准。
- provider 的错误类型和重试策略还没有统一；Step 16 会完善生产化错误处理。
- Gemini、Ollama 等 adapter 可沿用本章边界继续添加，不应修改会话层和工具循环。

## 下一步

→ [06-conversation-persistence.md](06-conversation-persistence.md)

Step 06 会把同一个 provider-backed `ChatSession` 的消息历史保存到 SQLite。你在本章选中的
provider 和 model 也将成为 Frame 的持久化配置的一部分。

Step 10 的 LangGraph 编排会从同一个 registry 取出 provider，并通过 adapter 的
`to_langchain_chat_model()` 建立框架专用的 bridge；上层不再重新读取 API Key 或硬编码 endpoint。

## 参考资料

- [OpenAI API 参考](https://platform.openai.com/docs/api-reference)
- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
- [DeepSeek API 文档](https://api-docs.deepseek.com/)
- [PEP 544 — Protocols: structural subtyping](https://peps.python.org/pep-0544/)
