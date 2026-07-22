# Python `unittest` 入门：为你的代码建立安全网

## 目标

学完后，你能够：

- 解释为什么要写单元测试；
- 运行项目中的一个测试文件，或运行全部测试；
- 读懂 `TestCase`、`test_...` 和常用断言；
- 用假对象（fake）测试需要网络或 API Key 的代码；
- 为一个新功能写出自己的第一个测试。

本项目使用 Python 标准库自带的 `unittest`，因此不需要额外安装测试框架。

## 为什么要使用单元测试？

写代码时可以手动运行程序，但手动验证有三个问题：

1. **容易漏步骤**：这次测了工具调用，下次可能忘记测流式输出。
2. **难以重复**：同一个 API 请求会受网络、余额和模型回复影响。
3. **重构不安心**：改一个 adapter，无法立刻知道是否破坏了旧会话或工具循环。

单元测试把“这段代码应该做什么”写成可重复运行的例子。每次改完代码，运行测试；如果测试仍然
通过，说明已被覆盖的行为没有改变。

在 Step 05 中，adapter 的职责是：

```text
ChatRequest → SDK 请求
SDK 响应    → ChatResponse / StreamEvent
```

这正适合单元测试：我们不访问真实 SDK 服务，而是准备一个可控的假 SDK 响应，检查转换是否正确。

## 测试的基本结构

一个 `unittest` 测试通常遵循 Arrange–Act–Assert（准备–执行–断言）：

```python
import unittest


class AddTests(unittest.TestCase):
    def test_adds_two_numbers(self) -> None:
        # Arrange：准备输入
        a, b = 2, 3

        # Act：执行要测试的行为
        result = a + b

        # Assert：确认结果
        self.assertEqual(result, 5)


if __name__ == "__main__":
    unittest.main()
```

这里有三条命名规则：

| 写法 | 含义 |
| --- | --- |
| `class ... (unittest.TestCase)` | 一组相关测试 |
| `def test_...` | 一条可被自动发现的测试 |
| `test_*.py` | 测试文件的常见名称，方便 `discover` 找到它 |

常用断言包括：

```python
self.assertEqual(actual, expected)       # 两个值相等
self.assertTrue(condition)                # 条件为真
self.assertIsNone(value)                  # 值是 None
self.assertIn(item, collection)           # 集合中包含某项
self.assertRaises(ValueError, func, arg)  # 调用会抛出指定异常
```

断言应描述外部可观察的行为，而不是绑定不重要的内部实现细节。

## 运行本项目的测试

在项目根目录执行：

```bash
# 只运行 OpenAI-compatible adapter 的测试
uv run python -B -m unittest tests.test_adapter -v

# 运行 tests/ 中的全部测试
uv run python -B -m unittest discover -s tests -v
```

`-v` 表示 verbose，会列出每个测试的方法名；`-B` 避免 Python 在源码目录写入 `.pyc` 文件。

看到下面的结果表示通过：

```text
Ran 5 tests in 0.0xxs

OK
```

如果失败，输出会显示失败的测试名、断言差异和对应行号。先读最后一段 traceback，再回到测试的
Arrange、Act、Assert 三部分，判断是代码行为错了，还是预期需要更新。

## 阅读 `tests/test_adapter.py`

[test_adapter.py](../../tests/test_adapter.py) 测试的是 Step 05 的
[OpenAICompatibleProvider](../../src/scientex_agent/providers/openai_compatible.py)。它不会读取
`DEEPSEEK_API_KEY`，也不会发送 HTTP 请求。

### 1. 用 fake 代替真实 SDK

真实代码会调用：

```python
client.chat.completions.create(...)
```

测试中的 `FakeOpenAIClient` 提供同样的属性路径，但返回预先准备好的对象：

```python
class FakeCompletions:
    def __init__(self, response, stream_chunks):
        self.response = response
        self.stream_chunks = stream_chunks
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.stream_chunks) if kwargs.get("stream") else self.response
```

这个 fake 只实现被测代码真正使用的最小接口。它的好处是：

- 每次返回完全相同的数据；
- 不需要 API Key、网络或付费额度；
- 可以检查 adapter 实际传给 SDK 的参数。

### 2. 测试一次普通对话转换

`test_chat_translates_request_and_response` 做了两类断言：

```python
# 检查 SDK 响应是否被正确翻译
self.assertEqual(response.content, "I will calculate that.")
self.assertEqual(response.usage.input_tokens, 11)

# 检查 adapter 是否发出了正确的 SDK 请求
self.assertEqual(
    client.completions.calls[0]["messages"],
    [{"role": "user", "content": "Hello"}],
)
```

第一类断言保护“响应解析”；第二类断言保护“请求转换”。二者缺一不可。

### 3. 测试流式 tool call 聚合

流式 API 会把工具调用拆到多个 chunk 中。例如：

```text
chunk 1: name = "add", arguments = '{"a": 2,'
chunk 2: arguments = ' "b": 3}'
```

`test_stream_accumulates_text_tool_calls_and_usage` 检查 adapter 最终是否还原为：

```python
ToolCall(id="call-1", name="add", arguments={"a": 2, "b": 3})
```

这类测试能防止重构时不小心漏掉某个 chunk 或把参数拼错。

## 如何为新功能添加测试

假设你要给 provider 增加一个 `supports_json_mode` 行为，可以按下面的顺序做：

1. 先写一个失败的 `test_...` 方法，描述想要的结果。
2. 运行这个单独的测试，确认它因为功能尚未实现而失败。
3. 实现最少的生产代码，让测试通过。
4. 运行全部测试，确认没有破坏既有行为。

示例骨架：

```python
class ProviderTests(unittest.TestCase):
    def test_json_mode_is_passed_to_sdk(self) -> None:
        # Arrange
        client = FakeOpenAIClient(...)
        provider = make_provider(client)

        # Act
        provider.chat(...)

        # Assert
        self.assertEqual(client.completions.calls[0]["response_format"], {...})
```

不要急着连接真实 API。先用 fake 明确 adapter 的契约；随后再把真实 API 命令作为少量的冒烟测试。

## 单元测试与冒烟测试的区别

| 类型 | 使用对象 | 目的 | 运行频率 |
| --- | --- | --- | --- |
| 单元测试 | fake SDK、固定输入 | 验证转换和业务逻辑 | 每次改代码 |
| 冒烟测试 | 真实 API Key、真实网络 | 验证配置和服务连通性 | 配置变更或发布前 |

两者都需要，但不能互相替代。Step 05 的 `tests/test_adapter.py` 是单元测试；例如
`uv run scientex_agent chat --provider deepseek "你好"` 是真实 API 冒烟测试。

## 练习

为 `OpenAICompatibleProvider` 再添加一条测试：当 SDK 返回无效 JSON 参数时，adapter 应返回
空字典 `{}`，而不是抛出 `JSONDecodeError`。提示：查看 `_parse_response()` 中已有的异常处理。

完成后回到 [Step 05：多提供商支持](05-multi-provider.md)，继续运行完整的 provider 迁移测试。
