"""No-network regression tests for the Step 05 provider migration."""

from __future__ import annotations

import unittest
from collections.abc import Iterator, Sequence

from scientex_agent.llm_client import ChatSession
from scientex_agent.provider_types import (
    ChatRequest,
    ChatResponse,
    ModelSpec,
    StreamEvent,
    ToolCall,
)
from scientex_agent.tools import ToolRegistry


class FakeProvider:
    """A deterministic provider that asks for a tool once, then answers."""

    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def list_models(self) -> Sequence[ModelSpec]:
        return (ModelSpec("fake", "fake-1", 1_000, supports_tools=True),)

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if request.tools and not any(message.role == "tool" for message in request.messages):
            return ChatResponse(tool_calls=(ToolCall("call-1", "add", {"a": 2, "b": 3}),))
        return ChatResponse(content="result: 5")

    def chat_stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        response = self.chat(request)
        if response.tool_calls:
            yield StreamEvent(response=response)
            return
        yield StreamEvent(text="result: ")
        yield StreamEvent(text="5")
        yield StreamEvent(response=response)


def tool_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register(
        name="add",
        description="Add two integers.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    )
    def add(a: int, b: int) -> int:
        return a + b

    return registry


class ProviderIntegrationTests(unittest.TestCase):
    def test_normal_session_uses_provider(self) -> None:
        provider = FakeProvider()
        session = ChatSession(provider=provider, model="fake-1")

        self.assertEqual(session.send("hello"), "result: 5")
        self.assertEqual([message["role"] for message in session.messages], ["user", "assistant"])
        self.assertEqual(provider.requests[0].messages[0].content, "hello")

    def test_tool_loop_uses_provider_neutral_messages(self) -> None:
        provider = FakeProvider()
        session = ChatSession(provider=provider, model="fake-1", registry=tool_registry())

        self.assertEqual(session.send_with_tools("2 + 3"), "result: 5")
        self.assertEqual([message["role"] for message in session.messages], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(provider.requests[1].messages[-1].role, "tool")

    def test_streaming_tool_loop_is_preserved(self) -> None:
        provider = FakeProvider()
        session = ChatSession(provider=provider, model="fake-1", registry=tool_registry())

        self.assertEqual("".join(session.send_with_tools_stream("2 + 3")), "result: 5")
        self.assertEqual([message["role"] for message in session.messages], ["user", "assistant", "tool", "assistant"])

if __name__ == "__main__":
    unittest.main()
