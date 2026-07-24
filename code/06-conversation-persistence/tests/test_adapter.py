"""Unit tests for the OpenAI-compatible provider adapter.

These tests use a small fake SDK client.  They never read API keys or make a
network request, so they are safe to run on every code change.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from scientex_agent.provider_types import ChatMessage, ChatRequest, ModelSpec
from scientex_agent.providers.openai_compatible import OpenAICompatibleProvider


def ns(**values: Any) -> SimpleNamespace:
    """Keep fake SDK objects readable in the tests below."""

    return SimpleNamespace(**values)


class FakeCompletions:
    def __init__(self, response: Any, stream_chunks: list[Any]) -> None:
        self.response = response
        self.stream_chunks = stream_chunks
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return iter(self.stream_chunks) if kwargs.get("stream") else self.response


class FakeOpenAIClient:
    def __init__(self, response: Any, stream_chunks: list[Any]) -> None:
        self.completions = FakeCompletions(response, stream_chunks)
        self.chat = ns(completions=self.completions)


def make_provider(client: FakeOpenAIClient) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name="fake-openai",
        base_url="https://example.invalid/v1",
        api_key="not-a-real-key",
        models=(ModelSpec("fake-openai", "fake-1", 1_000, supports_tools=True),),
        client=client,
    )


class OpenAICompatibleProviderTests(unittest.TestCase):
    def test_chat_translates_request_and_response(self) -> None:
        sdk_response = ns(
            choices=[
                ns(
                    message=ns(content="I will calculate that.", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=ns(prompt_tokens=11, completion_tokens=5),
        )
        client = FakeOpenAIClient(sdk_response, [])
        provider = make_provider(client)

        response = provider.chat(
            ChatRequest(
                model="fake-1",
                messages=(ChatMessage(role="user", content="Hello"),),
                temperature=0.2,
                max_tokens=40,
            )
        )

        self.assertEqual(response.content, "I will calculate that.")
        self.assertEqual(response.usage.input_tokens, 11)
        self.assertEqual(response.usage.output_tokens, 5)
        self.assertEqual(
            client.completions.calls[0]["messages"],
            [{"role": "user", "content": "Hello"}],
        )
        self.assertEqual(client.completions.calls[0]["temperature"], 0.2)

    def test_stream_accumulates_text_tool_calls_and_usage(self) -> None:
        tool_delta_1 = ns(
            index=0,
            id="call-1",
            function=ns(name="add", arguments='{"a": 2,'),
        )
        tool_delta_2 = ns(
            index=0,
            id=None,
            function=ns(name=None, arguments=' "b": 3}'),
        )
        chunks = [
            ns(choices=[ns(delta=ns(content="The answer is ", tool_calls=[]), finish_reason=None)], usage=None),
            ns(choices=[ns(delta=ns(content=None, tool_calls=[tool_delta_1]), finish_reason=None)], usage=None),
            ns(choices=[ns(delta=ns(content=None, tool_calls=[tool_delta_2]), finish_reason="tool_calls")], usage=None),
            ns(choices=[], usage=ns(prompt_tokens=8, completion_tokens=4)),
        ]
        client = FakeOpenAIClient(response=ns(), stream_chunks=chunks)
        provider = make_provider(client)

        events = list(
            provider.chat_stream(
                ChatRequest(
                    model="fake-1",
                    messages=(ChatMessage(role="user", content="2 + 3"),),
                )
            )
        )

        self.assertEqual([event.text for event in events if event.text], ["The answer is "])
        final = events[-1].response
        self.assertIsNotNone(final)
        assert final is not None  # Help static type checkers after the assertion above.
        self.assertEqual(final.tool_calls[0].name, "add")
        self.assertEqual(final.tool_calls[0].arguments, {"a": 2, "b": 3})
        self.assertEqual(final.usage.input_tokens, 8)
        self.assertTrue(client.completions.calls[0]["stream"])


if __name__ == "__main__":
    unittest.main()
