"""Simple tool-calling agent loop."""
from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

from openai import OpenAI

from .tools import ToolRegistry

def _execute_tool_calls(
    response: Any,
    registry: ToolRegistry,
) -> list[dict[str,Any]]:
    """Execute all tool calls in a response and return tool result messages."""
    tool_messages = []
    for tc in response.choices[0].message.tool_calls or []:
        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        result = registry.execute(tc.function.name,args)
        tool_messages.append({
            "role":"tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result, ensure_ascii=False)
        })
    return tool_messages

def chat_with_tools(
    client: OpenAI,
    model: str,
    messages: list[dict],
    registry: ToolRegistry,
    *,
    max_rounds: int = 10,
    **kwargs,
) -> str:
    """Non-streaming chat with tool calling.

    Args:
        client: OpenAI client.
        model: Model name.
        messages: Initial messages (system + history + new user message).
        registry: ToolRegistry with registered tools.
        max_rounds: Maximum LLM → tool → LLM rounds.
        **kwargs: Passed to client.chat.completions.create.

    Returns:
        Final assistant text response.
    """
    schemas = registry.schemas()
    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools = schemas or None,
            **kwargs,
        )
        choice = response.choices[0]
        msg = choice.message

        # No tool calls -> done
        if not msg.tool_calls:
            return msg.content or ""
        
        # Store assistant message with tool calls
        messages.append({
            "role":"assistant",
            "content":msg.content,
            "tool_calls":[
                {
                    "id":tc.id,
                    "type": "function",
                    "function":{
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute tools and append results
        tool_results = _execute_tool_calls(response,registry)
        messages.extend(tool_results)
    
    return messages[-1].get("content","") if messages else ""

def chat_with_tools_stream(
    client: OpenAI,
    model: str,
    messages: list[dict],
    registry: ToolRegistry,
    *,
    max_rounds: int =10,
    **kwargs,
) -> Generator[str,None,None]:
    """Streaming chat with tool calling.

    Yields text tokens. Tool calls are executed internally (not yielded).
    """
    schemas = registry.schemas()

    for _ in range(max_rounds):
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=schemas or None,
            stream=True,
            stream_options={"include_usage":True},
            **kwargs,
        )

        # Accumulate streaming response
        full_content = ""
        tool_calls_data: dict[int, dict] = {}

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Text
            if delta.content:
                full_content += delta.content
                yield delta.content

            # Tool calls (accumulate across chunks)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {
                            "id": tc_delta.id or "",
                            "function": {"name":"","arguments":""},
                            "type":"function"
                        }
                    if tc_delta.id:
                        tool_calls_data[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_data[idx]["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_data[idx]["function"]["arguments"] += tc_delta.function.arguments

        # If no tool calls, we're done
        if not tool_calls_data:
            return
        
        # Store assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": full_content or None,
            "tool_calls": list(tool_calls_data.values()),
        })

        # Execute tools and append results
        for tc_data in tool_calls_data.values():
            args = json.loads(tc_data["function"]["arguments"]) if tc_data["function"]["arguments"] else {}
            result = registry.execute(tc_data["function"]["name"], args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc_data["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })

    return