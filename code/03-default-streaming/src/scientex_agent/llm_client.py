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

    def send_stream(self,content:str,**kwargs)->str:
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

        stream_kwargs = {
            "stream": True,
            "stream_options":{"include_usage":True},
            **kwargs,
        }
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            **stream_kwargs,
        )
        full_reply = ""
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                full_reply += delta.content
                yield delta.content
        self.messages.append({"role": "assistant", "content": full_reply})

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