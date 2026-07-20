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