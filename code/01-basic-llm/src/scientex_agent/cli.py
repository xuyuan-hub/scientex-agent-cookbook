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