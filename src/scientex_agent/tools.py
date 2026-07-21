"""Tool definitions and execution for function calling."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

# === Tool Schema Definitions ===

def tool_schema(
    name:str,
    description:str,
    parameters:dict[str,Any],
)->dict[str,Any]:
    """Create an OpenAI-compatible tool schema.
    
    Args:
        name: Tool name. Use snake_case. Must match the registered handler.
        description: What the tool does. LLM uses this to decide when to call it.
        parameters: JSON Schema for the arguments.

    Returns:
        A tool definition dict ready to pass to the LLM.
    """
    return {
        "type":"function",
        "function":{
            "name":name,
            "description":description,
            "parameters":parameters
        }
    }

# === Tool Registry ===

class ToolRegistry:
    """A simple registry mapping tool names to handler functions.

    Usage::

        registry = ToolRegistry()

        @registry.register("add", "Add two numbers", {...})
        def add(a: int, b: int) -> int:
            return a + b

        # Get OpenAI-compatible schemas
        schemas = registry.schemas()

        # Execute a tool call from the LLM
        result = registry.execute("add", {"a": 1, "b": 2})
    """

    def __init__(self):
        self._handlers: dict[str,Callable] = {}
        self._schemas: list[dict[str,Any]] = []

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any]
    ):
        """Decorator to register a function as a tool.

        The decorated function becomes the handler. Its name must match `name`.
        """

        def decorator(func: Callable) -> Callable:
            self._handlers[name] = func
            self._schemas.append(tool_schema(name,description,parameters))
            return func
        
        return decorator
    
    def schemas(self) -> list[dict[str,Any]]:
        """Return OpenAI-compatible tool schemas for all registered tools."""
        return list(self._schemas)
    
    def execute(self,name:str, arguments:dict[str,Any]) -> Any:
        """Execute a tool and return the result.

        Result is a dict suitable for a 'tool' role message content.
        """
        handler = self._handlers.get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            result = handler(**arguments)
            return {"ok":True, "result": result}
        except Exception as exc:
            return {"ok":False, "error": str(exc)}
    
    def has(self,name:str) -> bool:
        return name in self._handlers
    
# === Built-in Tools ===

def registry_default_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """Register a set of basic utility tools and return the registry.

    If *registry* is ``None`` a fresh :class:`ToolRegistry` is created.
    """
    if registry is None:
        registry = ToolRegistry()

    @registry.register(
        name="get_current_time",
        description="Get the current date and time in a given timezone.",
        parameters={
            "type":"object",
            "properties":{
                "timezone":{
                    "type":"string",
                    "description":"IANA timezone, e.g. 'Asia/Shanghai', 'America/New_York'"
                }
            },
            "required":["timezone"]
        },
    )
    def get_current_time(timezone:str) -> str:
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo

        try:
            tz = ZoneInfo(timezone)
            now = datetime.now(tz)
            return now.strftime("%Y-%m-%d %H:%M:%S%Z")
        except Exception:
            # Fallback to UTC
            return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%Z")
        
    @registry.register(
        name="caculate",
        description="Evaluate a mathematical expression. Supports +,-,*,/,** and common math functions.",
        parameters={
            "type":"object",
            "properties":{
                "expression":{
                    "type":"string",
                    "description":"A Python math expression, e.g. '2 + 3 * 4' or 'sqrt(16)'"
                },
            },
            "required":["expression"]
        }
    )
    def caculate(expression:str) -> float |str:
        import math

        # Safe eval: only allow math functions, numbers, and basic operators
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e,
        }
        try:
            result = eval(expression,{"__builtin__":{}},allowed_names)
            return result
        except Exception as e:
            return f"Error: {e}"
    return registry