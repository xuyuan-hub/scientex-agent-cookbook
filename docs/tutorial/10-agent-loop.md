# Step 10: 完整 Agent 编排（LangGraph）

## 目标

将之前各步骤的能力（对话、工具、MCP、技能、内核）整合为一个基于 LangGraph 的完整 Agent 编排系统。

## 前置条件

- 完成 [09-python-kernel.md](09-python-kernel.md)
- 理解前 9 步的所有概念

## 设计思路

### 为什么要用 LangGraph

在第 4 步中，我们手写了一个简单的 tool-calling 循环。但随着系统变复杂，手写循环面临：

```python
# 手写循环的复杂性（当前 scientex 的 conversation_execution.py 有 512 行）
for _round in range(max_rounds):
    response = provider.chat(request)
    if tool_calls:
        for call in tool_calls:
            result = execute_tool(call)
            # 还需要处理：
            # - 流式输出中的 tool call 聚合
            # - 错误恢复
            # - 消息裁剪/压缩
            # - DMIL fallback（文本嵌入式工具调用）
            # - 中断和恢复
            # ...
```

LangGraph 把这些复杂性建模为**有向图**：

```
         ┌──────────┐
         │  START   │
         └────┬─────┘
              │
         ┌────▼─────┐     有 tool_calls      ┌──────────┐
         │   LLM    │───────────────────────→│  TOOLS   │
         │   Node   │                        │   Node   │
         └────┬─────┘←───────────────────────└──────────┘
              │ 无 tool_calls
         ┌────▼─────┐
         │   END    │
         └──────────┘
```

### LangGraph 核心概念

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 1. 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 2. 定义节点（函数）
def call_model(state: AgentState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 3. 定义路由（条件边）
def should_continue(state: AgentState):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END

# 4. 构建图
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))
graph.add_edge("tools", "agent")       # 执行完工具后回到 agent
graph.add_conditional_edges("agent", should_continue)
graph.set_entry_point("agent")

# 5. 编译（可加 checkpointer）
app = graph.compile(checkpointer=MemorySaver())
```

### 本次 Agent 的图结构

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                    ┌────▼─────┐
                    │  build   │  构建 system prompt + 消息历史
                    │  prompt  │
                    └────┬─────┘
                         │
                    ┌────▼─────┐     有 tool_calls     ┌─────────┐
                    │   LLM    │──────────────────────→│  tools  │
                    └────┬─────┘                       └────┬────┘
                         │ 无 tool_calls                   │
                    ┌────▼─────┐                       ←───┘
                    │  record  │  记录消息到 metadata
                    │  message │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │   END    │
                    └──────────┘
```

## 实现

### 1. 添加依赖

```toml
# pyproject.toml 新增
dependencies = [
    ...
    "langgraph>=0.2.0",
    "langchain-core>=0.3.0",
    "langchain-openai>=0.2.0",
]
```

```bash
uv sync
```

### 2. 创建 src/scientex_agent/agent_graph.py

```python
"""LangGraph-based agent orchestration.

Replaces the hand-written agent loop with a declarative StateGraph.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .provider_types import ChatMessage


# === Agent State ===

class AgentState(TypedDict):
    """State that flows through the agent graph."""
    messages: Annotated[list[BaseMessage], add_messages]
    system_prompt: str | None
    frame_id: str | None
    model: str


# === Tool Conversion ===

def convert_tools_for_langgraph(registry) -> list[BaseTool]:
    """Convert our ToolRegistry tools to LangGraph-compatible tools.

    This bridges the gap between our existing tool system and LangGraph's tool system.
    """
    from langchain_core.tools import StructuredTool
    from pydantic import create_model
    import json

    tools = []
    for schema in registry.schemas():
        func = schema["function"]
        name = func["name"]
        description = func.get("description", "")
        params = func.get("parameters", {"type": "object", "properties": {}})

        # Create a simple wrapper
        def make_handler(n):
            def handler(**kwargs) -> str:
                result = registry.execute(n, kwargs)
                return json.dumps(result, ensure_ascii=False)
            return handler

        # Build pydantic model from JSON Schema (simplified)
        fields = {}
        for prop_name, prop_info in params.get("properties", {}).items():
            prop_type = str
            if prop_info.get("type") == "integer":
                prop_type = int
            elif prop_info.get("type") == "number":
                prop_type = float
            elif prop_info.get("type") == "boolean":
                prop_type = bool
            required = prop_name in params.get("required", [])
            if required:
                fields[prop_name] = (prop_type, ...)
            else:
                fields[prop_name] = (prop_type, None)

        tool_model = create_model(f"Args_{name}", **fields) if fields else None
        tool = StructuredTool.from_function(
            func=make_handler(name),
            name=name,
            description=description,
            args_schema=tool_model,
        )
        tools.append(tool)

    return tools


# === Agent Graph ===

class AgentGraph:
    """LangGraph-based agent that orchestrates LLM + tools.

    Usage::

        graph = AgentGraph(llm=llm_client, tools=[...])
        app = graph.compile()

        # Non-streaming
        result = await app.ainvoke({"messages": [HumanMessage(content="Hello")]})

        # Streaming
        async for event in app.astream_events({"messages": [...]}, version="v2"):
            ...
    """

    def __init__(
        self,
        *,
        llm: ChatOpenAI,
        tools: list[BaseTool] | None = None,
        checkpointer: Any = None,
        max_tool_rounds: int = 30,
        max_message_tokens: int = 8000,
    ) -> None:
        self.llm = llm.bind_tools(tools or [])
        self.tools = tools or []
        self.checkpointer = checkpointer or MemorySaver()
        self.max_tool_rounds = max_tool_rounds
        self.max_message_tokens = max_message_tokens
        self._graph = self._build()

    def _build(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        # LLM node
        async def call_model(state: AgentState) -> dict:
            messages = list(state.get("messages", []))

            # Prepend system prompt if present
            if state.get("system_prompt"):
                # Only prepend if the first message isn't already a system message
                if not messages or not isinstance(messages[0], SystemMessage):
                    messages.insert(0, SystemMessage(content=state["system_prompt"]))

            # Trim messages to fit context
            trimmed = trim_messages(
                messages,
                max_tokens=self.max_message_tokens,
                strategy="last",
                token_counter=self.llm,
                include_system=True,
                start_on="human",
                allow_partial=False,
            )

            response = await self.llm.ainvoke(trimmed)
            return {"messages": [response]}

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))

        # Routing
        def route_after_agent(state: AgentState) -> str:
            messages = state.get("messages", [])
            if not messages:
                return END
            last = messages[-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"
            return END

        workflow.add_conditional_edges("agent", route_after_agent)
        workflow.add_edge("tools", "agent")
        workflow.set_entry_point("agent")

        return workflow

    def compile(self, **kwargs):
        """Compile the graph into a runnable app."""
        return self._graph.compile(checkpointer=self.checkpointer, **kwargs)

    async def run(
        self,
        messages: list[BaseMessage],
        *,
        system_prompt: str | None = None,
        thread_id: str | None = None,
    ) -> dict:
        """Non-streaming agent run.

        Args:
            messages: Initial messages.
            system_prompt: Optional system prompt.
            thread_id: Thread ID for checkpointing (enables conversation continuity).

        Returns:
            Final graph state.
        """
        app = self.compile()
        config = {"configurable": {"thread_id": thread_id or "default"}} if thread_id else {}
        initial_state: AgentState = {
            "messages": messages,
            "system_prompt": system_prompt,
            "frame_id": None,
            "model": "",
        }
        return await app.ainvoke(initial_state, config)  # type: ignore[arg-type]

    async def stream(
        self,
        messages: list[BaseMessage],
        *,
        system_prompt: str | None = None,
        thread_id: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Streaming agent run.

        Yields events as they occur:
        - {"token": "text"} — text token from LLM
        - {"tool_start": {"name": "..."}} — tool execution begins
        - {"tool_end": {"name": "...", "result": "..."}} — tool execution ends
        - {"done": True} — agent finished
        """
        app = self.compile()
        config = {"configurable": {"thread_id": thread_id or "default"}} if thread_id else {}
        initial_state: AgentState = {
            "messages": messages,
            "system_prompt": system_prompt,
            "frame_id": None,
            "model": "",
        }

        async for event in app.astream_events(initial_state, config, version="v2"):  # type: ignore[arg-type]
            kind = event.get("event", "")

            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield {"token": content}

            elif kind == "on_tool_start":
                yield {
                    "tool_start": {
                        "name": event.get("name", ""),
                        "input": event["data"].get("input", {}),
                    }
                }

            elif kind == "on_tool_end":
                yield {
                    "tool_end": {
                        "name": event.get("name", ""),
                        "output": event["data"].get("output", ""),
                    }
                }

        yield {"done": True}


# === Helper: Convert internal messages to LangChain format ===

def to_langchain_messages(messages: list[ChatMessage]) -> list[BaseMessage]:
    """Convert our internal ChatMessage list to LangChain format."""
    result = []
    for msg in messages:
        if msg.role == "system":
            result.append(SystemMessage(content=msg.content))
        elif msg.role == "user":
            result.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            ai_msg = AIMessage(content=msg.content)
            if msg.tool_calls:
                from langchain_core.messages import ToolCall as LCToolCall
                ai_msg.tool_calls = [
                    LCToolCall(
                        id=tc.id,
                        name=tc.name,
                        args=tc.arguments,
                    )
                    for tc in msg.tool_calls
                ]
            result.append(ai_msg)
        elif msg.role == "tool":
            result.append(ToolMessage(
                content=msg.content,
                tool_call_id=msg.tool_call_id or "",
            ))
    return result
```

### 3. 整合：创建 ScientexApp 门面

```python
# src/scientex_agent/app.py

"""Application facade — wires all services together."""

from __future__ import annotations

from pathlib import Path

from langchain_openai import ChatOpenAI

from .agent_graph import AgentGraph, convert_tools_for_langgraph
from .metadata import MetadataStore
from .skill_catalog import SkillCatalog
from .tools import ToolRegistry, register_default_tools, register_skill_tools
from .kernel import SubprocessPythonKernel
from .provider_registry import ProviderRegistry


class ScientexApp:
    """Main application — composition root and facade.

    Usage::

        app = ScientexApp(Path("~/.scientex_agent"))
        app.initialize()

        # Create a project
        project = app.create_project(name="My Research")

        # Create a frame (conversation)
        frame = app.create_frame(project_id=project.id, name="Chat 1")

        # Chat
        result = await app.chat(frame_id=frame.id, content="Hello!")
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.metadata = MetadataStore(self.data_dir / "app.db")
        self.skills = SkillCatalog()
        self.kernel: SubprocessPythonKernel | None = None
        self._graph_cache: dict[str, AgentGraph] = {}

    def initialize(self) -> None:
        """One-time setup: create directories, init DB, load skills."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata.initialize()
        self.skills.load()

    # === Projects & Frames ===

    def create_project(self, *, name: str, description: str = "", context: str = ""):
        return self.metadata.create_project(name=name, description=description, context=context)

    def list_projects(self):
        return self.metadata.list_projects()

    def create_frame(self, *, project_id: str, name: str = ""):
        return self.metadata.create_frame(project_id=project_id, name=name)

    def list_frames(self, project_id: str):
        return self.metadata.list_frames(project_id)

    # === Agent ===

    def _build_graph(self, frame_id: str, provider_name: str, model_name: str) -> AgentGraph:
        """Build or retrieve a cached AgentGraph for a frame."""
        cache_key = f"{frame_id}:{provider_name}:{model_name}"
        if cache_key in self._graph_cache:
            return self._graph_cache[cache_key]

        # Build tools
        registry = ToolRegistry()
        register_default_tools(registry)
        register_skill_tools(registry, self.skills)
        # MCP tools would be registered here if connected

        langgraph_tools = convert_tools_for_langgraph(registry)

        # Build LLM
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            streaming=True,
        )

        # Build graph
        graph = AgentGraph(llm=llm, tools=langgraph_tools)
        self._graph_cache[cache_key] = graph
        return graph

    async def chat(self, *, frame_id: str, content: str) -> dict:
        """Send a message and get the full response."""
        frame = self.metadata.get_frame(frame_id)
        if not frame:
            raise KeyError(f"Unknown frame: {frame_id}")

        # Build system prompt
        system_prompt = self._build_system_prompt(frame.project_id)
        self.skills.load()
        system_prompt += "\n\n" + self.skills.system_prompt_index()

        # Record user message
        self.metadata.append_message(frame_id=frame_id, role="user", content=content)

        # Load history
        history = self.metadata.list_messages(frame_id)
        from .agent_graph import to_langchain_messages
        messages = to_langchain_messages(history)
        # Add current message (it's already in history, but we'll let the graph handle it)

        # Build and run agent
        graph = self._build_graph(frame_id, "openai", "gpt-4o")
        result = await graph.run(
            messages=messages,
            system_prompt=system_prompt,
            thread_id=frame_id,
        )

        # Extract and save assistant response
        final_messages = result.get("messages", [])
        if final_messages:
            last = final_messages[-1]
            if hasattr(last, "content") and last.content:
                self.metadata.append_message(
                    frame_id=frame_id, role="assistant", content=last.content
                )

        return {"content": last.content if hasattr(last, "content") else ""}

    async def chat_stream(self, *, frame_id: str, content: str):
        """Streaming chat — yields events as they occur."""
        # Similar to chat() but uses graph.stream()
        frame = self.metadata.get_frame(frame_id)
        if not frame:
            raise KeyError(f"Unknown frame: {frame_id}")

        system_prompt = self._build_system_prompt(frame.project_id)
        self.skills.load()
        system_prompt += "\n\n" + self.skills.system_prompt_index()

        self.metadata.append_message(frame_id=frame_id, role="user", content=content)

        history = self.metadata.list_messages(frame_id)
        from .agent_graph import to_langchain_messages
        messages = to_langchain_messages(history)

        graph = self._build_graph(frame_id, "openai", "gpt-4o")
        full_content = ""

        async for event in graph.stream(
            messages=messages,
            system_prompt=system_prompt,
            thread_id=frame_id,
        ):
            yield event
            if "token" in event:
                full_content += event["token"]

        # Save final response
        if full_content:
            self.metadata.append_message(
                frame_id=frame_id, role="assistant", content=full_content
            )

    # === Helpers ===

    def _build_system_prompt(self, project_id: str) -> str:
        """Build the system prompt for a project."""
        project = self.metadata.get_project(project_id)
        project_context = project.context if project else ""
        return (
            "You are a scientific research assistant powered by Scientex.\n"
            "You can use tools to search literature, analyze data, and "
            "execute Python code.\n"
            "Always cite your sources and explain your reasoning.\n"
            "\n"
            f"Project context: {project_context}\n"
        )
```

## 验证

```python
import asyncio
from pathlib import Path
from scientex_agent.app import ScientexApp

async def main():
    app = ScientexApp(Path("/tmp/scientex_agent-test"))
    app.initialize()

    project = app.create_project(name="Test", description="Testing agent graph")
    frame = app.create_frame(project_id=project.id, name="Chat")

    # Test non-streaming
    result = await app.chat(frame_id=frame.id, content="What is 2+2?")
    print(f"Response: {result['content'][:200]}")

    # Test streaming
    print("Streaming: ", end="")
    async for event in app.chat_stream(frame_id=frame.id, content="Count from 1 to 5"):
        if "token" in event:
            print(event["token"], end="", flush=True)
        elif "tool_start" in event:
            print(f"\n[Tool: {event['tool_start']['name']}]", end="")
        elif "done" in event:
            print("\n[Done]")

asyncio.run(main())
```

## 深入理解

### LangGraph vs 手写循环

| 方面 | 手写循环 | LangGraph |
|---|---|---|
| 控制流 | 显式 for/if/break | 声明式图 |
| 流式 | 手动 yield | `.astream_events()` |
| 状态管理 | 手动管理变量 | TypedDict + reducer |
| 断点/恢复 | 需要自己实现 | `checkpointer` 开箱即用 |
| 错误恢复 | try/except 散落各处 | 可加 error node |
| 可视化 | 无 | `graph.get_graph().draw_mermaid()` |
| 测试 | 需要 mock 整个循环 | 可单独测试每个 node |

### 为什么绑定工具到 LLM

```python
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)
```

`bind_tools` 做了两件事：
1. 把 tool schemas 注入到每次 API 调用的 `tools` 参数
2. 自动解析响应中的 `tool_calls` 为 LangChain `ToolCall` 对象

### Checkpointer 的作用

```python
app = graph.compile(checkpointer=MemorySaver())

# 第一次调用
result = await app.ainvoke({"messages": [HumanMessage(content="我叫张三")]},
                            config={"configurable": {"thread_id": "conv-1"}})

# 第二次调用（自动加载之前的状态）
result = await app.ainvoke({"messages": [HumanMessage(content="我叫什么？")]},
                            config={"configurable": {"thread_id": "conv-1"}})
# → LLM 知道用户叫张三！
```

`thread_id` 是对话的标识符，`MemorySaver` 在内存中保存状态。
可以换成 `SqliteSaver` 实现持久化。

## 当前局限

1. `MemorySaver` 在内存中——重启后丢失。生产环境应换 `SqliteSaver`
2. 没有 human-in-the-loop（中断等待用户确认）
3. 没有子图（delegate 子任务）

## 下一步

→ [11-http-api.md](11-http-api.md)
