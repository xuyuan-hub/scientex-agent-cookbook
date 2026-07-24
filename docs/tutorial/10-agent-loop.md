# Step 10: 完整 Agent 编排（LangGraph）

## 目标

将 Provider、持久化会话、内置/MCP 工具、Skill 和 Python Kernel 组合为一个可恢复、可观测、可人工审批的 Agent 工作流。LangGraph 只负责编排和状态恢复；Step 05 的 provider adapter 仍是模型调用的唯一边界。

## 前置条件

- 完成 [09-python-kernel.md](09-python-kernel.md)
- 理解消息、工具和 Frame 的持久化模型

## 与前九步的边界

前九步分别证明了能力可以工作。本章不把它们重新实现为 LangChain 专用对象，也不把所有业务逻辑塞进一个 graph node。每个节点调用已有的领域服务：

```
Provider adapter  →  统一模型请求
Tool registry     →  统一工具 schema / 执行
MCPManager        →  连接生命周期与外部调用
SkillCatalog      →  按需读取、带版本的操作知识
PythonKernel      →  受控代码运行
MetadataStore     →  Project / Frame / Message
```

Graph 负责决定顺序、保存 checkpoint、在危险操作前暂停，并把状态变化输出为统一 run events。

## 设计思路

### 从循环到状态机

```
START → prepare → model ──无 tool call──→ persist → END
                  │
                  └─有 tool call→ policy → tools ─┬→ model
                                      │            │
                                      └─需审批→ interrupt
```

每条边都是显式契约。这样可以回答三个很实际的问题：

- 进程在工具执行后崩溃，是否能从最近成功状态恢复？
- 写文件或运行代码前，谁检查过权限、用户是否审批？
- 前端正在流式显示时，如何知道这是 token、工具开始、审批请求还是最终完成？

### 状态只保存可序列化业务数据

不要把 socket、SDK client、数据库连接或 Python coroutine 放进 graph state。它们属于应用依赖，由 node 在运行时取得；state 只保存可持久化的数据。

```python
# src/scientex_agent/agent_state.py
from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


class ToolIntent(TypedDict):
    id: str
    name: str
    arguments: dict[str, Any]
    risk: Literal["read", "write", "execute", "network"]


class AgentState(TypedDict):
    frame_id: str
    run_id: str
    messages: list[dict[str, Any]]
    selected_skill_ids: list[str]
    pending_tools: list[ToolIntent]
    approved_tool_ids: list[str]
    tool_rounds: int
    max_tool_rounds: int
    final_text: NotRequired[str]
    error: NotRequired[dict[str, str]]
```

`frame_id` 是产品层会话标识，`run_id` 是一次执行标识；两者不能混用。每个 node 只返回自己改变的字段，避免并发时用隐藏可变对象修改 state。

### 可恢复不等于可重放副作用

Checkpoint 可以恢复状态，却不能让外部世界自动回滚。因此每次有副作用的工具调用都需要 idempotency key：

```
run_id + tool_call_id → tool_execution_id
```

恢复时先查询该 execution 是否已有终态；已有则复用结构化结果，未开始才执行。对不可幂等操作（发邮件、下单、写外部数据库）必须经过审批，并在工具层提供业务去重。

## 实现

### 1. 添加编排与 checkpoint 依赖

```toml
# pyproject.toml
[project]
dependencies = [
    # ...
    "langgraph>=1.0",
    "langgraph-checkpoint-sqlite>=2.0",
]
```

生产环境使用与请求模式匹配的持久化 checkpointer；内存 checkpointer 只适合单元测试。SQLite 适合单机开发和单进程部署，多个 worker 共享写入时应迁移到支持并发的数据库。

### 2. 用节点封装业务操作

```python
# src/scientex_agent/agent_graph.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .agent_state import AgentState


@dataclass(frozen=True)
class AgentDependencies:
    app: "ScientexApp"
    policy: "ToolPolicy"
    events: "RunEventSink"


async def prepare(state: AgentState, deps: AgentDependencies) -> dict:
    frame = deps.app.get_frame_or_raise(state["frame_id"])
    messages = deps.app.load_prompt_messages(frame)
    await deps.events.emit("run.started", {"run_id": state["run_id"]})
    return {"messages": messages, "tool_rounds": 0, "max_tool_rounds": 8}


async def call_model(state: AgentState, deps: AgentDependencies) -> dict:
    response = await deps.app.provider_chat(
        frame_id=state["frame_id"],
        messages=state["messages"],
        skill_ids=state["selected_skill_ids"],
    )
    await deps.events.emit_model_response(response)
    return {
        "messages": [*state["messages"], response.message],
        "pending_tools": response.tool_intents,
        "final_text": response.text if not response.tool_intents else "",
    }


async def check_policy(state: AgentState, deps: AgentDependencies) -> dict:
    decisions = [deps.policy.decide(state["frame_id"], item)
                 for item in state["pending_tools"]]
    denied = [item.name for item, decision in zip(state["pending_tools"], decisions)
              if decision.kind == "deny"]
    if denied:
        return {"error": {"code": "tool_denied", "message": ", ".join(denied)}}

    approval = [item for item, decision in zip(state["pending_tools"], decisions)
                if decision.kind == "require_approval"]
    if approval:
        answer = interrupt({
            "kind": "tool_approval",
            "run_id": state["run_id"],
            "tools": approval,
        })
        approved = set(answer.get("approved_tool_ids", []))
        return {"approved_tool_ids": list(approved)}
    return {"approved_tool_ids": [item["id"] for item in state["pending_tools"]]}


async def execute_tools(state: AgentState, deps: AgentDependencies) -> dict:
    results = await deps.app.execute_idempotent_tools(
        frame_id=state["frame_id"],
        run_id=state["run_id"],
        tools=state["pending_tools"],
        approved_ids=set(state["approved_tool_ids"]),
    )
    await deps.events.emit_tool_results(results)
    tool_messages = [result.as_message() for result in results]
    return {
        "messages": [*state["messages"], *tool_messages],
        "pending_tools": [],
        "tool_rounds": state["tool_rounds"] + 1,
    }


async def persist(state: AgentState, deps: AgentDependencies) -> dict:
    await deps.app.persist_run(state)
    await deps.events.emit("run.completed", {
        "run_id": state["run_id"], "text": state.get("final_text", ""),
    })
    return {}
```

`ToolPolicy` 是普通、可单测的领域对象，而不是散落在 prompt 中的自然语言。对 `execute`、`write` 和跨信任边界的 `network` 操作，它默认要求审批；只读工具可以按项目策略自动通过。

### 3. 构建图和路由

```python
def route_after_model(state: AgentState) -> Literal["policy", "persist"]:
    if state.get("error") or not state["pending_tools"]:
        return "persist"
    if state["tool_rounds"] >= state["max_tool_rounds"]:
        return "persist"
    return "policy"


def route_after_policy(state: AgentState) -> Literal["tools", "persist"]:
    return "persist" if state.get("error") else "tools"


def build_graph(deps: AgentDependencies, checkpointer):
    graph = StateGraph(AgentState)
    graph.add_node("prepare", lambda state: prepare(state, deps))
    graph.add_node("model", lambda state: call_model(state, deps))
    graph.add_node("policy", lambda state: check_policy(state, deps))
    graph.add_node("tools", lambda state: execute_tools(state, deps))
    graph.add_node("persist", lambda state: persist(state, deps))

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "model")
    graph.add_conditional_edges("model", route_after_model)
    graph.add_conditional_edges("policy", route_after_policy)
    graph.add_edge("tools", "model")
    graph.add_edge("persist", END)
    return graph.compile(checkpointer=checkpointer)
```

实际项目中，node 函数可直接定义为 `async def` 并在图中注册；这里用闭包注入 `deps`，避免用模块级单例保存连接。检查点 config 始终携带稳定的 `thread_id=frame_id`，而一次请求使用独立 `run_id`：

```python
config = {"configurable": {"thread_id": frame_id}, "metadata": {"run_id": run_id}}
await graph.ainvoke(initial_state, config=config)
```

### 4. 定义前端/CLI 都能消费的运行事件

```python
# src/scientex_agent/run_events.py
from pydantic import BaseModel
from typing import Any, Literal


class RunEvent(BaseModel):
    run_id: str
    seq: int
    type: Literal[
        "run.started", "message.delta", "tool.started", "tool.completed",
        "approval.required", "run.completed", "run.failed",
    ]
    data: dict[str, Any]
```

事件首先是产品 API 契约，Step 11 再将它编码为 SSE。不要将 LangGraph 的内部 callback event 原样暴露给浏览器；内部格式升级不应破坏 Web 客户端。

## 验证

```bash
# 普通对话会产生 started、delta、completed 事件
uv run scientex_agent chat --frame FRAME_ID "解释 PCR 的退火温度"

# 需要写入或执行的工具会暂停，CLI 显示审批项
uv run scientex_agent chat --frame FRAME_ID "把分析结果保存成 CSV"

# 显示同一 Frame 的 checkpoint，确认可从中断处恢复
uv run scientex_agent runs inspect --frame FRAME_ID
```

至少测试以下情形：

- 无 tool call 的路径不会进入 policy/tools；
- 第九轮工具调用会停止并给出明确的 `max_tool_rounds` 错误；
- 用户拒绝审批后，不会调用工具；
- 同一 `run_id + tool_call_id` 恢复两次，只产生一次外部副作用；
- 在 `interrupt` 后以同一 `thread_id` 恢复，消息和审批状态完整保留。

## 深入理解

### 为什么不直接使用预构建 Agent

预构建 Agent 适合原型，但 Scientex 已有 provider adapter、MCP 风险属性、Artifact 审计和 Frame 数据模型。显式图把这些产品约束放在可见的位置，同时仍使用 LangGraph 的 checkpoint、streaming 和 interrupt 能力。

### 短期记忆与长期记忆

graph checkpoint 保存一个 Frame 内的短期工作状态；Step 15 的项目/用户记忆是跨 Frame 的长期知识。两者的保留期、权限和检索策略不同，不应混到同一 `messages` 列表。

## 当前局限

- 人工审批只有 CLI 交互；Web 端审批卡片在 Step 14 接入。
- 还没有可视化 trace、离线评测和指标采集；Step 16 加入可观测性与 CI。
- 单机 SQLite checkpointer 不适合作为多进程服务的共享协调器。

## 下一步

下一章将 `RunEvent`、Project 和 Frame 公开为版本化 HTTP API，并以 SSE 向客户端传输流式执行结果。
