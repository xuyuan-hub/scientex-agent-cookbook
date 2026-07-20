# Scientex 从零构建方案

## 目标

从零开始，渐进式构建一个本地科学 agent 平台（与当前 scientex_agent 功能对等），
每一步都可以独立运行和验证，逐步叠加功能。

**核心理念**：
- 先用最简单的代码把功能跑通，再重构优化
- 每一步产出可运行的 demo，不是纸上谈兵
- 优先使用成熟框架（OpenAI SDK, mcp, LangGraph），避免重复造轮子
- 理解每一步"为什么"，而不只是"怎么做"

## 技术栈

| 组件 | 选型 | 说明 |
|---|---|---|
| LLM SDK | `openai >= 2.46` | 统一 OpenAI 兼容接口 |
| Agent 编排 | `langgraph >= 0.2` | 从手写循环逐步迁移到 StateGraph |
| MCP 协议 | `mcp >= 1.9` | 官方 SDK，不自研客户端 |
| Web 框架 | `fastapi >= 0.110` + `uvicorn` | 异步 HTTP + SSE 流式 |
| 存储 | SQLite（内嵌） + 文件系统 | 零配置，适合本地桌面应用 |
| 前端 | Vanilla JS SPA | 无框架依赖，单文件构建 |
| 包管理 | `uv` | 快速、现代、兼容 pip |

## 前置条件

- Python >= 3.11
- `uv` 已安装（`pip install uv`）
- 至少一个 LLM API Key（OpenAI / DeepSeek / Anthropic 任选其一）

## 开发路线图

```
Phase 1: 基础通信 ────────────────────────────────
  00  项目搭建           pyproject.toml, 目录结构, uv sync
  01  首次 LLM 调用      openai>=2.46, 一条消息, 拿到回复
  02  多轮对话           消息历史列表, 上下文延续
  03  流式输出           SSE streaming, token 级别的实时显示
                                │
Phase 2: 工具与技能 ─────────────────────────────
  04  工具调用           function calling, tool loop
  05  多提供商           Anthropic, Gemini, DeepSeek, Azure
  06  对话持久化         SQLite 存储消息, 可恢复
  07  MCP 集成           外部工具服务器, mcp SDK
  08  技能系统           文件型 SKILL.md 目录
  09  Python 内核        代码执行沙箱
                                │
Phase 3: Agent 编排 ─────────────────────────────
  10  Agent 循环         完整的 agent 编排, LangGraph StateGraph
                                │
Phase 4: 应用服务 ───────────────────────────────
  11  HTTP API           FastAPI 路由, SSE 端点
  12  项目管理           多项目/多帧/多会话
  13  CLI                argparse 命令行
                                │
Phase 5: 界面与领域 ─────────────────────────────
  14  Web 前端           SPA, 对话界面, 文件浏览器
  15  领域功能           生物工具, 化学编辑器, 记忆, 验证
  16  生产化             备份恢复, 错误处理, 打包分发
```

## 如何使用

### 按顺序阅读

每个文档独立可执行，建议按编号顺序阅读。每一步都有：

- **目标**：这一步要达成什么
- **前置条件**：需要先完成哪些步骤
- **设计思路**：为什么这么设计
- **代码实现**：完整的、可运行的代码
- **运行验证**：如何测试
- **当前局限**：这一步还没解决的问题（留给下一步）

### 运行代码

```bash
# 进入项目根目录
cd scientex_agent

# 每步完成后验证
uv run python -c "from scientex_agent.xxx import yyy; ..."

# 或运行测试
uv run python -m unittest discover -s tests
```

### 开发环境

```bash
# 创建虚拟环境并安装依赖
uv sync

# 激活环境
source .venv/bin/activate
```

## 架构演进

```
Step 01-03: 脚本式的函数调用
  main()
    └── openai.chat.completions.create()

Step 04-06: 引入类和状态
  ChatSession
    ├── messages: list
    ├── tools: list
    └── send() → response

Step 07-09: 引入外部能力
  MCPConnector, SkillCatalog, PythonKernel
    └── 通过 ToolRegistry 接入 ChatSession

Step 10: 引入编排框架
  AgentGraph (LangGraph)
    ├── llm_node
    ├── tool_node
    └── conditional edges

Step 11-16: 应用化
  FastAPI + CLI + Frontend + Domain Features
    └── 把 AgentGraph 包装为可部署的应用
```

## 项目结构

跟着教程走完，你会在 `src/scientex_agent/` 下逐步构建出以下模块：

```
scientex_agent/
├── src/scientex_agent/
│   ├── __init__.py          # 包入口，版本号
│   ├── __main__.py          # python -m scientex_agent
│   ├── cli.py               # 命令行界面
│   ├── llm_client.py        # LLM 对话会话 (Step 01-03)
│   ├── tool_registry.py     # 工具注册与调用 (Step 04)
│   ├── providers/           # 多提供商适配 (Step 05)
│   ├── storage.py           # SQLite 持久化 (Step 06)
│   ├── mcp_connector.py     # MCP 协议集成 (Step 07)
│   ├── skill_catalog.py     # 技能系统 (Step 08)
│   ├── python_kernel.py     # 代码执行沙箱 (Step 09)
│   ├── agent_graph.py       # LangGraph 编排 (Step 10)
│   ├── api.py               # FastAPI HTTP 服务 (Step 11)
│   ├── project_manager.py   # 多项目管理 (Step 12)
│   └── web/                 # 前端静态文件 (Step 14)
├── tests/                   # 测试用例
└── docs/tutorial/           # 本教程
```
