# Scientex

从零开始，构建一个本地科学 AI Agent 平台。

## 这是什么？

Scientex 是一个运行在本地的 AI 科研助手 —— 支持多轮对话、工具调用、代码执行、MCP 协议、Web 界面。
这个项目不仅包含最终代码，更是一份**逐步教学方案**：每一步都可以独立运行，带你理解"为什么"而不仅是"怎么做"。

## 学习路线

```
Phase 1: 基础通信 (Steps 00-03)
  uv init 项目搭建 → 首次 LLM 调用 → 多轮对话 → 流式输出

Phase 2: 工具与技能 (Steps 04-09)
  工具调用 → 多提供商 → 对话持久化 → MCP 集成 → 技能系统 → Python 内核

Phase 3: Agent 编排 (Step 10)
  LangGraph 完整 agent 循环

Phase 4: 应用服务 (Steps 11-13)
  FastAPI HTTP API → 多项目管理 → CLI 命令行

Phase 5: 界面与领域 (Steps 14-16)
  Web 前端 → 领域功能 → 生产化交付
```

## 前置条件

- **Python >= 3.11**（推荐 3.13）
- **[uv](https://docs.astral.sh/uv/)** 包管理器（`pip install uv`）
- 至少一个 **LLM API Key**（DeepSeek / OpenAI / Anthropic 任选其一）

## 快速开始

```bash
# 1. 用 uv 初始化项目
uv init --package scientex-agent

# 2. 同步环境
uv sync

# 3. 打开教程，从第 00 步开始
# docs/tutorial/00-project-setup.md
```

或者直接阅读 → **[教程入口](docs/tutorial/README.md)**

## 每一步的结构

每篇教程都遵循统一格式：

| 章节 | 内容 |
|---|---|
| **目标** | 这一步要达成什么 |
| **前置条件** | 依赖哪些前置步骤 |
| **设计思路** | 为什么这么设计，对比优劣 |
| **代码实现** | 完整的、可运行的代码 |
| **运行验证** | 如何测试效果 |
| **当前局限** | 未解决的问题（留给下一步） |

## 技术栈

| 组件 | 选型 |
|---|---|
| LLM SDK | `openai`（统一 OpenAI 兼容接口） |
| Agent 编排 | `langgraph`（从手写循环演进到 StateGraph） |
| MCP 协议 | `mcp` 官方 SDK |
| Web 框架 | `fastapi` + `uvicorn` |
| 存储 | SQLite + 文件系统 |
| 前端 | Vanilla JS SPA |
| 包管理 | `uv` |

## License

MIT
