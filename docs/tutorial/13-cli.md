# Step 13: 命令行界面（CLI）

## 目标

提供适合人和自动化脚本的命令行入口：类型化参数、可预测的退出码、结构化 JSON 输出和明确的交互边界。CLI 与 HTTP API 共用 application service，不复制业务逻辑。

## 前置条件

- 完成 [12-project-management.md](12-project-management.md)
- 熟悉 Python 类型注解与 shell 基础

## 设计思路

### CLI 的职责

```
shell 参数 / stdin
   → Typer：解析、帮助、类型和退出码
   → command handler：选择显示格式、获取交互确认
   → Application service：实际业务
   → stdout（结果） / stderr（诊断）
```

CLI 负责体验，不负责重写服务层。`scientex_agent projects create` 和 `POST /api/v1/projects` 都调用同一个
`ProjectService.create()`；这能让单元测试覆盖真实规则，也避免 CLI 与 Web 对同一功能产生不同语义。

### 命令树

```
scientex_agent
├── init
├── chat
├── projects   list | create | export | verify-export
├── frames     list | create
├── artifacts  list | put | history | get
├── tools      list | call
├── skills     list | validate | show
├── mcp        doctor
├── runs       inspect | resume
└── serve
```

所有可能被脚本消费的命令均提供 `--format json`；默认表格/富文本只服务于终端阅读。成功写 stdout，警告和错误写 stderr，因此 `scientex ... --format json | jq` 不会被进度文本污染。

### 退出码和交互原则

| 退出码 | 含义 |
|---:|---|
| 0 | 成功 |
| 2 | 参数或输入校验错误 |
| 3 | 不存在、冲突或业务规则拒绝 |
| 4 | 用户拒绝审批 / 取消 |
| 5 | 外部服务、网络或执行失败 |
| 70 | 未预期内部错误 |

具有副作用的操作默认要求确认。自动化必须显式传 `--yes`；当 stdin 不是 TTY 且没有 `--yes` 时，命令失败而非偷偷执行。

## 实现

### 1. 添加 CLI 依赖

```toml
# pyproject.toml
[project]
dependencies = [
    # ...
    "typer>=0.15",
    "rich>=13",
]
```

Typer 从函数签名生成帮助和参数校验；Rich 仅用于人类输出。JSON 输出只使用标准库 `json`，保证脚本格式稳定。

### 2. 创建应用入口和依赖注入

```python
# src/scientex_agent/cli.py
from __future__ import annotations

import json
import sys
from enum import IntEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from .bootstrap import build_cli_services

app = typer.Typer(no_args_is_help=True, add_completion=False)
projects_app = typer.Typer(no_args_is_help=True)
artifacts_app = typer.Typer(no_args_is_help=True)
app.add_typer(projects_app, name="projects")
app.add_typer(artifacts_app, name="artifacts")

stdout = Console(file=sys.stdout)
stderr = Console(stderr=True)


class ExitCode(IntEnum):
    OK = 0
    VALIDATION = 2
    DOMAIN = 3
    CANCELLED = 4
    EXTERNAL = 5
    INTERNAL = 70


def emit(value: Any, output: str) -> None:
    if output == "json":
        sys.stdout.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    else:
        stdout.print(value)


@app.callback()
def main(
    ctx: typer.Context,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    output: Annotated[str, typer.Option("--format", case_sensitive=False)] = "table",
) -> None:
    if output not in {"table", "json"}:
        raise typer.BadParameter("--format must be table or json")
    ctx.obj = {"services": build_cli_services(data_dir), "output": output}
```

服务构造集中在 `bootstrap.py`，可以按 `--data-dir`、配置文件和环境变量创建。命令函数从 `ctx.obj` 取得服务，测试中以 fake service 覆盖它，而不修改全局变量。

### 3. 编写一个读命令和一个写命令

```python
@projects_app.command("list")
def list_projects(ctx: typer.Context) -> None:
    services = ctx.obj["services"]
    rows = [project.to_public_dict() for project in services.projects.list()]
    emit(rows, ctx.obj["output"])


@projects_app.command("create")
def create_project(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    description: Annotated[str, typer.Option("--description")] = "",
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation")] = False,
) -> None:
    if not yes and sys.stdin.isatty():
        confirmed = typer.confirm(f"Create project {name!r}?")
        if not confirmed:
            raise typer.Exit(ExitCode.CANCELLED)
    elif not yes:
        stderr.print("Refusing non-interactive write without --yes")
        raise typer.Exit(ExitCode.CANCELLED)

    project = ctx.obj["services"].projects.create(name=name, description=description)
    emit(project.to_public_dict(), ctx.obj["output"])
```

对 `chat`，不要把 token 和最终 JSON 混在一起：默认模式下流式打印到终端；`--format json` 则缓冲到 `run.completed` 后一次输出完整结果，或提供 `--events jsonl` 输出每行一个 `RunEvent`。

### 4. 统一错误转换

```python
def run() -> None:
    try:
        app()
    except ValidationError as error:
        stderr.print(f"Input error: {error.public_detail}")
        raise typer.Exit(ExitCode.VALIDATION) from error
    except DomainError as error:
        stderr.print(error.public_detail)
        raise typer.Exit(ExitCode.DOMAIN) from error
    except ExternalServiceError as error:
        stderr.print(f"External service error: {error.public_detail}")
        raise typer.Exit(ExitCode.EXTERNAL) from error
    except Exception:
        logger.exception("unhandled CLI exception")
        stderr.print("Unexpected internal error; see logs with the trace id.")
        raise typer.Exit(ExitCode.INTERNAL)
```

密码、API key 和完整 provider 请求不能出现在报错、`--verbose` 或 shell history 中。密钥通过环境变量、系统密钥链或交互式安全输入获取，CLI 只显示已脱敏的 provider 配置摘要。

## 验证

```bash
# 帮助是从签名生成的，并且不启动 provider
uv run scientex_agent --help
uv run scientex_agent projects --help

# 机器可读输出；可稳定被 jq 使用
uv run scientex_agent projects list --format json | jq '.[].name'

# 非交互写入必须明确确认
uv run scientex_agent projects create "CRISPR screen" --yes --format json

# 从事件流恢复一个待审批 run
uv run scientex_agent runs resume RUN_ID --approve TOOL_CALL_ID
```

使用 `typer.testing.CliRunner` 覆盖：帮助、合法参数、无效 `--format`、JSON stdout 的纯净性、非交互写入拒绝、每一种领域异常的退出码。端到端测试再用临时数据目录验证实际 Artifact 导出与恢复。

## 深入理解

### 为什么不继续堆叠 argparse

`argparse` 对小脚本完全足够；随着嵌套命令、类型、completion、测试和自动文档增加，Typer 可让命令接口直接由类型签名表达。选择它不是为了隐藏业务逻辑，恰恰是把“解析参数”和“调用服务”分开。

### CLI 与 HTTP 服务如何共存

本地使用时，CLI 可以直接创建 application service，减少一个常驻服务进程；需要远程管理时，另提供明确的 `--api-url` HTTP client。两种模式都使用相同 DTO 和 API 契约，避免把本机文件路径意外发送到远端。

## 当前局限

- 初版只提供单用户、本地默认配置；远程身份令牌和多用户命名空间在 Step 16 处理。
- Shell completion、配置编辑器和交互式 TUI 不属于本教程范围。
- 默认表格视图不取代 Web 的可视化 provenance 图。

## 下一步

下一章构建 Vue 3 + TypeScript 前端。它复用 `/api/v1` 和 RunEvent SSE 协议，而不是再维护一套原生 JavaScript 状态管理代码。
