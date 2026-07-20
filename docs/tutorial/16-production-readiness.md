# Step 16: 生产化 & 交付

## 目标

把开发版变成可交付的产品：错误处理、备份恢复、打包分发、性能优化、安全加固。

## 前置条件

- 完成前 15 步的所有功能

## 设计思路

### 生产化检查清单

```
☐ 错误处理：所有异常都有友好的用户提示
☐ 数据安全：备份/恢复/校验完整链路
☐ 打包分发：pip install 一键安装
☐ 日志系统：分级日志，可调试
☐ 性能优化：大文件流式传输，数据库索引
☐ 安全加固：输入验证，路径遍历防护
☐ 版本管理：版本号 + 升级迁移
☐ 文档：README + API 文档 + 开发者指南
```

## 实现

### 1. 日志系统

```python
# src/scientex_agent/logging_config.py

"""Centralized logging configuration."""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure logging for Scientex.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file to write logs to.

    Returns:
        Root logger.
    """
    logger = logging.getLogger("scientex_agent")
    logger.setLevel(level)
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets everything
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Usage
logger = setup_logging(level=logging.INFO)
logger.info("Scientex starting...")
logger.debug("This won't show unless level=DEBUG")
```

### 2. 错误处理

```python
# src/scientex_agent/errors.py

"""Application-level error handling."""


class ScientexError(Exception):
    """Base exception for Scientex."""
    def __init__(self, message: str, *, user_message: str | None = None):
        super().__init__(message)
        self.user_message = user_message or message


class ConfigurationError(ScientexError):
    """Configuration-related errors."""
    pass


class ProviderError(ScientexError):
    """LLM provider errors (API key, network, rate limit)."""
    pass


class StorageError(ScientexError):
    """Data storage errors."""
    pass


class ValidationError(ScientexError):
    """Input validation errors."""
    pass


# Error handler middleware for FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse

async def scientex_error_handler(request: Request, exc: ScientexError):
    return JSONResponse(
        status_code=400,
        content={"error": exc.user_message, "detail": str(exc)},
    )

async def unhandled_error_handler(request: Request, exc: Exception):
    import logging
    logger = logging.getLogger("scientex_agent")
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )
```

### 3. 输入验证与安全

```python
# src/scientex_agent/security.py

"""Security utilities — input validation, path safety, secret redaction."""

import os
import re
from pathlib import Path


# === Path Safety ===

def safe_path(base_dir: Path, user_path: str | Path) -> Path:
    """Resolve a user-supplied path, ensuring it stays within base_dir.

    Raises:
        ValueError: If the path escapes base_dir.
    """
    resolved = (base_dir / user_path).resolve()
    if not str(resolved).startswith(str(base_dir.resolve())):
        raise ValueError(f"Path escapes base directory: {user_path}")
    return resolved


# === Input Validation ===

def validate_frame_id(frame_id: str) -> bool:
    """Validate frame ID format (alphanumeric, 12 chars)."""
    return bool(re.match(r"^[a-f0-9]{12}$", frame_id))


def validate_project_name(name: str) -> str:
    """Validate and sanitize a project name."""
    name = name.strip()
    if not name:
        raise ValueError("Project name cannot be empty")
    if len(name) > 200:
        raise ValueError("Project name too long (max 200 chars)")
    # Remove potentially dangerous characters
    name = re.sub(r"[/\\:*?\"<>|]", "", name)
    return name


# === Secret Redaction ===

_SECRET_PATTERNS = [
    (re.compile(r"(sk-ant-[a-zA-Z0-9_-]+)"), "sk-ant-***"),
    (re.compile(r"(sk-[a-zA-Z0-9_-]{20,})"), "sk-***"),
    (re.compile(r"(AIza[a-zA-Z0-9_-]{30,})"), "AIza***"),
    (re.compile(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"), "***@***"),
]


def redact_secrets(text: str) -> str:
    """Redact API keys and credentials from text."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# === Rate Limiting ===

import time
from collections import defaultdict


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, max_requests: int = 60, window: float = 60.0):
        self.max_requests = max_requests
        self.window = window
        self._clients: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if a client is within rate limits."""
        now = time.time()
        window_start = now - self.window

        # Remove old entries
        self._clients[client_id] = [
            t for t in self._clients[client_id] if t > window_start
        ]

        if len(self._clients[client_id]) >= self.max_requests:
            return False

        self._clients[client_id].append(now)
        return True
```

### 4. 数据库迁移

```python
# src/scientex_agent/migrations.py

"""Database schema migrations.

When the schema version changes, apply migrations in order.
"""

SCHEMA_VERSION = 1

MIGRATIONS = {
    1: """
        -- Initial schema
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY
        );
        INSERT OR IGNORE INTO schema_version (version) VALUES (1);
    """,
    # Future migrations:
    # 2: """
    #     ALTER TABLE projects ADD COLUMN tags TEXT DEFAULT '';
    #     UPDATE schema_version SET version = 2;
    # """,
}


def run_migrations(db_path: str) -> None:
    """Apply any pending migrations to the database."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        # Get current version
        row = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()
        current = row[0] if row and row[0] else 0

        for version in range(current + 1, SCHEMA_VERSION + 1):
            if version in MIGRATIONS:
                conn.executescript(MIGRATIONS[version])
                conn.commit()

    finally:
        conn.close()
```

### 5. 打包配置

```toml
# pyproject.toml（完整版）

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "scientex_agent"
version = "0.2.0"
description = "Local scientific agent platform with pluggable LLM providers"
requires-python = ">=3.11"
readme = "README.md"
license = { text = "MIT" }
authors = [{ name = "Your Name", email = "you@example.com" }]
keywords = ["scientific", "agent", "llm", "bioinformatics"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "openai>=2.46.0",
    "mcp>=1.9",
    "langgraph>=0.2.0",
    "langchain-core>=0.3.0",
    "langchain-openai>=0.2.0",
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "pandas>=2.2",
    "pyyaml>=6.0",
]

[project.scripts]
scientex_agent = "scientex_agent.cli:main"

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.23",
    "playwright>=1.46",
    "ruff>=0.4",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
scientex_agent = [
    "web/*",
    "web/**/*",
    "skills/*/SKILL.md",
    "skills/*/*.md",
    "skills/*/*.py",
]

[tool.pytest.ini_options]
pythonpath = ["src"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]
```

### 6. 测试

```python
# tests/test_app.py

"""Integration tests for the Scientex application."""

import unittest
import tempfile
from pathlib import Path
import asyncio

from scientex_agent.app import ScientexApp


class TestScientexApp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = ScientexApp(Path(self.tmp.name))
        self.app.initialize()

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_project(self):
        project = self.app.create_project(name="Test")
        self.assertEqual(project.name, "Test")
        self.assertTrue(len(project.id) == 12)

    def test_list_projects(self):
        self.app.create_project(name="P1")
        self.app.create_project(name="P2")
        projects = self.app.list_projects()
        self.assertEqual(len(projects), 2)

    def test_create_frame(self):
        project = self.app.create_project(name="Test")
        frame = self.app.create_frame(project_id=project.id, name="Chat")
        self.assertEqual(frame.name, "Chat")
        self.assertEqual(frame.project_id, project.id)

    def test_message_persistence(self):
        project = self.app.create_project(name="Test")
        frame = self.app.create_frame(project_id=project.id)
        self.app.metadata.append_message(
            frame_id=frame.id, role="user", content="Hello"
        )
        self.app.metadata.append_message(
            frame_id=frame.id, role="assistant", content="Hi!"
        )
        messages = self.app.metadata.list_messages(frame.id)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[1].role, "assistant")


class TestAsyncChat(unittest.IsolatedAsyncioTestCase):
    """Tests that require async/await."""

    async def test_chat_requires_api_key(self):
        """Chat should raise an error if no API key is configured."""
        # This test requires DEEPSEEK_API_KEY or OPENAI_API_KEY in env
        import os
        if not os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            self.skipTest("No API key configured")

        import tempfile
        app = ScientexApp(Path(tempfile.mkdtemp()))
        app.initialize()
        project = app.create_project(name="Test")
        frame = app.create_frame(project_id=project.id)

        result = await app.chat(frame_id=frame.id, content="Say hello in 3 words")
        self.assertIn("content", result)
        self.assertTrue(len(result["content"]) > 0)


if __name__ == "__main__":
    unittest.main()
```

### 7. README

```markdown
# Scientex

Local scientific agent platform with pluggable LLM providers.

## Quick Start

```bash
# Install
pip install scientex_agent

# Configure
cat > .env.local << 'EOF'
DEEPSEEK_API_KEY=sk-your-key
DEEPSEEK_MODEL=deepseek-chat
EOF

# Initialize
scientex_agent init

# Chat
scientex_agent chat

# Start web server
scientex_agent run-server
# Open http://127.0.0.1:8765
```

## Features

- Multi-provider LLM support (OpenAI, Anthropic, DeepSeek, Gemini)
- Tool calling with 200+ scientific tools
- File-based skill system with 30+ built-in skills
- Python code execution in isolated subprocess
- MCP protocol for external tool servers
- FastAPI HTTP server with SSE streaming
- Project management with export/import
- Backup, verify, and restore
- Web UI with streaming chat

## Development

```bash
git clone https://github.com/.../scientex_agent
cd scientex_agent
uv sync
uv run python -m unittest discover -s tests
```
```

## 验证

```bash
# 全部测试
uv run python -m unittest discover -s tests -v

# 打包
uv run python -m build

# 安装
pip install dist/scientex_agent-0.2.0-py3-none-any.whl

# 验证安装
scientex_agent --version
scientex_agent init --data-dir /tmp/test-install
scientex_agent chat --data-dir /tmp/test-install "Hello"
```

---

## 🎉 完成

至此，你已经从零开始构建了一个完整的科学 agent 平台。
回顾整个过程：

```
00  项目搭建              ← 现在可以 pip install
01  首次 LLM 调用          ← 从零到第一条回复
02  多轮对话               ← 上下文记忆
03  流式输出               ← 实时显示
04  工具调用               ← LLM 能做事了
05  多提供商               ← 不再绑死一家
06  对话持久化             ← 重启不丢失
07  MCP 集成              ← 无限扩展工具
08  技能系统               ← 领域知识注入
09  Python 内核            ← 代码执行能力
10  Agent 编排             ← LangGraph 专业编排
11  HTTP API              ← FastAPI + SSE
12  项目管理               ← 多项目/多帧/产物
13  CLI                   ← 命令行完整体验
14  Web 前端              ← SPA 界面
15  领域功能               ← 生物/化学/记忆/验证
16  生产化                 ← 可交付的产品
```

继续改进的方向：
- Agent 间协作（多 agent 对话）
- 远程计算后端（SLURM, AWS Batch）
- 插件市场（分享和安装技能）
- 移动端适配
- RAG（检索增强生成）集成
