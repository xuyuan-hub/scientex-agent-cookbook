# Step 06: 对话持久化（SQLite）

## 目标

将对话历史保存到 SQLite 数据库中，支持关闭后重新打开继续对话。

## 前置条件

- 完成 [05-multi-provider.md](05-multi-provider.md)

## 设计思路

### 为什么需要持久化

```
之前：ChatSession 的 messages 存内存中 → 关掉程序 → 全丢了
现在：每次对话都写入 SQLite → 随时重启 → 之前的对话还在
```

### 数据模型

```
Project（项目）
  ├── id, name, description, created_at
  │
  └── Frame（对话帧）
        ├── id, project_id, name, status
        │
        └── Message（消息）
              ├── frame_id, idx, role, content, created_at
              │
              └── CompactionArchive（压缩归档）
                    └── id, frame_id, summary, messages_json
```

**为什么叫 Frame 而不是 Conversation？**
Frame（帧）是比 Conversation 更灵活的概念：
- 一个 Project 可以有多个 Frame
- Frame 可以有父子关系（子任务派生）
- Frame 有状态（running, completed, failed）

### SQLite 3 层模式

```
app.py          ← 业务代码
metadata.py     ← 数据访问层（repository 模式）
models.py       ← 纯数据模型（dataclass）
```

`models.py` 定义数据结构，`metadata.py` 实现 SQL 操作，
`app.py` 调用 `metadata.py` 的方法，不直接写 SQL。

## 实现

### 1. 创建 src/scientex_agent/models.py

```python
"""Shared data models — frozen dataclasses, no logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    description: str
    context: str = ""
    memory_enabled: bool = True
    created_at: int = 0
    updated_at: int = 0


@dataclass(frozen=True)
class Frame:
    id: str
    project_id: str
    parent_frame_id: str | None = None
    root_frame_id: str = ""
    name: str = ""
    task_summary: str = ""
    status: str = "active"
    provider: str | None = None
    model: str | None = None
    created_at: int = 0
    updated_at: int = 0
    completed_at: int | None = None


@dataclass(frozen=True)
class Message:
    frame_id: str
    idx: int                    # 0-based index within frame
    role: str                   # "user", "assistant", "tool", "system"
    content: Any                # string or dict (for tool results)
    provider_message_id: str | None = None
    created_at: int = 0
```

### 2. 创建 src/scientex_agent/metadata.py

```python
"""SQLite metadata store for projects, frames, and messages."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

from .models import Frame, Message, Project


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class MetadataStore:
    """SQLite-backed store for all application metadata.

    Usage::

        store = MetadataStore(Path("~/.scientex_agent/app.db"))
        store.initialize()  # creates tables if needed

        project = store.create_project(name="My Research", description="...")
        frame = store.create_frame(project_id=project.id, name="Chat 1")
        msg = store.append_message(frame_id=frame.id, role="user", content="Hello")
        messages = store.list_messages(frame.id)
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)

    def initialize(self) -> None:
        """Create the schema if it does not exist (idempotent)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # === Projects ===

    def create_project(
        self,
        *,
        name: str,
        description: str = "",
        context: str = "",
    ) -> Project:
        now = _now_ms()
        pid = _new_id()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO projects (id, name, description, context, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (pid, name, description, context, now, now),
            )
        return Project(id=pid, name=name, description=description,
                       context=context, created_at=now, updated_at=now)

    def get_project(self, project_id: str) -> Project | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return _row_to_project(row) if row else None

    def list_projects(self) -> list[Project]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        return [_row_to_project(r) for r in rows]

    # === Frames ===

    def create_frame(
        self,
        *,
        project_id: str,
        name: str = "",
        parent_frame_id: str | None = None,
        root_frame_id: str | None = None,
    ) -> Frame:
        now = _now_ms()
        fid = _new_id()
        root = root_frame_id or fid
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO frames (id, project_id, parent_frame_id, root_frame_id, name,
                   status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'active', ?, ?)""",
                (fid, project_id, parent_frame_id, root, name, now, now),
            )
        return Frame(id=fid, project_id=project_id, parent_frame_id=parent_frame_id,
                     root_frame_id=root, name=name, created_at=now, updated_at=now)

    def get_frame(self, frame_id: str) -> Frame | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM frames WHERE id = ?", (frame_id,)).fetchone()
        return _row_to_frame(row) if row else None

    def list_frames(self, project_id: str) -> list[Frame]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM frames WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        return [_row_to_frame(r) for r in rows]

    # === Messages ===

    def append_message(
        self,
        *,
        frame_id: str,
        role: str,
        content: Any,
    ) -> Message:
        now = _now_ms()
        with self._connect() as conn:
            # Get next index
            row = conn.execute(
                "SELECT COALESCE(MAX(idx), -1) + 1 AS next_idx FROM messages WHERE frame_id = ?",
                (frame_id,),
            ).fetchone()
            idx = row["next_idx"]

            content_json = _serialize_content(content)
            conn.execute(
                """INSERT INTO messages (frame_id, idx, role, content, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (frame_id, idx, role, content_json, now),
            )
        return Message(frame_id=frame_id, idx=idx, role=role, content=content, created_at=now)

    def list_messages(self, frame_id: str) -> list[Message]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE frame_id = ? ORDER BY idx",
                (frame_id,),
            ).fetchall()
        return [_row_to_message(r) for r in rows]

    def message_count(self, frame_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM messages WHERE frame_id = ?",
                (frame_id,),
            ).fetchone()
        return row["cnt"] if row else 0


# === Row → Dataclass Helpers ===

def _row_to_project(row: sqlite3.Row) -> Project:
    return Project(
        id=row["id"], name=row["name"], description=row["description"],
        context=row["context"] or "",
        memory_enabled=bool(row.get("memory_enabled", True)),
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


def _row_to_frame(row: sqlite3.Row) -> Frame:
    return Frame(
        id=row["id"], project_id=row["project_id"],
        parent_frame_id=row.get("parent_frame_id"),
        root_frame_id=row.get("root_frame_id", ""),
        name=row.get("name", ""), task_summary=row.get("task_summary", ""),
        status=row.get("status", "active"),
        provider=row.get("provider"), model=row.get("model"),
        created_at=row.get("created_at", 0), updated_at=row.get("updated_at", 0),
        completed_at=row.get("completed_at"),
    )


def _row_to_message(row: sqlite3.Row) -> Message:
    return Message(
        frame_id=row["frame_id"], idx=row["idx"], role=row["role"],
        content=_deserialize_content(row["content"]),
        provider_message_id=row.get("provider_message_id"),
        created_at=row.get("created_at", 0),
    )


def _serialize_content(content: Any) -> str:
    if isinstance(content, str):
        return json.dumps({"type": "text", "text": content}, ensure_ascii=False)
    return json.dumps(content, ensure_ascii=False, default=str)


def _deserialize_content(raw: str) -> Any:
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and obj.get("type") == "text":
            return obj.get("text", raw)
        return obj
    except (json.JSONDecodeError, TypeError):
        return raw


# === SQL Schema ===

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    context TEXT NOT NULL DEFAULT '',
    memory_enabled INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS frames (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    parent_frame_id TEXT,
    root_frame_id TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL DEFAULT '',
    task_summary TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    provider TEXT,
    model TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    completed_at INTEGER
);

CREATE TABLE IF NOT EXISTS messages (
    frame_id TEXT NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    provider_message_id TEXT,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (frame_id, idx)
);
"""
```

### 3. 更新 ChatSession 支持持久化

```python
# 在 llm_client.py 中

class PersistentChatSession(ChatSession):
    """ChatSession that persists messages to SQLite."""

    def __init__(
        self,
        frame_id: str,
        metadata: "MetadataStore",
        model: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        super().__init__(model=model, client=client)
        self.frame_id = frame_id
        self.metadata = metadata
        # Restore existing messages
        self._restore()

    def _restore(self) -> None:
        """Load messages from DB into memory."""
        stored = self.metadata.list_messages(self.frame_id)
        for msg in stored:
            role = msg.role
            content = msg.content
            if isinstance(content, str):
                self.messages.append({"role": role, "content": content})
            else:
                self.messages.append({"role": role, "content": str(content)})

    def send(self, content: str, **kwargs) -> str:
        self.metadata.append_message(
            frame_id=self.frame_id, role="user", content=content
        )
        reply = super().send(content, **kwargs)
        # Already appended in super().send()
        return reply
```

## 验证

```python
from pathlib import Path
from scientex_agent.metadata import MetadataStore

# 初始化数据库
data_dir = Path("/tmp/scientex_agent-test")
store = MetadataStore(data_dir / "app.db")
store.initialize()

# 创建项目 → 创建 Frame → 存储消息
project = store.create_project(name="Test Project")
frame = store.create_frame(project_id=project.id, name="Chat 1")

store.append_message(frame_id=frame.id, role="user", content="Hello")
store.append_message(frame_id=frame.id, role="assistant", content="Hi there!")

# 恢复消息
messages = store.list_messages(frame.id)
for msg in messages:
    print(f"[{msg.role}] {msg.content}")

# 重启后消息仍然存在（重新打开 store 验证）
store2 = MetadataStore(data_dir / "app.db")
msgs2 = store2.list_messages(frame.id)
assert len(msgs2) == 2  # 消息还在
```

## 深入理解

### WAL 模式

```sql
PRAGMA journal_mode=WAL;
```

WAL（Write-Ahead Logging）模式的优势：
- 读操作不阻塞写操作
- 写操作不阻塞读操作
- 比默认的 DELETE 模式更适合并发场景

### 为什么用 UUID 而不是自增 ID

```python
# ❌ 自增：简单但有问题
CREATE TABLE projects (id INTEGER PRIMARY KEY AUTOINCREMENT);
# → 导出到另一个数据库时 ID 会冲突

# ✅ UUID：全局唯一
id = uuid.uuid4().hex[:12]  # 12 字符，够用且 URL 友好
# → 可以安全地合并、导出、分布式存储
```

### Content 序列化

消息内容可能是纯文本字符串，也可能是复杂的 tool result 对象。
统一用 JSON 存储：

```python
# 存入
content = json.dumps({"type": "text", "text": "Hello"}, ensure_ascii=False)

# 取出
obj = json.loads(content)
text = obj.get("text", content)
```

## 当前局限

1. 没有消息压缩/归档——对话太长会超 context window
2. 没有项目和 Frame 的更新/删除操作
3. 没有软删除或回收站机制

## 下一步

→ [07-mcp-integration.md](07-mcp-integration.md)
