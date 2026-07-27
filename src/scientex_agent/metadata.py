"""SQLite metadata store for projects, frames, and messages."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from .models import Frame,Message,Project

def _now_ms() -> int:
    return int(time.time()*1000)

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

    def __init__(self, db_path: str |Path)->None:
        self._path = Path(db_path)

    def initialize(self) -> None:
        """Create the schema if it does not exist (idempotent)."""
        self._path.parent.mkdir(parents=True,exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.Connection(str(self._path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # === Projects ===

    def create_project(
        self,
        *,
        name: str,
        description: str="",
        context: str=""
    ) -> Project:
        now = _now_ms()
        pid = _new_id()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO projects (id, name, description, context, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (pid, name, description, context, now, now),
            )
            return Project(id=pid,name=name,description=description,
                           context=context,created_at=now,updated_at=now)

    def get_project(self,project_id: str)->Project|None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return _row_to_project(row) if row else None

    def list_projects(self) -> list[Project]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        return [_row_to_project(row) for row in rows]

    # === Frames ===

    def create_frame(
        self,
        *,
        project_id: str,
        name: str = "",
        parent_frame_id: str | None = None,
        root_frame_id: str| None = None
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

# === Row -> Dataclass Helpers ===

def _row_to_project(row:sqlite3.Row) -> Project:
    return Project(
        id=row["id"],name=row["name"],description=row["description"],
        context=row["context"] or "",
        memory_enabled=bool(row["memory_enabled"]),
        created_at=row["created_at"],updated_at=row["updated_at"],
    )

def _row_to_frame(row: sqlite3.Row) -> Frame:
    return Frame(
        id=row["id"], project_id=row["project_id"],
        parent_frame_id=row["parent_frame_id"],
        root_frame_id=row["root_frame_id"],
        name=row["name"], task_summary=row["task_summary"],
        status=row["status"],
        provider=row["provider"], model=row["model"],
        created_at=row["created_at"], updated_at=row["updated_at"],
        completed_at=row["completed_at"],
    )


def _row_to_message(row: sqlite3.Row) -> Message:
    return Message(
        frame_id=row["frame_id"], idx=row["idx"], role=row["role"],
        content=_deserialize_content(row["content"]),
        provider_message_id=row["provider_message_id"],
        created_at=row["created_at"],
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
