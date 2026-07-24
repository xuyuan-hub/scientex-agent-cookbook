"""Shared data models — frozen dataclasses, no logic."""

from __future__ import annotations

from dataclasses import dataclass
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
    idx: int
    role: str
    content: Any
    provider_message_id: str | None = None
    created_at: int = 0
