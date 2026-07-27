"""Regression tests for SQLite row-to-dataclass conversion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scientex_agent.metadata import MetadataStore


class MetadataStoreTests(unittest.TestCase):
    def test_sqlite_rows_restore_projects_frames_and_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MetadataStore(Path(temp_dir) / "app.db")
            store.initialize()

            project = store.create_project(name="Test Project")
            frame = store.create_frame(project_id=project.id, name="Chat 1")
            store.append_message(frame_id=frame.id, role="user", content="Hello")
            store.append_message(frame_id=frame.id, role="assistant", content="Hi there!")

            restored_project = store.get_project(project.id)
            restored_frame = store.get_frame(frame.id)
            restored_messages = store.list_messages(frame.id)

        self.assertEqual(restored_project, project)
        self.assertEqual(restored_frame, frame)
        self.assertEqual(
            [message.content for message in restored_messages], ["Hello", "Hi there!"]
        )


if __name__ == "__main__":
    unittest.main()
