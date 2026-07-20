# Step 12: 项目管理（多项目/多帧/产物）

## 目标

实现多项目管理：创建、切换、导出/导入项目，以及文件产物的版本管理。

## 前置条件

- 完成 [11-http-api.md](11-http-api.md)
- 理解 [06-conversation-persistence.md](06-conversation-persistence.md) 的数据模型

## 设计思路

### 数据层级

```
Project (项目)
  ├── name, description, context
  ├── memory_enabled
  │
  ├── Frame (对话帧)
  │     ├── name, status
  │     ├── parent_frame_id (父子关系)
  │     ├── Messages (消息历史)
  │     └── Execution Records (代码执行记录)
  │
  ├── Artifacts (文件产物)
  │     ├── filename, content_type
  │     └── ArtifactVersions (版本)
  │
  ├── Memories (持久记忆)
  └── Notes (笔记)
```

### 项目生命周期

```
create → active → export/import/share → delete
```

每个项目的数据完全隔离：
- SQLite 中通过 `project_id` 外键关联
- 文件系统中通过 `artifacts/{project_id}/` 目录隔离

### 产物版本管理

```
artifacts/
└── {project_id}/
    └── {artifact_id}/
        ├── v1_data.csv        ← 第一版
        ├── v2_data.csv        ← 第二版
        └── v3_data.csv        ← 当前版本
```

每次保存同名文件自动创建新版本，保留完整历史。

## 实现

### 1. 扩展 models.py

```python
# src/scientex_agent/models.py 新增

@dataclass(frozen=True)
class Artifact:
    """A file artifact stored in a project."""
    id: str
    project_id: str
    root_frame_id: str           # which frame created this
    frame_id: str | None         # which specific frame
    filename: str
    latest_version_id: str | None
    folder_id: str | None        # for folder organization
    is_user_upload: bool = False
    is_ephemeral: bool = False   # temporary artifacts
    created_at: int = 0
    updated_at: int = 0


@dataclass(frozen=True)
class ArtifactVersion:
    """A specific version of an artifact file."""
    id: str
    artifact_id: str
    version_number: int          # 1, 2, 3, ...
    frame_id: str | None         # which frame produced this version
    producing_execution_id: str | None
    content_type: str            # MIME type
    size_bytes: int
    checksum: str                # SHA-256
    storage_path: str            # relative path under artifacts/
    language: str | None = None  # programming language if code
    extracted_code: str | None = None
    created_at: int = 0


@dataclass(frozen=True)
class Memory:
    """A durable memory entry for a project."""
    id: str
    project_id: str
    category: str                # "user", "project", "knowledge"
    content: str
    created_at: int = 0
    updated_at: int = 0


@dataclass(frozen=True)
class Note:
    """A freeform note attached to a project."""
    id: str
    project_id: str | None
    title: str
    content: str
    created_at: int = 0
    updated_at: int = 0
```

### 2. 创建 src/scientex_agent/artifact_store.py

```python
"""File-based artifact storage with versioning."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from .metadata import MetadataStore
from .models import Artifact, ArtifactVersion


class ArtifactStore:
    """Manages versioned file artifacts for projects.

    Directory structure::

        {data_dir}/artifacts/
        └── {project_id}/
            ├── {artifact_id}/
            │   ├── v0001_data.csv
            │   ├── v0002_data.csv
            │   └── latest → v0002_data.csv
            └── ...

    Usage::

        store = ArtifactStore(data_dir, metadata_store)
        version = store.save(
            project_id="abc",
            filename="results.csv",
            data=b"col1,col2\\n1,2\\n",
            content_type="text/csv",
            frame_id="frame-1",
        )
        content = store.read(version)
    """

    def __init__(self, data_dir: str | Path, metadata: MetadataStore) -> None:
        self._root = Path(data_dir) / "artifacts"
        self.metadata = metadata

    def _project_dir(self, project_id: str) -> Path:
        return self._root / project_id

    def _artifact_dir(self, artifact_id: str) -> Path:
        # artifact_id includes project_id, but we use it as a unique key
        return self._root / artifact_id[:12] / artifact_id

    def save(
        self,
        *,
        project_id: str,
        filename: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        frame_id: str | None = None,
        is_user_upload: bool = False,
    ) -> ArtifactVersion:
        """Save a new version of a file.

        If an artifact with the same filename already exists in the project,
        a new version is created. Otherwise, a new artifact is created.
        """
        from .metadata import _new_id, _now_ms

        # Find existing artifact by filename in this project
        existing = self.metadata.get_artifact_by_filename(project_id, filename)

        if existing:
            artifact_id = existing.id
            version_number = self.metadata.next_artifact_version(artifact_id)
        else:
            artifact_id = _new_id()
            version_number = 1
            self.metadata.create_artifact(
                id=artifact_id,
                project_id=project_id,
                filename=filename,
                frame_id=frame_id,
                is_user_upload=is_user_upload,
            )

        # Compute checksum
        checksum = hashlib.sha256(data).hexdigest()

        # Write file
        version_dir = self._artifact_dir(artifact_id)
        version_dir.mkdir(parents=True, exist_ok=True)
        version_filename = f"v{version_number:04d}_{filename}"
        version_path = version_dir / version_filename
        version_path.write_bytes(data)

        # Update latest symlink
        latest_link = version_dir / f"latest_{filename}"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(version_filename)

        # Record in metadata
        version = self.metadata.create_artifact_version(
            artifact_id=artifact_id,
            version_number=version_number,
            content_type=content_type,
            size_bytes=len(data),
            checksum=checksum,
            storage_path=str(version_path.relative_to(self._root)),
            frame_id=frame_id,
        )

        return version

    def read(self, version: ArtifactVersion) -> bytes:
        """Read the content of an artifact version."""
        path = self._root / version.storage_path
        if not path.exists():
            raise FileNotFoundError(f"Artifact version file not found: {path}")
        return path.read_bytes()

    def latest_version(self, artifact_id: str) -> ArtifactVersion | None:
        """Get the latest version of an artifact."""
        return self.metadata.get_latest_artifact_version(artifact_id)

    def read_latest(self, project_id: str, filename: str) -> bytes | None:
        """Read the latest version of an artifact by project + filename."""
        artifact = self.metadata.get_artifact_by_filename(project_id, filename)
        if not artifact or not artifact.latest_version_id:
            return None
        version = self.metadata.get_artifact_version(artifact.latest_version_id)
        if not version:
            return None
        return self.read(version)

    def delete_artifact(self, artifact_id: str) -> None:
        """Delete an artifact and all its versions."""
        artifact_dir = self._artifact_dir(artifact_id)
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        self.metadata.delete_artifact(artifact_id)

    def export_project(self, project_id: str, target_path: str | Path) -> Path:
        """Export a project's artifacts as a zip file."""
        import zipfile

        target = Path(target_path)
        project_dir = self._project_dir(project_id)

        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
            if project_dir.exists():
                for file_path in project_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = str(file_path.relative_to(self._root))
                        zf.write(file_path, arcname)

            # Also export metadata as JSON
            import json
            project = self.metadata.get_project(project_id)
            if project:
                zf.writestr(
                    "project.json",
                    json.dumps({
                        "id": project.id,
                        "name": project.name,
                        "description": project.description,
                        "context": project.context,
                    }, ensure_ascii=False, indent=2),
                )

        return target

    def import_project(self, package_path: str | Path, project_name: str | None = None):
        """Import a project from a zip file."""
        import zipfile
        import json

        package = Path(package_path)
        project_id = None

        with zipfile.ZipFile(package, "r") as zf:
            # Read project metadata
            if "project.json" in zf.namelist():
                meta = json.loads(zf.read("project.json"))
                project_name = project_name or meta.get("name", "Imported Project")
                project_id = meta.get("id")

            # Create project
            project = self.metadata.create_project(
                name=project_name or "Imported Project",
                description=f"Imported from {package.name}",
            )

            # Extract files
            for member in zf.namelist():
                if member == "project.json":
                    continue
                # Extract to artifacts directory
                target = self._root / member
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))

        return project
```

### 3. 扩展 API 路由

```python
# api_server.py 新增路由

# === Artifacts ===

@api.get("/projects/{project_id}/artifacts")
async def list_artifacts(project_id: str) -> list[dict]:
    artifacts = scientex.metadata.list_artifacts(project_id)
    return [{"id": a.id, "filename": a.filename, "latest_version_id": a.latest_version_id,
             "is_user_upload": a.is_user_upload, "created_at": a.created_at} for a in artifacts]

@api.get("/artifacts/{artifact_id}/versions")
async def list_versions(artifact_id: str) -> list[dict]:
    versions = scientex.metadata.list_artifact_versions(artifact_id)
    return [{"id": v.id, "version_number": v.version_number, "content_type": v.content_type,
             "size_bytes": v.size_bytes, "checksum": v.checksum} for v in versions]

@api.get("/artifact-versions/{version_id}")
async def get_artifact_version(version_id: str):
    version = scientex.metadata.get_artifact_version(version_id)
    if not version:
        raise HTTPException(status_code=404)
    data = scientex.artifacts.read(version)
    from fastapi.responses import Response
    return Response(content=data, media_type=version.content_type)

@api.post("/projects/{project_id}/artifacts/upload")
async def upload_artifact(project_id: str, file: UploadFile):
    data = await file.read()
    version = scientex.artifacts.save(
        project_id=project_id, filename=file.filename or "upload",
        data=data, content_type=file.content_type or "application/octet-stream",
        is_user_upload=True,
    )
    return {"id": version.id, "version_number": version.version_number, "checksum": version.checksum}

# === Export/Import ===

@api.post("/projects/{project_id}/export")
async def export_project(project_id: str) -> dict:
    import tempfile
    output = Path(tempfile.gettempdir()) / f"{project_id}_export.zip"
    path = scientex.artifacts.export_project(project_id, output)
    return {"path": str(path)}

@api.post("/projects/import")
async def import_project(file: UploadFile, name: str | None = None) -> dict:
    import tempfile
    tmp = Path(tempfile.gettempdir()) / f"import_{file.filename}"
    tmp.write_bytes(await file.read())
    try:
        project = scientex.artifacts.import_project(tmp, name)
        return {"id": project.id, "name": project.name}
    finally:
        tmp.unlink(missing_ok=True)

# === Memories ===

@api.get("/projects/{project_id}/memories")
async def list_memories(project_id: str) -> list[dict]:
    memories = scientex.metadata.list_memories(project_id)
    return [{"id": m.id, "category": m.category, "content": m.content} for m in memories]

@api.post("/projects/{project_id}/memories")
async def create_memory(project_id: str, req: CreateMemoryRequest) -> dict:
    memory = scientex.metadata.create_memory(project_id=project_id, category=req.category, content=req.content)
    return {"id": memory.id}
```

## 验证

```python
from pathlib import Path
from scientex_agent.metadata import MetadataStore
from scientex_agent.artifact_store import ArtifactStore

data_dir = Path("/tmp/scientex_agent-test")
meta = MetadataStore(data_dir / "app.db")
meta.initialize()
store = ArtifactStore(data_dir, meta)

# 创建项目
project = meta.create_project(name="Test")

# 保存文件（第一版）
v1 = store.save(project_id=project.id, filename="data.csv",
                data=b"name,value\na,1\n", content_type="text/csv")
print(f"v1: version {v1.version_number}, checksum={v1.checksum[:8]}")

# 保存文件（第二版）
v2 = store.save(project_id=project.id, filename="data.csv",
                data=b"name,value\na,1\nb,2\n", content_type="text/csv")
print(f"v2: version {v2.version_number}, checksum={v2.checksum[:8]}")

# 读取最新版本
data = store.read_latest(project.id, "data.csv")
print(f"Latest content: {data.decode()}")

# 导出
zip_path = store.export_project(project.id, Path("/tmp/export_test.zip"))
print(f"Exported to: {zip_path}")
```

## 当前局限

1. 没有文件夹组织（所有产物扁平存放）
2. 没有产物预览生成
3. 大文件上传没有进度反馈

## 下一步

→ [13-cli.md](13-cli.md)
