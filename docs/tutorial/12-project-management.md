# Step 12: 项目管理（多项目 / Frame / 产物）

## 目标

把 Step 06 的 Project 与 Frame 扩展成可复现的科研工作空间：Agent、MCP 和 Kernel 的输入/输出成为带版本、来源和 hash 的 Artifact；导出包可以在另一台机器上检查和重放。

## 前置条件

- 完成 [11-http-api.md](11-http-api.md)
- 理解 SQLite 事务与文件系统原子写入

## 设计思路

### 工作空间模型

```
Project
  ├── Frame（对话 / 任务分支）
  │     └── Run（一次 Agent 执行，含事件和工具调用）
  │
  ├── Artifact（稳定逻辑名称，例如 results/qc.csv）
  │     └── ArtifactVersion（不可变版本，含 blob SHA-256）
  │
  └── ProvenanceEdge（run / tool / artifact / external source 之间的关系）
```

Message 是对话记录；Artifact 是用户或工具产生、值得长期保留的文件/结构化数据；Run 是一次可观测的执行。三者分开，才能避免“聊天文本里有文件名”成为唯一证据。

### 二进制内容与元数据分离

- SQLite 保存 Artifact 元数据、版本、关联和事务状态；
- blob 文件按内容 hash 存在受控根目录；
- 用户看到的是稳定逻辑路径，实际文件名不直接接受用户输入；
- 版本不可覆盖，最新版本只是一个查询结果。

```
data/
├── metadata.sqlite
└── blobs/
    └── sha256/
        └── 5e/
            └── 5e884898...    # 内容寻址；相同内容只保存一次
```

这不是 Git 的替代品；它解决运行数据的可追溯性和去重。源代码、Skill 与配置仍应进入 Git 并在 Run 中记录 commit/hash。

### 写入顺序

```
用户/工具提交 bytes
  → 在 blobs/.tmp 写入并计算 SHA-256
  → fsync + 原子 rename 到 blobs/sha256/...
  → SQLite BEGIN IMMEDIATE
  → 插入 artifact_versions 与 provenance
  → COMMIT
```

如果进程在 rename 后崩溃，会留下无引用 blob；启动时的垃圾回收可安全清理它。反过来，绝不能先在数据库声明版本再写文件，否则读者会看到不存在的 Artifact。

## 实现

### 1. 扩展领域模型

```python
# src/scientex_agent/models.py（新增）
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Artifact:
    id: str
    project_id: str
    logical_path: str
    media_type: str
    created_at: int


@dataclass(frozen=True)
class ArtifactVersion:
    id: str
    artifact_id: str
    version: int
    blob_sha256: str
    size_bytes: int
    created_by_run_id: str | None
    metadata_json: str
    created_at: int


@dataclass(frozen=True)
class ProvenanceEdge:
    project_id: str
    source_kind: Literal["run", "tool_execution", "artifact", "external"]
    source_id: str
    target_artifact_version_id: str
    relation: Literal["generated", "used", "derived_from", "cites"]
```

`logical_path` 不是操作系统路径。接受它时先规范化为 POSIX 相对路径，禁止空段、`..`、绝对路径和控制字符；文件系统内部只根据 hash 寻址。

### 2. 创建内容寻址 BlobStore

```python
# src/scientex_agent/blob_store.py
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


class BlobStore:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._tmp = self._root / ".tmp"
        self._tmp.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, content: bytes) -> tuple[str, int]:
        digest = hashlib.sha256(content).hexdigest()
        destination = self._root / "sha256" / digest[:2] / digest
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            return digest, len(content)

        fd, temporary_name = tempfile.mkstemp(dir=self._tmp, prefix="blob-")
        try:
            with os.fdopen(fd, "wb") as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            try:
                os.replace(temporary_name, destination)
            except FileExistsError:
                pass  # 并发写入同一 hash：另一方已经完成
        finally:
            Path(temporary_name).unlink(missing_ok=True)
        return digest, len(content)

    def open(self, digest: str):
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("invalid sha256 digest")
        return (self._root / "sha256" / digest[:2] / digest).open("rb")
```

生产实现对输入使用流式 hash，避免大文件全部驻留内存；示例保持短小以展示原子顺序。blob 根目录由应用配置决定，任何用户提供的路径都不能绕过它。

### 3. 在事务中创建不可变版本

```python
# src/scientex_agent/artifact_service.py（核心方法）
def create_version(
    self,
    *,
    project_id: str,
    logical_path: str,
    content: bytes,
    media_type: str,
    run_id: str | None,
    metadata: dict,
) -> ArtifactVersion:
    normalized = normalize_logical_path(logical_path)
    digest, size = self._blobs.put_bytes(content)

    with self._metadata.transaction(immediate=True) as conn:
        artifact = self._metadata.get_or_create_artifact(
            conn, project_id=project_id, logical_path=normalized, media_type=media_type
        )
        next_version = self._metadata.next_artifact_version(conn, artifact.id)
        version = self._metadata.insert_artifact_version(
            conn,
            artifact_id=artifact.id,
            version=next_version,
            blob_sha256=digest,
            size_bytes=size,
            created_by_run_id=run_id,
            metadata=metadata,
        )
        if run_id:
            self._metadata.insert_provenance(
                conn, project_id=project_id, source_kind="run", source_id=run_id,
                target_artifact_version_id=version.id, relation="generated",
            )
    return version
```

同一 Artifact 的 `version` 设置唯一约束 `(artifact_id, version)`。并发冲突返回 409 或在有限次数内重新读取版本号后重试；绝不能使用“读最大值、连接关闭、再写入”的无事务流程。

### 4. 设计 API 和导出包

Step 11 的 API 增加以下端点：

```
GET  /api/v1/projects/{project_id}/artifacts
GET  /api/v1/projects/{project_id}/artifacts/{artifact_id}/versions
GET  /api/v1/artifact-versions/{version_id}/content
POST /api/v1/projects/{project_id}/exports
POST /api/v1/projects/imports
```

导出格式使用 ZIP + `manifest.json`，而不是 pickle：

```json
{
  "format": "scientex-project-export",
  "format_version": 1,
  "project": {"id": "proj_01", "name": "CRISPR screen"},
  "artifacts": [
    {
      "logical_path": "results/qc.csv",
      "version": 2,
      "sha256": "5e8848...",
      "archive_path": "blobs/5e8848..."
    }
  ],
  "skills": [{"id": "literature-review@1.0.0", "sha256": "..."}],
  "source_revision": "git:abc123"
}
```

导入时先验证 schema、archive 成员路径、每个 blob hash 和大小限制；通过后写入临时目录，最后在事务中登记。不要解压到用户指定路径，也不要相信 ZIP 内的文件名。

### 5. 把 Run、Artifact 与引用连起来

`artifact__write`、kernel 输出和 MCP 下载都必须带 `run_id` 与输入 Artifact version id。UI 显示结果时可提供：

```
这个 CSV
  ← Run run_123（模型 / provider / skill hash）
  ← Tool pubmed__search（参数和时间）
  ← 输入 data/raw/papers.json@3（SHA-256）
```

这条 provenance DAG 在科学场景中比“最终回答写得通顺”更重要：它使用户可以检查来源、复用某一步或发现错误后只重新计算受影响的下游版本。

## 验证

```bash
# 第一次创建与第二次更新同一逻辑文件，版本递增
uv run scientex_agent artifacts put --project PROJECT_ID results/qc.csv ./qc-v1.csv
uv run scientex_agent artifacts put --project PROJECT_ID results/qc.csv ./qc-v2.csv
uv run scientex_agent artifacts history --project PROJECT_ID results/qc.csv

# 导出后在临时目录验证清单和 hash，不导入
uv run scientex_agent projects export PROJECT_ID --output project.scitex.zip
uv run scientex_agent projects verify-export project.scitex.zip
```

至少测试：相同 bytes 去重、逻辑路径穿越被拒绝、版本号在并发写入中不重复、写入失败不会出现数据库悬挂版本、篡改 ZIP 内 blob 后导入失败、Run 生成的 Artifact 有完整 provenance edge。

## 深入理解

### 为什么不直接把文件放在 SQLite BLOB

小型项目可以这样做，但把大量二进制文件放进 SQLite 会放大数据库、备份和写锁。内容寻址文件存储结合 SQLite 元数据，能获得更好的大文件流式读写与去重，同时保留事务性的版本记录。部署成多机服务时，再将 `BlobStore` 替换为对象存储适配器。

### 删除与保留策略

用户删除逻辑 Artifact 时，先标记删除或移除引用，不立即删除 blob。只在所有版本、导出和保留期都不再引用后，由带 dry-run 报告的垃圾回收清理。科研记录通常需要比界面可见内容更长的保留期。

## 当前局限

- 目前以本地 SQLite + 文件系统为目标，不提供多人并发编辑或云对象存储。
- provenance 记录关系与 hash，不自动判断科学结论是否正确。
- 导出包不包含 provider API key、kernel 状态或任意本机绝对路径。

## 下一步

下一章将这些能力包装为稳定的 Typer CLI：交互式聊天方便探索，脚本化命令和 JSON 输出方便自动化与 CI。
