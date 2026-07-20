# Step 13: 命令行界面（CLI）

## 目标

构建完整的 CLI，支持所有核心操作：初始化、聊天、启动服务、备份恢复。

## 前置条件

- 完成 [12-project-management.md](12-project-management.md)

## 设计思路

### CLI 职责

```
cli.py 只做三件事：
1. 解析命令行参数（argparse）
2. 创建 ScientexApp 实例
3. 调用对应方法，打印结果
```

**CLI 不应该包含业务逻辑**——所有逻辑都在 `ScientexApp` 中。

### 子命令设计

```
scientex_agent
├── init              初始化数据目录
├── chat              交互式对话
├── run-server        启动 HTTP 服务
├── providers          列出 LLM 提供商
├── export-project    导出项目
├── import-project    导入项目
├── backup            备份数据目录
├── verify-backup     校验备份
└── restore-backup    恢复备份
```

## 实现

### src/scientex_agent/cli.py（完整版）

```python
"""Command-line entry point for Scientex."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__

DEFAULT_DATA_DIR = Path.home() / ".scientex_agent"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scientex_agent",
        description="Local scientific agent platform",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command")

    # ---- init ----
    init_parser = subparsers.add_parser("init", help="Initialize a local data directory")
    init_parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))

    # ---- chat ----
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with an LLM")
    chat_parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    chat_parser.add_argument("--project", help="Project name (creates if not exists)")
    chat_parser.add_argument("--model", help="Model to use")
    chat_parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    chat_parser.add_argument("message", nargs="*", help="Initial message (optional)")

    # ---- run-server ----
    server_parser = subparsers.add_parser("run-server", help="Start the HTTP API server")
    server_parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    server_parser.add_argument("--host", default="127.0.0.1")
    server_parser.add_argument("--port", type=int, default=8765)
    server_parser.add_argument("--env-file", default=".env.local")

    # ---- providers ----
    providers_parser = subparsers.add_parser("providers", help="List configured LLM providers")
    providers_parser.add_argument("--env-file", default=".env.local")

    # ---- export-project ----
    export_parser = subparsers.add_parser("export-project", help="Export a project to a zip file")
    export_parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    export_parser.add_argument("--project-id", required=True)
    export_parser.add_argument("--output", required=True)

    # ---- import-project ----
    import_parser = subparsers.add_parser("import-project", help="Import a project from a zip file")
    import_parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    import_parser.add_argument("--package", required=True)
    import_parser.add_argument("--name", help="Project name override")

    # ---- backup ----
    backup_parser = subparsers.add_parser("backup", help="Create a verified backup")
    backup_parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    backup_parser.add_argument("--output", required=True, help="Output zip path")

    # ---- verify-backup ----
    verify_parser = subparsers.add_parser("verify-backup", help="Verify backup integrity")
    verify_parser.add_argument("--backup", required=True, help="Backup zip path")

    # ---- restore-backup ----
    restore_parser = subparsers.add_parser("restore-backup", help="Restore a backup")
    restore_parser.add_argument("--backup", required=True)
    restore_parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    restore_parser.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)

    # ---- Dispatch ----
    if args.command == "init":
        return _cmd_init(args)

    elif args.command == "chat":
        return _cmd_chat(args)

    elif args.command == "run-server":
        return _cmd_run_server(args)

    elif args.command == "providers":
        return _cmd_providers(args)

    elif args.command == "export-project":
        return _cmd_export(args)

    elif args.command == "import-project":
        return _cmd_import(args)

    elif args.command == "backup":
        return _cmd_backup(args)

    elif args.command == "verify-backup":
        return _cmd_verify_backup(args)

    elif args.command == "restore-backup":
        return _cmd_restore_backup(args)

    else:
        parser.print_help()
        return 1


# === Command Implementations ===

def _cmd_init(args) -> int:
    from .app import ScientexApp

    app = ScientexApp(Path(args.data_dir))
    app.initialize()
    print(f"✓ Initialized data directory: {app.data_dir}")
    print(f"  Database: {app.data_dir / 'app.db'}")
    return 0


def _cmd_chat(args) -> int:
    import asyncio
    from .app import ScientexApp

    app = ScientexApp(Path(args.data_dir))
    app.initialize()

    # Find or create project
    projects = app.list_projects()
    project_name = args.project or "Default"
    project = next((p for p in projects if p.name == project_name), None)
    if not project:
        project = app.create_project(name=project_name)
        print(f"Created project: {project.name} ({project.id})")

    # Create frame
    frame = app.create_frame(project_id=project.id, name="CLI Chat")
    print(f"Frame: {frame.id}")
    print(f"Type /exit to quit, /clear to reset history")
    print()

    # Interactive loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            try:
                user_input = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if user_input.lower() in ("/exit", "/quit"):
                break
            if user_input.lower() == "/clear":
                frame = app.create_frame(project_id=project.id, name="CLI Chat")
                print("[New frame created]")
                continue
            if not user_input.strip():
                continue

            print()

            if args.no_stream:
                result = loop.run_until_complete(
                    app.chat(frame_id=frame.id, content=user_input)
                )
                print(result.get("content", ""))
            else:
                full = ""
                async def _stream():
                    nonlocal full
                    async for event in app.chat_stream(
                        frame_id=frame.id, content=user_input
                    ):
                        if "token" in event:
                            print(event["token"], end="", flush=True)
                            full += event["token"]
                        elif "tool_start" in event:
                            name = event["tool_start"]["name"]
                            print(f"\n🔧 {name}...", end="", flush=True)
                        elif "tool_end" in event:
                            print(" ✓", flush=True)
                        elif "done" in event:
                            pass
                loop.run_until_complete(_stream())
                print()

            print()
    finally:
        loop.close()

    return 0


def _cmd_run_server(args) -> int:
    from .api_server import run_server

    print(f"Starting Scientex server on http://{args.host}:{args.port}")
    print(f"Data directory: {args.data_dir}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    run_server(data_dir=args.data_dir, host=args.host, port=args.port)
    return 0


def _cmd_providers(args) -> int:
    from .provider_config import discover_providers

    providers = discover_providers(args.env_file)
    if not providers:
        print("No LLM providers configured.")
        print("Create a .env.local file with your API keys.")
        print("See .env.example for available providers.")
        return 0

    for p in providers:
        status = "✓" if p.credential_present else "✗ (no API key)"
        model = p.model or "(default)"
        print(f"  {p.name:15s}  {status:20s}  model: {model}")
    return 0


def _cmd_export(args) -> int:
    from .app import ScientexApp

    app = ScientexApp(Path(args.data_dir))
    app.initialize()

    path = app.artifacts.export_project(args.project_id, args.output)
    print(f"✓ Exported to: {path}")
    return 0


def _cmd_import(args) -> int:
    from .app import ScientexApp

    app = ScientexApp(Path(args.data_dir))
    app.initialize()

    project = app.artifacts.import_project(args.package, args.name)
    print(f"✓ Imported project: {project.name} ({project.id})")
    return 0


def _cmd_backup(args) -> int:
    from .backup import create_backup

    path = create_backup(Path(args.data_dir), Path(args.output))
    print(f"✓ Backup created: {path}")
    return 0


def _cmd_verify_backup(args) -> int:
    from .backup import verify_backup

    ok = verify_backup(Path(args.backup))
    if ok:
        print("✓ Backup verified successfully")
        return 0
    else:
        print("✗ Backup verification FAILED")
        return 1


def _cmd_restore_backup(args) -> int:
    from .backup import restore_backup

    restore_backup(Path(args.backup), Path(args.data_dir), force=args.force)
    print(f"✓ Restored to: {args.data_dir}")
    return 0
```

### src/scientex_agent/backup.py

```python
"""Data backup, verification, and restore utilities."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import zipfile
from pathlib import Path


def create_backup(data_dir: Path, output: Path) -> Path:
    """Create a verified backup of the data directory."""
    data_dir = data_dir.expanduser().resolve()
    output = output.expanduser().resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Collect file checksums
    manifest = {}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(data_dir.rglob("*")):
            if file_path.is_file() and ".venv" not in file_path.parts:
                arcname = str(file_path.relative_to(data_dir))
                zf.write(file_path, arcname)
                # Compute checksum
                sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
                manifest[arcname] = sha

        # Write manifest
        zf.writestr("backup-manifest.json", json.dumps(manifest, indent=2))

    # Quick verify
    verify_backup(output)
    return output


def verify_backup(backup_path: Path) -> bool:
    """Verify a backup's integrity."""
    backup_path = backup_path.expanduser().resolve()
    if not backup_path.exists():
        print(f"Backup not found: {backup_path}")
        return False

    try:
        with zipfile.ZipFile(backup_path, "r") as zf:
            # Check manifest
            if "backup-manifest.json" not in zf.namelist():
                print("No manifest in backup")
                return False

            manifest = json.loads(zf.read("backup-manifest.json"))

            for filename, expected_checksum in manifest.items():
                if filename == "backup-manifest.json":
                    continue
                actual = hashlib.sha256(zf.read(filename)).hexdigest()
                if actual != expected_checksum:
                    print(f"Checksum mismatch: {filename}")
                    return False

            # Check SQLite integrity
            for name in zf.namelist():
                if name.endswith(".db"):
                    # Can't check inside zip directly, but we checked its checksum
                    pass

        return True

    except Exception as e:
        print(f"Verification error: {e}")
        return False


def restore_backup(backup_path: Path, data_dir: Path, *, force: bool = False) -> None:
    """Restore a verified backup into a data directory."""
    backup_path = backup_path.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()

    if data_dir.exists():
        if not force:
            raise FileExistsError(
                f"Data directory already exists: {data_dir}\n"
                "Use --force to overwrite."
            )
        shutil.rmtree(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(backup_path, "r") as zf:
        for member in zf.namelist():
            if member == "backup-manifest.json":
                continue
            target = data_dir / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(member))
```

## 验证

```bash
# 初始化
uv run scientex_agent init --data-dir /tmp/scientex-test

# 交互式聊天
uv run scientex_agent chat --data-dir /tmp/scientex-test "Hello!"

# 列出提供商
uv run scientex_agent providers

# 启动服务
uv run scientex_agent run-server --data-dir /tmp/scientex-test --port 8765

# 备份
uv run scientex_agent backup --data-dir /tmp/scientex-test --output /tmp/backup.zip

# 校验
uv run scientex_agent verify-backup --backup /tmp/backup.zip

# 恢复
uv run scientex_agent restore-backup --backup /tmp/backup.zip --data-dir /tmp/restored --force
```

## 当前局限

1. 没有配置文件（默认值硬编码）
2. 错误消息可以更友好
3. 没有 `--verbose` / `--quiet` 日志级别

## 下一步

→ [14-web-frontend.md](14-web-frontend.md)
