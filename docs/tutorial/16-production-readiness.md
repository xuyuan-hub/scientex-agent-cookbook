# Step 16: 生产化与交付

## 目标

把前十五章的演示能力收束为可交付的软件：有明确部署边界、可迁移的数据、可观测的运行、可验证的构建和可回滚的发布。生产化不是“加一个日志文件”；它要求安全、质量、运维和用户恢复路径同时成立。

## 前置条件

- 完成 [15-domain-features.md](15-domain-features.md)
- 运行过后端、Vue 前端和端到端测试
- 明确目标部署模式：单用户本地，或受控服务端

## 设计思路

### 先选择部署画像

| 画像 | 适用场景 | 数据与安全边界 |
|---|---|---|
| 本地单用户 | 个人科研工作站 | SQLite + 本地 blob；浏览器仅监听 loopback；本地可信 kernel |
| 受控团队服务 | 实验室/组织内部 | 身份认证、TLS、反向代理、持久卷、审计；不可信 kernel 使用隔离运行器 |
| 公网多租户 | 不在本教程的交付范围 | 需要专门的身份、配额、隔离、密钥、合规与安全评审 |

不要把“本地默认配置”暴露到公网后仍称为生产。尤其是 MCP、Python kernel、Artifact 下载和 LLM key，在受控团队服务中都必须重新评估权限模型。

### 交付链路

```
源码 + Skill + API schema
  → ruff / typecheck / pytest / Vitest / Playwright
  → 构建 Python wheel + Vue dist
  → SBOM / 依赖与许可证检查
  → 可复现发布产物
  → 部署前 migrate + backup
  → 健康检查、结构化日志、指标、告警
```

每次发布都能回答：使用了哪个 Git revision、哪个前端构建、哪个数据库 schema、哪些 Skill hash；出现问题时也能回到上一个版本而不破坏用户数据。

## 实现

### 1. 集中类型化配置

```python
# src/scientex_agent/settings.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCIENTEX_",
        env_file=".env",
        extra="ignore",
    )

    environment: Literal["development", "local", "server"] = "local"
    data_dir: Path = Path.home() / ".local" / "share" / "scientex"
    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8765, ge=1, le=65535)
    web_dist: Path | None = None
    allowed_origins: tuple[AnyHttpUrl, ...] = ()
    openai_api_key: SecretStr | None = None
    database_url: str | None = None

    @property
    def is_server(self) -> bool:
        return self.environment == "server"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

将 `pydantic-settings>=2.0` 加入运行时依赖；`pydantic` 本身不包含这个配置包。

`.env`、数据目录、SQLite、blob、日志和导出包必须写入 `.gitignore`。`SecretStr` 防止意外 `repr()` 输出，但不是密钥管理系统：服务器应从部署平台的 secret store 注入，轮换密钥时不用重建前端。

### 2. 结构化日志、trace 与健康检查

```python
# 统一日志字段；实现可采用 structlog 或标准 logging 的 JSON formatter
logger.info(
    "tool.completed",
    extra={
        "trace_id": trace_id,
        "run_id": run_id,
        "frame_id": frame_id,
        "tool": tool_name,
        "duration_ms": elapsed_ms,
        "outcome": "ok",
    },
)
```

日志默认不记录 prompt 全文、MCP 参数、序列内容、Artifact 内容、cookie 或 API key。对异常文本先经过 redact，再记录到受权限保护的日志位置。HTTP middleware 创建或接收 `X-Request-Id`，并把同一个 trace id 传播到 RunEvent、provider 调用和工具执行。

健康检查分层：

```
GET /api/v1/health/live   → 进程存活，不访问外部依赖
GET /api/v1/health/ready  → 数据库可写、必要配置和关键依赖就绪
```

不要让 load balancer 的 liveness 检查因为临时 provider 网络故障而无限重启服务；provider 可用性应作为单独指标或 readiness 子项。

### 3. 错误、安全和权限基线

- 所有 API 使用 Step 11 的 Problem Details；stack trace 只入日志。
- 所有外部输入用 Pydantic 校验，所有逻辑路径经过 Artifact path 规范化。
- MCP Server、Skill、Profile 和 kernel image 采用 allowlist；不从模型输出构造 shell 命令。
- 服务端启用 TLS、身份认证、项目级授权与 CSRF/Origin 策略；本地 loopback 模式不把这些假设带到公网。
- 写入、执行、网络和外部副作用工具由 `ToolPolicy` 记录审批决定；限流由反向代理或共享存储实现，不使用进程内字典伪装成集群限流。
- 上传、导入 ZIP 和下载 Artifact 均有大小、类型、hash、路径与权限检查。

安全评审必须把 Python kernel 单列：`trusted-local` 只允许可信本地用户；服务端的任何不可信代码只能进入隔离运行器，且默认无网络、最小文件挂载、非 root、资源上限和可销毁文件系统。

### 4. 版本化迁移与备份恢复

建立单独 schema 版本表，迁移在应用启动前的显式命令中运行，而不是 HTTP 请求期间隐式 `CREATE TABLE`：

```bash
uv run scientex_agent migrate status
uv run scientex_agent migrate apply
uv run scientex_agent backup create --output backups/scientex-2026-07-24.zip
uv run scientex_agent backup verify backups/scientex-2026-07-24.zip
```

每个迁移必须有唯一递增编号、事务说明、数据回填策略和测试。SQLite 备份使用 SQLite backup API 或一致性快照；不能在 WAL 写入时直接复制一个 `.sqlite` 文件。备份包包含数据库、引用 blob、manifest、schema version 与 checksum；恢复先验证到临时位置，再经用户确认原子切换。发布前执行一次真实的“备份 → 新环境恢复 → smoke test”。

### 5. 构建前后端发布物

```toml
# pyproject.toml（节选）
[project]
requires-python = ">=3.13"

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

```bash
# 后端
uv sync --all-groups
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest --cov=scientex_agent

# 前端
cd web
pnpm install --frozen-lockfile
pnpm lint
pnpm build
pnpm test:unit
pnpm exec playwright test
```

发布脚本应先运行 `pnpm build`，再将 `web/dist` 复制或打包到 Python wheel 的 package data 中。开发服务器不属于发布物。构建应生成版本信息（Git revision、构建时间、前端 asset manifest）和 SBOM；依赖漏洞扫描/许可证策略在 CI 中作为门禁。

### 6. CI 的最小门禁

```yaml
# .github/workflows/ci.yml（阶段示意）
jobs:
  backend:
    steps: [checkout, setup-python, uv-sync, ruff, pyright, pytest]
  frontend:
    steps: [checkout, setup-node, pnpm-install-frozen, lint, build, vitest]
  e2e:
    needs: [backend, frontend]
    steps: [start-api, start-preview, playwright]
  package:
    needs: [backend, frontend, e2e]
    steps: [build-wheel, install-wheel-in-clean-env, smoke-test]
```

CI 还应保存 API OpenAPI schema 与前端截图/trace 作为失败诊断附件。不要只测试源码目录下的 Python：必须从构建出的 wheel 在干净环境启动一次，才能发现漏打包 Skill、静态资源或迁移文件。

### 7. 发布检查清单

- [ ] 所有测试、lint、typecheck、构建和依赖检查通过。
- [ ] OpenAPI schema、Skill catalog、Profile 与数据库迁移均已审查。
- [ ] 备份已验证，恢复演练通过，升级/回滚步骤已记录。
- [ ] API key 与生产数据不在仓库、日志、截图或前端 bundle 中。
- [ ] `live` 和 `ready` 健康检查、结构化日志和告警接收方可用。
- [ ] 本次版本、Git revision、schema version、前端构建 hash 已写入 release note。
- [ ] 服务端部署使用 TLS、认证、最小权限目录和受隔离的 kernel。

## 验证

```bash
# 从干净环境安装并运行打包结果
uv build
uv venv /tmp/scientex-smoke
/tmp/scientex-smoke/bin/pip install dist/*.whl
/tmp/scientex-smoke/bin/scientex_agent init --data-dir /tmp/scientex-data

# 数据恢复演练
uv run scientex_agent backup create --output /tmp/scientex-backup.zip
uv run scientex_agent backup verify /tmp/scientex-backup.zip
uv run scientex_agent backup restore /tmp/scientex-backup.zip --data-dir /tmp/scientex-restore --yes

# 前端生产预览
cd web && pnpm build && pnpm vite preview
```

验收不是只看页面能打开：创建 Project、发起流式 Run、拒绝高风险工具、生成 Artifact、导出、停止服务、恢复备份后都应可验证。对 server profile，还要验证未认证请求、跨项目访问、超大上传、恶意 ZIP、MCP 失联和 kernel 超时等失败路径。

## 深入理解

### 可观测性与隐私的平衡

为了排障，需要 run id、时延、错误类型、工具名和版本；为了保护研究数据，不应把完整 prompt、序列和文件内容送入遥测。先定义事件字段和保留期，再选择日志/指标/trace 后端。可观测性系统本身也是数据处理者，需要访问控制与脱敏测试。

### 单机 SQLite 的上线边界

SQLite 在单用户或单进程服务中非常可靠，且便于备份；当需要多实例写入、跨机器高可用或复杂并发协作时，应迁移 metadata/checkpoint 到服务型数据库，blob 到对象存储。不要仅通过增大连接池来掩盖底层写入模型不匹配。

## 完成

至此，Scientex 已从一次 LLM 调用演进为具备 provider 抽象、持久化、MCP、Skill、受控执行、可恢复 Agent、FastAPI、**Vue 3 前端**、Artifact 证据链和发布门禁的科学 Agent 平台。

后续迭代应继续遵循本教程的原则：先定义边界与可验证契约，再实现能力；任何自动化都不应绕过用户数据、工具权限和科学证据的审查。
