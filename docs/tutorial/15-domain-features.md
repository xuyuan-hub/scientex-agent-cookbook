# Step 15: 领域功能（科学工作流、记忆与验证）

## 目标

在通用 Agent 平台上增加可追溯的科研领域能力：受 schema 约束的分析工具、项目范围记忆、结论证据与验证状态。目标不是让模型“看起来懂科学”，而是让每个可交付结论能追到输入、方法、工具版本和证据。

## 前置条件

- 完成 [14-web-frontend.md](14-web-frontend.md)
- 完成 [12-project-management.md](12-project-management.md) 的 Artifact 与 provenance
- 了解 [08-skill-system.md](08-skill-system.md) 的 Skill 风险模型

## 设计思路

### 领域能力是垂直切片

```
用户问题
  → Skill：选择 SOP 与允许的工具
  → Domain tool：Pydantic 输入 / 输出、确定性实现、版本
  → Artifact：原始输入、结果、图表、参数
  → Verification：声明、证据、状态
  → Vue：结果、来源、警告、可复现入口
```

不要把生物信息学算法直接写进 prompt，也不要让前端重新计算研究结论。领域工具封装在 Python 服务中，使用清晰 schema 和独立测试；工具调用结果只是一条消息，真正的结果要保存为 Artifact version。

### 三类记忆，三种权限

| 类型 | 范围 | 例子 | 默认策略 |
|---|---|---|---|
| 短期工作记忆 | 单一 Frame | 当前检索式、待审批工具 | graph checkpoint，Frame 结束后可归档 |
| 项目记忆 | 单一 Project | 样本命名、研究目标、已确认偏好 | 用户可查看、编辑、删除 |
| 用户长期偏好 | 跨 Project | 默认语言、单位偏好 | 明确 opt-in，绝不从敏感原始数据自动推断 |

记忆不是聊天全文的另一个副本。它必须有来源、置信度、有效期与删除机制；检索到的记忆只作为上下文提示，不能覆盖当前用户输入或证据。

### 验证状态不是“真实 / 虚假”二元标签

对科研回答更诚实的模型是：

```
draft → supported | insufficient_evidence | contradicted | needs_review
```

状态来自可检查的 evidence，且显示范围和局限。例如“在已检索的 25 篇文章中有 18 篇支持”不等价于“科学上已证实”。

## 实现

### 1. 用 Pydantic 定义领域工具契约

以下以 DNA 序列基础统计为例。它是确定性的、无网络、只读工具，适合作为第一个垂直切片；真实测序/数据库工具沿用同一结构。

```python
# src/scientex_agent/domain/sequence_stats.py
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SequenceStatsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: str = Field(min_length=1, max_length=5_000_000)
    alphabet: Literal["DNA"] = "DNA"

    @field_validator("sequence")
    @classmethod
    def normalize_sequence(cls, value: str) -> str:
        normalized = "".join(value.upper().split())
        invalid = sorted(set(normalized) - set("ACGTN"))
        if invalid:
            raise ValueError(f"invalid DNA bases: {''.join(invalid)}")
        return normalized


class SequenceStatsOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    length: int
    gc_fraction: float | None
    counts: dict[str, int]
    sequence_sha256: str
    tool_version: str = "1.0.0"


def sequence_stats(data: SequenceStatsInput) -> SequenceStatsOutput:
    counts = Counter(data.sequence)
    defined = counts["A"] + counts["C"] + counts["G"] + counts["T"]
    return SequenceStatsOutput(
        length=len(data.sequence),
        gc_fraction=(counts["G"] + counts["C"]) / defined if defined else None,
        counts={base: counts[base] for base in "ACGTN"},
        sequence_sha256=hashlib.sha256(data.sequence.encode()).hexdigest(),
    )
```

工具注册记录风险、版本和 IO schema：

```python
registry.register(
    name="sequence__stats",
    description="Calculate length and GC fraction for a DNA sequence.",
    input_model=SequenceStatsInput,
    output_model=SequenceStatsOutput,
    risk="read",
    version="1.0.0",
    handler=sequence_stats,
)
```

请求大序列时，优先传 Artifact version id，而不是把 MB 级 FASTA 粘进模型上下文。工具读取该 Artifact 后将输入 hash、版本与输出写入 provenance。

### 2. 以 Artifact 表达可复现结果

```python
# domain service 中的调用顺序
input_version = artifacts.require_readable(frame.project_id, request.input_artifact_id)
result = sequence_stats(SequenceStatsInput(sequence=read_fasta(input_version)))
result_version = artifacts.write_json(
    project_id=frame.project_id,
    logical_path="analysis/sequence-stats.json",
    value=result.model_dump(mode="json"),
    run_id=run_id,
    metadata={
        "tool": "sequence__stats",
        "tool_version": result.tool_version,
        "input_artifact_version_id": input_version.id,
    },
)
```

UI 显示的是结构化结果和 Artifact 链接，而不是只显示“GC 含量为 52%”。用户可以下载 JSON、查看输入版本和重新运行参数。

### 3. 建立可编辑、带来源的项目记忆

```python
# src/scientex_agent/memory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ProjectMemory:
    id: str
    project_id: str
    kind: Literal["fact", "preference", "decision"]
    content: str
    source_artifact_version_id: str | None
    source_message_id: str | None
    confidence: float
    status: Literal["proposed", "confirmed", "rejected", "expired"]
    expires_at: int | None
```

写入路径应为“模型提出 → 用户确认或规则验证 → 保存”。例如 Agent 发现“样本 A 使用 hg38”时，先创建 `proposed` memory；用户在 Vue 中确认后才可作为后续默认上下文。每次检索仅返回当前 Project 中 `confirmed`、未过期、与任务相关的少量条目，并把 memory id 写入 Run provenance。

### 4. 将结论与证据分开保存

```python
# src/scientex_agent/verification.py
from pydantic import BaseModel, Field
from typing import Literal


class Evidence(BaseModel):
    artifact_version_id: str | None = None
    external_uri: str | None = None
    quote: str = Field(max_length=1_000)
    retrieved_at: int
    relevance: float = Field(ge=0, le=1)


class Claim(BaseModel):
    text: str = Field(min_length=1, max_length=2_000)
    status: Literal["draft", "supported", "insufficient_evidence", "contradicted", "needs_review"]
    evidence: list[Evidence] = []
    scope_note: str = ""
```

事实验证不是“让第二个 LLM 投票”。流程应是：提取可检验 claim → 检索/选择证据 → 记录来源版本与摘录 → 按可解释规则给出状态 → 高风险或矛盾情况交给专家。LLM 可以辅助提取和概述，但不能伪造引用或把未检索到的结果写成支持证据。

### 5. 用 Profile 管理可复现运行配置

```python
class AgentProfile(BaseModel):
    id: str
    version: str
    provider: str
    model: str
    system_prompt_template: str
    enabled_skill_ids: tuple[str, ...]
    enabled_tool_names: tuple[str, ...]
    max_tool_rounds: int = 8
```

Profile 是可审查、版本化的运行配置，不是用户可随意提交的一段 prompt。每个 Run 固化 profile id/version、provider/model、Skill hash、tool version、输入 Artifact 与配置 hash；这样“重跑”才有明确含义。

## 验证

```bash
# 对确定性工具的输入/输出运行 schema 校验
uv run pytest tests/domain/test_sequence_stats.py

# 将一个分析结果保存为 Artifact 并展示来源
uv run scientex_agent tools call sequence__stats '{"sequence":"ACGTNN"}'
uv run scientex_agent artifacts history --project PROJECT_ID analysis/sequence-stats.json

# 记忆先提议再确认
uv run scientex_agent memories propose --project PROJECT_ID --kind fact \
  --content "Reference genome is GRCh38"
uv run scientex_agent memories confirm MEMORY_ID
```

测试除了正确结果，还应覆盖：非法碱基、纯 N 序列的除零情况、相同输入 hash、Artifact provenance、跨 Project 记忆隔离、过期记忆不被检索、缺少 evidence 时 claim 不得标记为 `supported`。

## 深入理解

### 科学工具的版本为什么重要

同一个工具名在算法、参考数据库、阈值或依赖版本改变后可能给出不同结论。工具 version、容器 image digest、数据库 release、参数和输入 hash 应一同进入结果元数据；没有它们的“可复现”只是口号。

### 隐私与敏感数据

序列、患者信息和实验记录可能具有敏感性。最小化发送给 provider/MCP 的字段，允许项目禁用远程工具与长期记忆，并在界面显示数据离开本机前的目标与范围。真实医疗或临床使用还需要组织级合规评估，本项目不以该教程替代它。

## 当前局限

- 本章只提供一个确定性序列工具样例，未实现完整的 FASTQ、变异或结构生物学流水线。
- Claim 状态辅助用户判断，不能替代同行评审、实验验证或临床决策。
- 向量检索、远程文献库和多模态图表可以后续接入，但必须沿用 Artifact、来源和权限边界。

## 下一步

最后一章把测试、配置、安全、迁移、可观测性和发布流程收束为生产就绪门槛，并给出本地与服务端部署的不同边界。
