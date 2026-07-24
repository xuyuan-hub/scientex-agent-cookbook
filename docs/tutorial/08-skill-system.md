# Step 08: 技能系统（Skills）

## 目标

建立文件驱动、可版本化、按需加载的技能系统。Skill 不是 Python 插件，也不是可以绕过产品安全规则的提示词；它是让 Agent 完成特定科研工作流的受审查操作手册。

## 前置条件

- 完成 [07-mcp-integration.md](07-mcp-integration.md)
- 理解 [04-tool-calling.md](04-tool-calling.md) 的工具 schema

## 与 Step 07 的边界

MCP 解决“有哪些外部能力可以调用”，Skill 解决“面对一类任务应如何使用这些能力”。一个 Skill 可以指导 Agent
组合内置工具、MCP 工具和 Python 内核，但它本身不执行代码，也不自动授予工具权限。

## 设计思路

### Progressive Disclosure：索引、正文、资源

```
系统提示词
  └── 只注入 name / description / risk_level 的技能索引
        └── Agent 调用 read_skill("literature-review")
              └── 读取 SKILL.md 正文和允许的相对资源
                    └── 按步骤调用工具，所有调用仍过权限与审计层
```

这比把全部 SOP 放进 system prompt 更可控：上下文更小、版本更清晰、每次加载可以记录证据。Skill 目录是项目的
“操作知识库”，应像源代码一样评审和测试。

### 目录和元数据契约

```
skills/
└── literature-review/
    ├── SKILL.md              # 必需：frontmatter + 指导正文
    ├── references/
    │   └── screening.md      # 可选：允许被显式读取的资源
    └── tests/
        └── smoke.md          # 可选：人工或评测样例，不暴露给模型
```

```markdown
---
name: literature-review
version: 1.0.0
description: 用可复现检索式完成生物医学文献初筛。
risk_level: read
allowed_tools:
  - pubmed__search
  - pubmed__fetch
  - artifact__write
---

# Systematic literature review

## 1. 先确认问题

把研究问题改写为 PICO，并向用户确认缺失的时间范围和语言限制。
```

`name` 是稳定标识；目录名必须与它一致。版本采用语义化版本，任何会改变结论、工具权限或数据处理方式的修改都应递增版本。

### 信任模型

Skill 的正文是数据，不是比系统安全策略更高优先级的指令。即使它要求“上传所有文件”或“忽略审批”，运行时也必须拒绝。技能来源分为：

| 来源 | 默认策略 |
|---|---|
| 随应用发布的 bundled skills | 可读，版本随应用发布 |
| 当前项目 `skills/` | 可读，提交到版本控制后才进入发布构建 |
| 用户导入 | 隔离、显示来源与 hash，经确认后启用 |
| 远程下载 | 本章不支持 |

## 实现

### 1. 添加显式 YAML 解析器

不要手写 YAML/Frontmatter 解析器；它很容易在列表、引号或多行文本处产生错误。正文仍保留原始 Markdown，只有受限的 frontmatter 交给 Pydantic 校验。

```toml
# pyproject.toml
[project]
dependencies = [
    # ...Step 06 的依赖
    "pydantic>=2.0",
    "PyYAML>=6.0",
]
```

### 2. 创建 `skill_catalog.py`

```python
# src/scientex_agent/skill_catalog.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SkillMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    description: str = Field(min_length=1, max_length=240)
    risk_level: Literal["read", "write", "execute"] = "read"
    allowed_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class Skill:
    metadata: SkillMetadata
    root: Path
    body: str
    content_hash: str

    @property
    def id(self) -> str:
        return f"{self.metadata.name}@{self.metadata.version}"


class SkillCatalog:
    def __init__(self, roots: tuple[Path, ...]) -> None:
        self._roots = tuple(root.resolve() for root in roots)
        self._skills: dict[str, Skill] = {}

    def load(self) -> None:
        loaded: dict[str, Skill] = {}
        for root in self._roots:
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*/SKILL.md")):
                skill = self._parse(path)
                if skill.metadata.name in loaded:
                    raise ValueError(f"duplicate skill name: {skill.metadata.name}")
                loaded[skill.metadata.name] = skill
        self._skills = loaded

    def list(self) -> list[dict[str, str]]:
        return [
            {"name": skill.metadata.name, "version": skill.metadata.version,
             "description": skill.metadata.description,
             "risk_level": skill.metadata.risk_level}
            for skill in self._skills.values()
        ]

    def read(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as error:
            raise KeyError(f"unknown skill: {name}") from error

    def read_resource(self, name: str, relative_path: str) -> str:
        skill = self.read(name)
        resource = (skill.root / "references" / relative_path).resolve()
        allowed_root = (skill.root / "references").resolve()
        if not resource.is_relative_to(allowed_root) or not resource.is_file():
            raise ValueError("skill resource must stay under references/")
        return resource.read_text(encoding="utf-8")

    @staticmethod
    def _parse(path: Path) -> Skill:
        raw = path.read_text(encoding="utf-8")
        if not raw.startswith("---\n"):
            raise ValueError(f"{path}: missing YAML frontmatter")
        _, frontmatter, body = raw.split("---", 2)
        metadata = SkillMetadata.model_validate(yaml.safe_load(frontmatter))
        root = path.parent.resolve()
        if root.name != metadata.name:
            raise ValueError(f"{path}: directory and skill name must match")
        return Skill(
            metadata=metadata,
            root=root,
            body=body.lstrip("\n"),
            content_hash=hashlib.sha256(raw.encode()).hexdigest(),
        )
```

这里有两个刻意的限制：`read_resource()` 只允许 `references/` 下的文件，且使用 `resolve()` + `is_relative_to()` 防止
`../../` 路径穿越；`load()` 遇到重名立即报错，而不是让目录扫描顺序悄悄决定覆盖关系。

### 3. 暴露最小工具面

```python
# src/scientex_agent/skill_tools.py
from typing import Any


def list_skills(catalog: SkillCatalog) -> dict[str, Any]:
    """返回小型索引，适合放入 Agent 上下文。"""
    return {"skills": catalog.list()}


def read_skill(catalog: SkillCatalog, name: str) -> dict[str, Any]:
    skill = catalog.read(name)
    return {
        "id": skill.id,
        "hash": skill.content_hash,
        "allowed_tools": list(skill.metadata.allowed_tools),
        "body": skill.body,
    }
```

工具注册时可将 `allowed_tools` 传给 Step 10 的策略节点。它是**更严格的白名单**，不是全局工具目录的替代品：最终调用仍需同时满足项目策略、用户权限和 MCP Server 的风险等级。

### 4. 构建稳定的系统提示索引

```python
def skill_index_prompt(catalog: SkillCatalog) -> str:
    lines = ["Available skills (read one only when useful):"]
    for item in catalog.list():
        lines.append(
            f"- {item['name']}@{item['version']} [{item['risk_level']}]: "
            f"{item['description']}"
        )
    return "\n".join(lines)
```

完整正文通过 `read_skill` 按需取得。把 `id` 和 SHA-256 hash 写入 run event，之后即使 Skill 升级，也能重现某次回答所遵循的版本。

## 验证

```bash
# 载入并检查所有 frontmatter；失败时显示文件和字段错误
uv run scientex_agent skills validate skills/

# 只显示索引，不打印完整正文
uv run scientex_agent skills list

# 读取某个确切版本的正文和 hash
uv run scientex_agent skills show literature-review
```

建议至少写四个单元测试：有效目录可加载、重复 name 失败、错误 frontmatter 失败、`../secret.txt` 无法通过
`read_resource()` 读取。再增加一个评测样例：给出文献问题时，Agent 先加载合适 Skill，再按其中的检索模板调用只读工具。

## 深入理解

### Skill 与 Prompt Template 的区别

Prompt template 通常是一次性生成文本的参数化模板；Skill 是长期维护的工作流知识，含风险声明、可用工具和可验证步骤。两者可以共存，但不要把可执行权限藏在任意 Markdown 里。

### 技能变更如何审查

将 `skills/` 放入 Git，PR 中显示正文 diff 和 frontmatter diff。尤其审查 `allowed_tools`、`risk_level`、外部链接、文件写入和实验性结论。发布时生成 catalog 清单（name、version、hash）并随应用一起保存。

## 当前局限

- 只支持本地 Markdown Skill，不提供在线市场、自动下载或自动安装。
- 对正文只做结构和路径安全校验，不自动验证科学事实；Step 15 引入证据与验证工作流。
- Skill 的效果评估还只是手工样例；Step 16 会把评测、lint 和发布检查接入 CI。

## 下一步

下一章实现受控 Python 内核。Skill 可以提供分析步骤和参考资源，但代码执行必须在独立、安全边界明确的运行时中完成。
