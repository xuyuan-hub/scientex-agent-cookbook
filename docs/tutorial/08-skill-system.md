# Step 08: 技能系统（Skills）

## 目标

实现基于文件的技能目录系统，让 agent 可以加载领域知识来指导行为。

## 前置条件

- 完成 [07-mcp-integration.md](07-mcp-integration.md)
- 熟悉工具调用 [04-tool-calling.md](04-tool-calling.md)

## 设计思路

### 什么是 Skill

Skill 是**给 LLM 的领域指导文档**，告诉它如何完成特定任务。

它与 tool 的区别：

| | Tool | Skill |
|---|---|---|
| 是什么 | 可执行的函数 | 指导文档 |
| LLM 怎么用 | 调用 `tool_name(args)` | 先读文档，再按指导操作 |
| 例子 | `calculate("2+3")` | "如何进行文献综述" |
| 存储 | Python 代码 | Markdown 文件 |

### Progressive Disclosure 模式

```
Level 1: System prompt 中只有 skill 的名称 + 一句话描述
  → LLM 知道有哪些技能，但不知道细节

Level 2: LLM 调用 read_skill("literature-review") 读取完整指导
  → LLM 获取详细的步骤说明

Level 3: LLM 按指导执行，调用相关工具
  → search_papers → read_paper → extract_findings → ...
```

**好处**：system prompt 不会因技能太多而爆炸，LLM 按需加载。

### 文件格式

```
skills/
└── literature-review/
    ├── SKILL.md          # 必须：技能指导文档
    ├── references.md     # 可选：参考资料
    └── kernel.py         # 可选：注入 Python 内核的辅助函数
```

`SKILL.md` 格式：
```markdown
---
name: literature-review
description: How to search and synthesize scientific literature.
license: CC-BY-4.0
tags: literature, search, synthesis
---

# Literature Review

## Step 1: Define your research question
...

## Step 2: Search databases
Use `search_papers` tool with the following strategy:
...

## Step 3: Screen results
...
```

## 实现

### 1. 创建 src/scientex_agent/skill_catalog.py

```python
"""File-based skill catalog with progressive disclosure."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable


# Default skills directory (bundled with the package)
DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent / "skills"


@dataclass(frozen=True)
class Skill:
    """A parsed skill definition."""
    name: str
    description: str
    tags: tuple[str, ...] = ()
    body: str = ""                         # Full markdown content
    license: str = ""
    category: str = ""
    requirements: tuple[str, ...] = ()
    source_path: str = ""                  # Path to SKILL.md


class SkillCatalog:
    """Loads and indexes skills from directories of SKILL.md files.

    Usage::

        catalog = SkillCatalog()
        catalog.load()  # scans DEFAULT_SKILLS_DIR

        # List all skills (name + description only)
        for skill in catalog.list_skills():
            print(skill["name"], skill["description"])

        # Read a full skill body
        body = catalog.read_skill("literature-review")

        # Search for skills
        results = catalog.search("protein")
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        *,
        extra_directories: Iterable[str | Path] = (),
    ) -> None:
        self.directory = Path(directory) if directory else DEFAULT_SKILLS_DIR
        self.extra_directories = tuple(Path(d) for d in extra_directories)
        self._skills: dict[str, Skill] = {}
        self._cache_signature: tuple | None = None

    def load(self) -> None:
        """(Re)load all skills from configured directories."""
        # Check if cache is valid
        signature = self._compute_signature()
        if signature == self._cache_signature:
            return

        self._skills.clear()
        self._scan_directory(self.directory)

        for extra_dir in self.extra_directories:
            if extra_dir.exists():
                self._scan_directory(extra_dir)

        self._cache_signature = signature

    def _scan_directory(self, directory: Path) -> None:
        """Scan one directory for skill subdirectories."""
        if not directory.exists():
            return

        for skill_dir in sorted(directory.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            skill = self._parse_skill(skill_md)
            if skill:
                # Later directories override earlier ones
                self._skills[skill.name] = skill

    def _parse_skill(self, path: Path) -> Skill | None:
        """Parse a SKILL.md file."""
        text = path.read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter(text)
        if not frontmatter:
            return None

        name = frontmatter.get("name", "")
        if not name:
            return None

        tags = _parse_tags(frontmatter.get("tags", ""))
        requirements = _parse_tags(frontmatter.get("requirements", ""))

        return Skill(
            name=name,
            description=frontmatter.get("description", ""),
            tags=tuple(tags),
            body=body,
            license=frontmatter.get("license", ""),
            category=frontmatter.get("category", ""),
            requirements=tuple(requirements),
            source_path=str(path),
        )

    def _compute_signature(self) -> tuple:
        """Compute a cache signature from directory stats."""
        parts = []
        for directory in (self.directory, *self.extra_directories):
            if not directory.exists():
                continue
            for skill_dir in sorted(directory.iterdir()):
                skill_md = skill_dir / "SKILL.md"
                if skill_md.exists():
                    stat = skill_md.stat()
                    parts.append((str(skill_md), stat.st_mtime, stat.st_size))
        return tuple(parts)

    # === Public API ===

    def list_skills(self) -> list[dict]:
        """Return summarized skill info (name + description + tags)."""
        self.load()
        return [
            {
                "name": s.name,
                "description": s.description,
                "tags": list(s.tags),
                "license": s.license,
                "category": s.category,
                "requirements": list(s.requirements),
            }
            for s in sorted(self._skills.values(), key=lambda s: s.name)
        ]

    def read_skill(self, name: str, *, filter_sections: str | None = None) -> str:
        """Read the full body of a skill.

        Args:
            name: Skill name.
            filter_sections: Optional keyword to filter body sections.

        Returns:
            Full markdown body, or filtered sections.

        Raises:
            KeyError: If skill not found.
        """
        self.load()
        skill = self._skills.get(name)
        if skill is None:
            raise KeyError(f"Unknown skill: {name}")

        if filter_sections:
            return _filter_markdown_sections(skill.body, filter_sections)
        return skill.body

    def get_skill(self, name: str) -> Skill:
        """Get the full Skill object."""
        self.load()
        skill = self._skills.get(name)
        if skill is None:
            raise KeyError(f"Unknown skill: {name}")
        return skill

    def search(self, query: str, *, limit: int = 10) -> list[dict]:
        """Simple case-insensitive search across name, description, and tags.

        Returns list of skill summaries, best matches first.
        """
        self.load()
        query_lower = query.lower()
        results = []

        for skill in self._skills.values():
            score = 0
            if query_lower in skill.name.lower():
                score += 100
            if query_lower in skill.description.lower():
                score += 50
            for tag in skill.tags:
                if query_lower in tag.lower():
                    score += 30
            if query_lower in skill.body.lower():
                score += 1  # body match is weak

            if score > 0:
                results.append({
                    "name": skill.name,
                    "description": skill.description,
                    "tags": list(skill.tags),
                    "score": score,
                })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    def system_prompt_index(self, names: Iterable[str] | None = None) -> str:
        """Generate a compact 'available skills' block for the system prompt."""
        self.load()
        skills = self._skills.values()
        if names:
            skills = [s for s in skills if s.name in set(names)]

        lines = ["## Available Skills", ""]
        for skill in sorted(skills, key=lambda s: s.name):
            lines.append(f"- **{skill.name}**: {skill.description}")
        return "\n".join(lines)

    def discovery_block(self, query: str) -> str:
        """Generate a skill discovery hint based on a user query."""
        results = self.search(query, limit=5)
        if not results:
            return ""

        lines = [
            "The following skills may help with this request. "
            "Use read_skill(name) to load full guidance:",
            "",
        ]
        for r in results:
            lines.append(f"- **{r['name']}**: {r['description']}")
        return "\n".join(lines)

    def fingerprint(self) -> str:
        """Return a hash of all skill content for cache invalidation."""
        self.load()
        h = hashlib.sha256()
        for skill in sorted(self._skills.values(), key=lambda s: s.name):
            h.update(skill.name.encode())
            h.update(skill.body.encode())
        return h.hexdigest()

    def list_resources(self, skill_name: str) -> list[str]:
        """List non-SKILL.md files in a skill directory."""
        skill = self.get_skill(skill_name)
        skill_dir = Path(skill.source_path).parent
        resources = []
        for entry in sorted(skill_dir.iterdir()):
            if entry.is_file() and entry.name not in ("SKILL.md", "__pycache__"):
                if not entry.name.startswith("."):
                    resources.append(entry.name)
        return resources

    def read_resource(self, skill_name: str, resource_name: str) -> str:
        """Read a resource file from a skill directory."""
        skill = self.get_skill(skill_name)
        skill_dir = Path(skill.source_path).parent
        resource_path = skill_dir / resource_name
        # Safety: prevent path traversal
        if not resource_path.resolve().is_relative_to(skill_dir.resolve()):
            raise ValueError(f"Invalid resource path: {resource_name}")
        return resource_path.read_text(encoding="utf-8")


# === Frontmatter Parser ===

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split SKILL.md into (frontmatter_dict, body).

    Frontmatter is YAML-like but simpler — only flat key: value pairs.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    frontmatter = {}
    for line in m.group(1).split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        frontmatter[key] = value

    body = text[m.end():].strip()
    return frontmatter, body


def _parse_tags(tags_str: str) -> list[str]:
    """Parse 'tag1, tag2, tag3' or '[tag1, tag2]' into a list."""
    tags_str = tags_str.strip("[]")
    if not tags_str:
        return []
    return [t.strip().strip('"').strip("'") for t in tags_str.split(",") if t.strip()]


def _filter_markdown_sections(body: str, keyword: str) -> str:
    """Return only the sections of body whose headings match keyword."""
    keyword_lower = keyword.lower()
    lines = body.split("\n")
    result = []
    include = False
    for line in lines:
        if line.startswith("#"):
            include = keyword_lower in line.lower()
        if include:
            result.append(line)
    return "\n".join(result) if result else body
```

### 2. 注册技能相关工具

```python
# src/scientex_agent/tools.py 中添加

def register_skill_tools(registry: ToolRegistry, catalog: SkillCatalog) -> None:
    """Register tools for the skill system."""

    @registry.register(
        name="read_skill",
        description="Read the full guidance for a specific skill. "
                    "Use this when you need detailed instructions on how to "
                    "complete a task that a skill covers.",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to read, e.g. 'literature-review'"
                },
                "filter": {
                    "type": "string",
                    "description": "Optional: only show sections matching this keyword"
                },
            },
            "required": ["name"],
        },
    )
    def read_skill(name: str, filter: str = "") -> str:
        try:
            return catalog.read_skill(name, filter_sections=filter or None)
        except KeyError:
            return f"Skill not found: {name}. Available: {[s['name'] for s in catalog.list_skills()]}"

    @registry.register(
        name="search_skills",
        description="Search for available skills by keyword. "
                    "Returns matching skills sorted by relevance.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, e.g. 'protein design' or '文献检索'"
                },
            },
            "required": ["query"],
        },
    )
    def search_skills(query: str) -> list[dict]:
        return catalog.search(query)
```

### 3. 创建示例技能

```markdown
<!-- skills/example-research/SKILL.md -->

---
name: example-research
description: How to conduct a systematic literature search and synthesis.
tags: research, literature, systematic-review
---

# Systematic Literature Search

## Overview

Follow these steps to conduct a thorough scientific literature search.

## Step 1: Define Your Question

Use the PICO framework:
- **P**opulation: What organism/system?
- **I**ntervention: What treatment/exposure?
- **C**omparison: What's the alternative?
- **O**utcome: What are you measuring?

## Step 2: Search Strategy

Start broad, then narrow:
1. Search with primary keywords
2. Filter by date (last 5-10 years)
3. Filter by article type (review, original research)

## Step 3: Screen Results

Read titles → filter → read abstracts → filter → read full text.

## Step 4: Synthesize

Extract key findings, note contradictions, identify gaps.
```

## 验证

```python
from scientex_agent.skill_catalog import SkillCatalog

catalog = SkillCatalog()
catalog.load()

# List skills
print("Available skills:")
for skill in catalog.list_skills():
    print(f"  - {skill['name']}: {skill['description']}")

# Read a skill
body = catalog.read_skill("example-research")
print(f"\nSkill body ({len(body)} chars):")
print(body[:200] + "...")

# Search
results = catalog.search("literature")
print(f"\nSearch 'literature': {len(results)} results")
for r in results:
    print(f"  - {r['name']} (score: {r['score']})")

# System prompt index
print(f"\nSystem prompt block:\n{catalog.system_prompt_index()}")
```

## 当前局限

1. 技能加载是同步的——技能很多时启动会慢（当前 ~30 个技能无影响）
2. 没有技能的热加载（修改 SKILL.md 后需重启）
3. 没有技能版本管理

## 下一步

→ [09-python-kernel.md](09-python-kernel.md)
