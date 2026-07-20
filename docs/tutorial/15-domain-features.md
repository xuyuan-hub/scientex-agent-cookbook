# Step 15: 领域功能

## 目标

集成科学领域特有功能：生物信息学工具、化学分子编辑器、记忆系统、事实验证。

## 前置条件

- 完成 [14-web-frontend.md](14-web-frontend.md)
- 理解 MCP 集成 [07-mcp-integration.md](07-mcp-integration.md)
- 理解技能系统 [08-skill-system.md](08-skill-system.md)

## 设计思路

### 领域功能全景

```
Scientex 领域功能
├── 🔬 生物信息学工具
│   ├── 文献搜索 (PubMed, EuropePMC, arXiv, bioRxiv)
│   ├── 序列分析 (UniProt, InterPro)
│   ├── 结构生物学 (PDB, AlphaFold)
│   ├── 遗传变异 (ClinVar, gnomAD)
│   ├── 药物/靶点 (ChEMBL)
│   └── 临床试验 (ClinicalTrials.gov)
│
├── ⚗️ 化学工具
│   └── 分子编辑器 (Ketcher sketcher)
│
├── 🧠 记忆系统
│   └── 跨对话持久记忆
│
├── ✅ 事实验证
│   └── 声明验证和跟踪
│
└── 👤 Agent 配置
    └── 可切换的 agent 角色和系统提示
```

### 架构模式

所有领域功能通过同一模式接入：

```
1. 数据源/服务（本地或远程）
   ↓
2. MCP Server 或直接 HTTP 调用
   ↓
3. Tool Registry（统一注册）
   ↓
4. Agent 通过工具调用使用
```

**关键原则**：领域功能是**内容**而不是**架构**。核心架构不需要因为增加领域功能而改变。

## 实现

### 1. 生物信息学工具集成

```python
# src/scientex_agent/bio_tools.py

"""Bioinformatics tool integration via MCP servers.

Each domain has its own MCP server. Tools are discovered and registered
dynamically when the server starts.
"""

DOMAIN_SERVERS = {
    "literature": {
        "command": "python",
        "args": ["-m", "scientex_agent.bundled_bio_tools.lib.mcp_literature.run_server"],
        "description": "PubMed, EuropePMC, arXiv, bioRxiv — search and fetch papers",
    },
    "protein_annotation": {
        "command": "python",
        "args": ["-m", "scientex_agent.bundled_bio_tools.lib.mcp_protein_annotation.run_server"],
        "description": "UniProt, InterPro, QuickGO — protein sequence and function",
    },
    "human_genetics": {
        "command": "python",
        "args": ["-m", "scientex_agent.bundled_bio_tools.lib.mcp_human_genetics.run_server"],
        "description": "ClinVar, gnomAD, CADD — human genetic variation",
    },
    "structure_interactions": {
        "command": "python",
        "args": ["-m", "scientex_agent.bundled_bio_tools.lib.mcp_structure_interactions.run_server"],
        "description": "PDB, AlphaFold, STRING — protein structures and interactions",
    },
    "chembl": {
        "command": "python",
        "args": ["-m", "scientex_agent.bundled_bio_tools.lib.mcp_chembl.run_server"],
        "description": "ChEMBL — bioactive molecules, targets, drugs",
    },
    # ... 更多领域服务器
}


class BioToolManager:
    """Manages the lifecycle of bio MCP servers.

    Servers are started lazily — only when a tool from that domain
    is first requested.
    """

    def __init__(self) -> None:
        self._servers: dict[str, SyncMCPServer] = {}
        self._tools: dict[str, str] = {}  # tool_name → domain

    def start_domain(self, domain: str) -> list[dict]:
        """Start an MCP server for a domain and return its tools."""
        if domain not in DOMAIN_SERVERS:
            raise ValueError(f"Unknown domain: {domain}")

        cfg = DOMAIN_SERVERS[domain]
        server = SyncMCPServer(
            command=cfg["command"],
            args=cfg["args"],
            server_name=domain,
        )
        tools = server.start()
        self._servers[domain] = server
        for tool in tools:
            full_name = f"bio_{domain}_{tool['name']}"
            self._tools[full_name] = domain
        return tools

    def call_tool(self, domain: str, tool_name: str, args: dict) -> str:
        """Call a tool on a domain's MCP server."""
        if domain not in self._servers:
            self.start_domain(domain)
        return self._servers[domain].call_tool(tool_name, args)

    def search_tools(self, query: str) -> list[dict]:
        """Search all bio tools by keyword."""
        results = []
        query_lower = query.lower()
        for domain, cfg in DOMAIN_SERVERS.items():
            if query_lower in domain.lower() or query_lower in cfg["description"].lower():
                results.append({"domain": domain, "description": cfg["description"]})
        return results

    def close_all(self) -> None:
        for server in self._servers.values():
            server.close()
```

### 2. 记忆系统

```python
# src/scientex_agent/memory.py

"""Durable memory system for projects.

Memories persist across conversations within a project.
LLM can read and write memories via tools.
"""

from .metadata import MetadataStore


class MemoryManager:
    """Manages durable memories for a project.

    Usage::

        mgr = MemoryManager(metadata_store)
        mgr.remember(project_id, "user", "User prefers Chinese replies")
        memories = mgr.recall(project_id, category="user")
    """

    def __init__(self, metadata: MetadataStore) -> None:
        self.metadata = metadata

    def remember(self, project_id: str, category: str, content: str):
        """Store a new memory. Deduplicates by content similarity."""
        # Simple dedup: check exact match
        existing = self.metadata.list_memories(project_id)
        for mem in existing:
            if mem.content.strip() == content.strip():
                return mem  # Already stored

        return self.metadata.create_memory(
            project_id=project_id,
            category=category,
            content=content,
        )

    def recall(
        self,
        project_id: str,
        *,
        category: str | None = None,
        query: str | None = None,
        limit: int = 20,
    ) -> list:
        """Retrieve memories, optionally filtered."""
        memories = self.metadata.list_memories(project_id)
        if category:
            memories = [m for m in memories if m.category == category]
        if query:
            query_lower = query.lower()
            memories = [m for m in memories if query_lower in m.content.lower()]
        return memories[-limit:]  # Most recent

    def forget(self, memory_id: str) -> None:
        self.metadata.delete_memory(memory_id)

    def format_for_prompt(self, project_id: str, limit: int = 40) -> str:
        """Format memories for injection into the system prompt."""
        memories = self.recall(project_id, limit=limit)
        if not memories:
            return ""

        lines = ["## Durable Memories", ""]
        for m in memories:
            lines.append(f"- [{m.category}] {m.content}")
        return "\n".join(lines)
```

### 3. 事实验证系统

```python
# src/scientex_agent/verification.py

"""Fact-check and claim verification system."""


class VerificationManager:
    """Tracks verification of factual claims.

    Usage::

        vm = VerificationManager(metadata)
        check = vm.record_claim(
            frame_id="...",
            claim="BRCA1 is located on chromosome 17",
        )
        vm.update_verdict(check.id, "confirmed", evidence="...", confidence=0.95)
    """

    def __init__(self, metadata) -> None:
        self.metadata = metadata

    def record_claim(self, frame_id: str, claim: str):
        """Record a claim for later verification."""
        return self.metadata.create_verification_check(
            frame_id=frame_id,
            claim=claim,
            verdict="pending",
            evidence="",
            confidence=0.0,
        )

    def update_verdict(
        self,
        check_id: str,
        verdict: str,  # "confirmed" | "refuted" | "uncertain"
        evidence: str = "",
        confidence: float = 0.0,
    ):
        self.metadata.update_verification_check(
            id=check_id,
            verdict=verdict,
            evidence=evidence,
            confidence=confidence,
        )

    def list_claims(self, frame_id: str) -> list:
        return self.metadata.list_verification_checks(frame_id)

    def pending_claims(self, frame_id: str) -> list:
        return [
            c for c in self.list_claims(frame_id)
            if c.verdict == "pending"
        ]
```

### 4. Agent Profile 系统

```python
# src/scientex_agent/profiles.py

"""Agent profile management — switchable personas with custom system prompts."""

import json
from pathlib import Path


DEFAULT_PROFILES = [
    {
        "name": "default",
        "displayName": "General Scientific Assistant",
        "systemPrompt": (
            "You are a general-purpose life-science research assistant. "
            "Help with literature search, data analysis, and experimental design."
        ),
        "allowedSkills": [],  # empty = all skills allowed
    },
    {
        "name": "bioinformatics",
        "displayName": "Bioinformatics Specialist",
        "systemPrompt": (
            "You are a bioinformatics specialist. Focus on sequence analysis, "
            "structural biology, and genomics. Use bio* tools extensively."
        ),
        "allowedSkills": ["alphafold2", "esmfold2", "variant-interpretation"],
    },
    {
        "name": "literature-reviewer",
        "displayName": "Literature Review Specialist",
        "systemPrompt": (
            "You are a scientific literature review specialist. "
            "Systematically search, evaluate, and synthesize research papers. "
            "Always check for publication date, journal reputation, and methodological quality."
        ),
        "allowedSkills": ["literature-review", "evidence-quality"],
    },
]


class ProfileManager:
    """Manage agent profiles."""

    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir / "profiles"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        for profile in DEFAULT_PROFILES:
            path = self._dir / f"{profile['name']}.json"
            if not path.exists():
                path.write_text(json.dumps(profile, indent=2))

    def list_profiles(self) -> list[dict]:
        profiles = []
        for path in sorted(self._dir.glob("*.json")):
            profiles.append(json.loads(path.read_text()))
        return profiles

    def get_profile(self, name: str) -> dict | None:
        path = self._dir / f"{name}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def save_profile(self, name: str, data: dict) -> None:
        path = self._dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2))
```

## 验证

```python
# 测试生物工具
from scientex_agent.bio_tools import BioToolManager

mgr = BioToolManager()
tools = mgr.search_tools("genetics")
print(f"Genetics tools: {len(tools)}")
for t in tools:
    print(f"  {t['domain']}: {t['description']}")

# 测试记忆系统
from scientex_agent.memory import MemoryManager
from scientex_agent.metadata import MetadataStore

meta = MetadataStore(Path("/tmp/scientex_agent-test/app.db"))
meta.initialize()
project = meta.create_project(name="Test")
mm = MemoryManager(meta)
mm.remember(project.id, "user", "User is a biologist studying cancer")
memories = mm.recall(project.id, query="cancer")
print(f"Cancer-related memories: {len(memories)}")

# 测试 profile
from scientex_agent.profiles import ProfileManager
pm = ProfileManager(Path("/tmp/scientex_agent-test"))
profiles = pm.list_profiles()
print(f"Available profiles: {[p['displayName'] for p in profiles]}")
```

## 当前局限性

1. 生物工具需要逐个启动 MCP 服务器，启动延迟较高
2. 记忆系统没有语义搜索（只做关键词匹配）
3. Profile 系统没有 UI 切换界面

## 下一步

→ [16-production-readiness.md](16-production-readiness.md)
