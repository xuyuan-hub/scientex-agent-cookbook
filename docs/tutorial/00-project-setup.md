# Step 00: 项目搭建

## 目标

创建一个可运行的 Python 项目骨架，包含依赖管理和基本的目录结构。

## 前置条件

- Python >= 3.11
- `uv` 已安装

## 设计思路

### 为什么用 `src/` 布局

```
scientex_agent/
├── src/
│   └── scientex_agent/     # 所有 Python 代码
│       ├── __init__.py
│       └── __main__.py
├── tests/                 # 测试
├── docs/                  # 文档
│   └── tutorial/          # 本教程
├── pyproject.toml         # 项目配置
├── .env.example           # 环境变量模板
└── .gitignore
```

`src/` 布局的优势：

- 避免意外导入项目根目录的模块
- `pip install -e .` 后 `import scientex_agent` 和直接运行表现一致
- 明确区分"源代码"和"配置文件/文档"

### 为什么用 `uv`

- 比 pip 快 10-100 倍
- 原生支持 `pyproject.toml`
- 锁文件 (`uv.lock`) 保证可复现
- 直接替代 pip/venv/pip-tools

## 实现

### 1. 用 `uv init` 初始化项目

```bash
uv init --package scientex-agent
```

这会自动生成以下内容：

- `pyproject.toml` — 项目元数据和配置（`name = "scientex_agent"`）
- `README.md` — 项目说明
- `src/scientex_agent/__init__.py` — 包入口（`uv` 自动把 `-` 转成 `_`）
- `.python-version` — Python 版本声明

`--package` 标志表示创建一个可分发的 Python 包（含 `src/` 布局），而非简单脚本。

### 2. 添加 CLI 入口

编辑 `pyproject.toml`，添加 `[project.scripts]`、`[build-system]` 并设置 `[tool.uv]`：

→ [pyproject.toml](../../code/00-project-setup/pyproject.toml)

### 3. 创建核心模块

```bash
# __main__.py — 支持 python -m scientex_agent
touch src/scientex_agent/__main__.py
# cli.py — CLI 入口
touch src/scientex_agent/cli.py
```

各文件内容：

- [src/scientex_agent/__init__.py](../../code/00-project-setup/src/scientex_agent/__init__.py)
- [src/scientex_agent/cli.py](../../code/00-project-setup/src/scientex_agent/cli.py)
- [src/scientex_agent/__main__.py](../../code/00-project-setup/src/scientex_agent/__main__.py)
- [.env.example](../../code/00-project-setup/.env.example)

## 验证

```bash
# 测试导入
uv run python -c "import scientex_agent; print(scientex_agent.__version__)"

# 测试 __main__
uv run python -m scientex_agent

# 测试 CLI
uv run scientex_agent
```

预期输出：

```
Scientex v0.1.0 - Hello, world!
```

## 当前局限

- 啥也干不了，只有骨架
- 下一步：接入 LLM，实现第一次对话

## 下一步

→ [01-basic-llm.md](01-basic-llm.md)
