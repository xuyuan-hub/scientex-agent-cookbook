# 附：用 VS Code 调试你的 Python 代码

> 这篇是给 Python 初学者 / VS Code 调试新手的快速入门。
> 跟着 [Step 05](05-multi-provider.md) 走到一半，当你发现"print 不够用了"的时候，回来看这篇。

## 为什么不用 print？

调试代码时，大多数人的第一反应是加 `print()`：

```python
def chat(self, request):
    print(f"DEBUG request: {request}")        # ← 加这一行
    messages = _to_openai_messages(request)
    print(f"DEBUG messages: {messages}")      # ← 再加这一行
    ...
```

能用，但有几个痛点：

| print 的问题                           | 调试器的解决方式         |
| -------------------------------------- | ------------------------ |
| 每次改完要重跑整个程序                 | 程序停在断点，想改就改   |
| 想看的变量没打印 → 再 print → 再重跑 | 鼠标悬停就能看到任意变量 |
| 调试完要删掉 print，不然提交上去很丢人 | 断点不污染代码           |
| 深层调用栈里发生了什么？看不出来       | 调用栈面板一键跳转       |
| 想"如果 x>10 才停下来"？print 做不到   | 条件断点，一句话搞定     |

**结论**：`print` 适合 10 行的小脚本；超过 50 行的项目，调试器能让你快 10 倍。

---

## 5 分钟上手：调试器核心操作

VS Code 调试只需要记住 **5 个快捷键**：

| 快捷键              | 作用                                         | 类比                  |
| ------------------- | -------------------------------------------- | --------------------- |
| **F9**        | 在光标所在行**切换断点**               | 在地图上插红旗        |
| **F5**        | **启动 / 继续**调试                    | 出发 / 开到下一个红旗 |
| **F10**       | **单步跳过**（执行当前行，不进入函数） | 跨过一栋房子          |
| **F11**       | **单步进入**（进入当前行调用的函数）   | 走进那栋房子          |
| **Shift+F11** | **单步跳出**（从当前函数返回）         | 从房子里出来          |

**调试时界面会变成这样**：

```
┌─────────────────────────────────────────────────┐
│  代码编辑器                                       │
│  断点行高亮（黄箭头 ← 当前停在这里）              │
│                                                 │
├─────────────────────┬───────────────────────────┤
│  VARIABLES          │  CALL STACK               │
│  ├─ Locals          │  ├─ chat()  ← 当前       │
│  │  ├─ request =... │  ├─ _interactive_chat()  │
│  │  ├─ messages =...│  └─ main()               │
│  │  └─ response =.. │                          │
│  ├─ Watch           ├───────────────────────────┤
│  └─ ...             │  DEBUG CONSOLE            │
│                     │  可以即时执行 Python 表达式 │
└─────────────────────┴───────────────────────────┘
```

---

## 配置 launch.json

VS Code 用 `.vscode/launch.json` 文件描述"怎么启动程序"。

### 方法 A：让 VS Code 自动生成

1. 打开任意 `.py` 文件
2. 按 **F5**
3. VS Code 会问："Select a debug configuration" → 选 **Python Debugger: Current File with Arguments** 之类的
4. 会自动生成一个基础的 `launch.json`

但自动生成的不够好用。推荐**手动改成下面这样**。

### 方法 B：直接抄（推荐）

创建（或编辑）项目根目录下的 `.vscode/launch.json`：

```json
{
  "version": "0.2.0",
  "configurations": [

    // ─── 1. 调试当前打开的文件 ───
    {
      "name": "🐍 Python: 当前文件",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env.local",
      "justMyCode": false
    },

    // ─── 2. 调试 CLI 的交互式聊天 ───
    {
      "name": "💬 CLI: 交互式",
      "type": "debugpy",
      "request": "launch",
      "module": "scientex_agent",
      "args": ["chat", "--interactive"],
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env.local",
      "justMyCode": false
    },

    // ─── 3. 调试 CLI + 指定 provider（Step 05 用这个！）───
    {
      "name": "💬 CLI: 指定 provider",
      "type": "debugpy",
      "request": "launch",
      "module": "scientex_agent",
      "args": [
        "chat",
        "--provider", "${input:providerName}",
        "--interactive"
      ],
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env.local",
      "justMyCode": false
    },

    // ─── 4. 运行单测 ───
    {
      "name": "🧪 当前测试文件",
      "type": "debugpy",
      "request": "launch",
      "module": "unittest",
      "args": ["${file}"],
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env.local"
    }
  ],

  // 让 "指定 provider" 这个配置每次启动时弹出输入框
  "inputs": [
    {
      "id": "providerName",
      "type": "pickString",
      "description": "选择要调试的 provider",
      "options": ["deepseek", "anthropic", "openai"],
      "default": "deepseek"
    }
  ]
}
```

### 关键字段解释

| 字段           | 作用                                                       | 例子                                                    |
| -------------- | ---------------------------------------------------------- | ------------------------------------------------------- |
| `name`       | 在调试配置下拉菜单里显示的名字                             | `"💬 CLI: 指定 provider"`                             |
| `type`       | 调试器类型，Python 固定`debugpy`                         | —                                                      |
| `request`    | `"launch"`（启动新进程） vs `"attach"`（连接已运行的） | 我们用`launch`                                        |
| `program`    | 要执行的 Python 文件路径                                   | `"${file}"` = 当前打开的文件                          |
| `module`     | 用`python -m xxx` 方式启动（比 `program` 更常用）      | `"scientex_agent"`                                    |
| `args`       | 命令行参数数组                                             | `["chat", "--interactive"]`                           |
| `envFile`    | 从这个文件加载环境变量                                     | `.env.local`（你的 API key 在这里）                   |
| `console`    | 用哪种终端                                                 | `"integratedTerminal"` = VS Code 内嵌终端，能接收输入 |
| `justMyCode` | `false` = 可以走进第三方库（如 `openai` SDK）的代码    | 调试 adapter 时特别有用                                 |

### 怎么选配置？

按 **F5** 之前，看 VS Code 左侧的 **"Run and Debug" 面板**（虫子图标 🐞），顶部有个下拉菜单，选你要用的配置：

```
┌──────────────────────────────┐
│ 🔍 RUN AND DEBUG             │
│ ┌──────────────────────────┐ │
│ │ 💬 CLI: 指定 provider  ▼ │ │  ← 点这个切换
│ └──────────────────────────┘ │
│                              │
│ ▶ Start Debugging  (F5)      │
│                              │
│ BREAKPOINTS                  │
│ ☐ Uncaught Exceptions        │
│ ☐ All Exceptions             │
└──────────────────────────────┘
```

---

## 🎯 实战场景：Step 05 调试

### 场景 A：看 adapter 把 ChatRequest 翻译成什么格式

你写好了 `OpenAICompatibleProvider.chat()`，想看 `_to_openai_messages()` 到底翻译成了什么——

1. 打开 `src/scientex_agent/providers/openai_compatible.py`
2. 在 `messages = _to_openai_messages(request)` 这行按 **F9** 打断点
3. 在 `response = self._client.chat.completions.create(...)` 这行也按 **F9**
4. 选配置 **"💬 CLI: 指定 provider"**，按 **F5**
5. 选 `deepseek`，在 CLI 里输入 "ping"
6. 程序停在第一个断点 → 把鼠标悬停在 `messages` 上，看到完整的 OpenAI 格式
7. 按 **F10** 走一步 → 停在 SDK 调用前，再悬停看 `request` 本身
8. 按 **F5** 继续，程序跑到第二个断点，看 SDK 的 response 长什么样

**关键动作**：鼠标悬停看变量 > 左侧 VARIABLES 面板看所有局部变量 > DEBUG CONSOLE 里即时求值。

### 场景 B：调试 Anthropic adapter 的消息转换

Anthropic 的消息格式很怪（tool result 要合并进 user message 数组），写的时候最容易出 bug。

1. 在 `anthropic.py` 的 `_convert_messages()` 第一行按 **F9**
2. 启动调试
3. 用 **F10** 一行行走，每走一行都在 VARIABLES 里看 `result` 列表是怎么变化的
4. 特别注意 `elif msg.role == "tool":` 这个分支 —— 看 `result[-1]` 是不是 user 角色，是不是被正确合并了

### 场景 C：调试 CLI 参数解析

你加了 `--provider` 参数，但不确定 `args.provider` 拿到的是什么——

1. 打开 `src/scientex_agent/cli.py`
2. 在 `if args.provider:` 这行按 **F9**
3. 选配置 **"💬 CLI: 指定 provider"**，启动
4. 停在断点时，在 DEBUG CONSOLE 输入 `args` → 看到完整的 argparse Namespace
5. 输入 `args.provider` → 看到你选的那个 provider 名字

---

## 进阶技巧

### 1. 条件断点

**场景**：你在一个循环里，但只想停在"第 3 次循环"或者"当 model=claude-sonnet-5 时"。

**做法**：右键断点 → **Add Condition** → 输入 Python 表达式：

```python
# 第 3 次循环才停
i == 3

# 只在某个模型时停
request.model == "claude-sonnet-5"

# 只在出错时停
response is None
```

条件返回 `True` 时才会断下来。

### 2. Logpoint（不打断执行的"print"）

**场景**：你想看某行被执行时的变量值，但不想打断执行流程。

**做法**：右键断点位置 → **Add Logpoint...** → 输入一段文本，用 `{表达式}` 插值：

```
调用了 _convert_messages，输入了 {len(messages)} 条消息
```

程序会在这行打印这串文字（到 DEBUG CONSOLE），但不会停下来。**比 print 好在**：不用改代码、不用重启、调试完自动消失。

### 3. Watch 表达式

**场景**：你一直在看某个复杂表达式的值（比如 `registry._providers["deepseek"].list_models()`），每次去 DEBUG CONSOLE 敲太累。

**做法**：在左侧 VARIABLES 面板下方找到 **WATCH**，点 `+`，输入表达式。每次单步都会自动重新求值并显示。

### 4. Debug Console 即时求值

程序停在断点时，在 DEBUG CONSOLE 里可以直接执行 Python 代码：

```python
# 看当前函数的参数
> request
ChatRequest(messages=[...], model='deepseek-v4-pro', ...)

# 调用一个函数看结果
> _to_openai_messages(request)
[{'role': 'user', 'content': 'ping'}]

# 修改一个变量（让程序"假装"走了另一条路）
> request = ChatRequest(messages=[], model='other', max_tokens=10)
```

**这个能力极其强大**——你可以不重启程序就试各种情况。

### 5. 进入第三方库代码

默认情况下 `justMyCode: true`，调试器不会进入 `openai/` 或 `anthropic/` 这些第三方库的代码。

想看 SDK 内部在做什么？把 launch.json 里的 `justMyCode` 改成 `false`：

```json
"justMyCode": false
```

然后按 **F11** 进入 `self._client.chat.completions.create(...)` 调用，就能走到 OpenAI SDK 的代码里，看它怎么构造 HTTP 请求、怎么处理 response。

---

## 常见坑

### 坑 1：调试时输入没反应

**现象**：按 F5 启动调试，CLI 出来了，但输入 `> xxx` 没反应。

**原因**：`console` 配置成了 `"internalConsole"`（VS Code 内部的只读调试控制台），不能接收用户输入。

**解决**：改成 `"integratedTerminal"`（VS Code 内嵌的真正终端）。

### 坑 2：API key 没加载

**现象**：跑起来直接报错 `KeyError: 'DEEPSEEK_API_KEY'`。

**原因**：忘了配 `envFile`，或者路径写错了。

**解决**：检查 launch.json 里有 `"envFile": "${workspaceFolder}/.env.local"`，且 `.env.local` 文件存在。

### 坑 3：断点变成空心圆 ⚪，不停下来

**现象**：打了断点，但程序跑过去了没停。

**原因**：

- 模块没被真正导入（比如你改的代码不在当前运行的版本里）
- 断点在注释或空行上
- 程序走的是 `.venv/lib/...` 里的旧代码，不是你当前编辑的

**解决**：

1. 检查是不是用 `module: "scientex_agent"` 启动（这样会走当前工作目录的代码）
2. 重新 `uv sync` 一下，确保开发模式安装
3. 重启调试

### 坑 4：调试器卡在 "Connecting..."

**现象**：按 F5 后左下角一直转圈。

**原因**：通常是 Python 解释器没选对。

**解决**：按 `Cmd+Shift+P` → `Python: Select Interpreter` → 选 `.venv/bin/python`。

---

## 小结

调试器 = 时间机器。它能让你：

- **停下**：在任何一行暂停程序
- **观察**：看任何变量的当前值
- **步进**：一行行走，看程序是怎么一步步变化的
- **修改**：临时改变量，试不同的路径
- **回溯**：看调用栈，知道"我是怎么走到这里的"

**下次你想加 `print()` 时**，先试试 F9 + F5。五分钟学会，省下的是几个小时。

---

## 相关资料

- [VS Code Python Debugging 官方文档](https://code.visualstudio.com/docs/python/debugging)
- [launch.json 配置参考](https://code.visualstudio.com/docs/editor/debugging#_launchjson-attributes)
- [debugpy 文档](https://github.com/microsoft/debugpy)
