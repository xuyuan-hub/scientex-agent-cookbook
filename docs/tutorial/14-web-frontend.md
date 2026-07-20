# Step 14: Web 前端（SPA）

## 目标

构建一个单页 Web 应用（SPA），提供对话界面、文件浏览器和项目管理。

## 前置条件

- 完成 [13-cli.md](13-cli.md)
- 前端基础（HTML/CSS/JS）

## 设计思路

### 为什么不用前端框架

对于本地桌面 agent 工具：
- **打包简单**：静态文件直接内嵌在 Python 包里
- **零构建**：不需要 webpack/vite/npm，直接写好就能用
- **轻量**：整个前端只有几个文件，加载快
- **易维护**：不需要同时维护 Python 和 Node.js 两套环境

使用**模块化 vanilla JS**：
```
web/
├── index.html              # 入口
├── app.css                 # 样式
├── app.js                  # 主入口
├── core/
│   ├── api.js              # HTTP 请求封装
│   ├── state.js            # 应用状态管理
│   └── router.js           # 客户端路由
└── features/
    ├── composer.js         # 消息输入组件
    ├── conversation.js     # 对话渲染 + 流式
    └── projects.js         # 项目管理
```

### 核心交互流程

```
用户输入消息
  → composer.js 捕获事件
  → api.js POST /frames/{id}/chat/stream (SSE)
  → conversation.js 逐 token 渲染
  → 工具调用卡片折叠/展开
  → 自动滚动到底部
```

## 实现

### 1. index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scientex</title>
    <link rel="stylesheet" href="app.css">
</head>
<body>
    <div id="app">
        <!-- Sidebar -->
        <aside id="sidebar">
            <div id="project-list"></div>
            <div id="frame-list"></div>
        </aside>

        <!-- Main area -->
        <main id="main">
            <div id="conversation"></div>
            <div id="composer">
                <textarea id="message-input" rows="1"
                    placeholder="Type a message... (Enter to send, Shift+Enter for new line)"></textarea>
                <button id="send-btn">Send</button>
            </div>
        </main>
    </div>

    <script src="core/api.js"></script>
    <script src="core/state.js"></script>
    <script src="features/conversation.js"></script>
    <script src="features/composer.js"></script>
    <script src="features/projects.js"></script>
    <script src="app.js"></script>
</body>
</html>
```

### 2. core/api.js

```javascript
// core/api.js — HTTP client for Scientex API
const API_BASE = '';

const api = {
    async get(path) {
        const resp = await fetch(API_BASE + path);
        if (!resp.ok) throw new Error(await resp.text());
        return resp.json();
    },

    async post(path, body) {
        const resp = await fetch(API_BASE + path, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!resp.ok) throw new Error(await resp.text());
        return resp.json();
    },

    // SSE streaming
    async streamChat(path, body, onToken, onToolStart, onToolEnd, onDone) {
        const resp = await fetch(API_BASE + path, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const event = JSON.parse(line.slice(6));
                    if (event.token && onToken) onToken(event.token);
                    else if (event.tool_start && onToolStart) onToolStart(event.tool_start);
                    else if (event.tool_end && onToolEnd) onToolEnd(event.tool_end);
                    else if (event.done && onDone) onDone();
                } catch (e) {
                    // Skip malformed events
                }
            }
        }
    },
};
```

### 3. core/state.js

```javascript
// core/state.js — Simple reactive state management
const state = {
    currentProject: null,
    currentFrame: null,
    projects: [],
    frames: [],
    messages: [],

    listeners: {},

    on(event, callback) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(callback);
    },

    emit(event, data) {
        (this.listeners[event] || []).forEach(cb => cb(data));
    },

    async loadProjects() {
        this.projects = await api.get('/projects');
        this.emit('projects-changed', this.projects);
    },

    async loadFrames(projectId) {
        this.frames = await api.get(`/projects/${projectId}/frames`);
        this.emit('frames-changed', this.frames);
    },

    async loadMessages(frameId) {
        this.messages = await api.get(`/frames/${frameId}/messages`);
        this.emit('messages-changed', this.messages);
    },
};
```

### 4. features/conversation.js

```javascript
// features/conversation.js — Conversation rendering with SSE streaming

function renderMessage(msg) {
    const div = document.createElement('div');
    div.className = `message message-${msg.role}`;

    if (msg.role === 'user') {
        div.innerHTML = `<div class="bubble user-bubble">${escapeHtml(msg.content)}</div>`;
    } else if (msg.role === 'assistant') {
        div.innerHTML = `<div class="bubble assistant-bubble">${escapeHtml(msg.content)}</div>`;
    }

    return div;
}

function renderConversation(messages) {
    const container = document.getElementById('conversation');
    container.innerHTML = '';
    messages.forEach(msg => {
        container.appendChild(renderMessage(msg));
    });
    container.scrollTop = container.scrollHeight;
}

// Streaming message rendering
function createStreamingMessage() {
    const container = document.getElementById('conversation');
    const div = document.createElement('div');
    div.className = 'message message-assistant streaming';
    const bubble = document.createElement('div');
    bubble.className = 'bubble assistant-bubble';
    div.appendChild(bubble);
    container.appendChild(div);
    return {
        append(text) {
            bubble.textContent += text;
            container.scrollTop = container.scrollHeight;
        },
        finalize() {
            div.classList.remove('streaming');
        },
    };
}

// Tool call card
function createToolCard(toolStart) {
    const container = document.getElementById('conversation');
    const card = document.createElement('div');
    card.className = 'tool-card';
    card.innerHTML = `
        <div class="tool-card-header">
            <span class="tool-icon">🔧</span>
            <span class="tool-name">${escapeHtml(toolStart.name)}</span>
            <span class="tool-status">Running...</span>
        </div>
        <div class="tool-card-body" style="display:none">
            <pre>${JSON.stringify(toolStart.input, null, 2)}</pre>
        </div>
    `;
    card.addEventListener('click', () => {
        const body = card.querySelector('.tool-card-body');
        body.style.display = body.style.display === 'none' ? 'block' : 'none';
    });
    container.appendChild(card);
    return card;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
```

### 5. features/composer.js

```javascript
// features/composer.js — Message input and send

function setupComposer() {
    const input = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');

    async function send() {
        const content = input.value.trim();
        if (!content) return;
        if (!state.currentFrame) {
            alert('Please select or create a project first.');
            return;
        }

        input.value = '';
        input.disabled = true;
        sendBtn.disabled = true;

        // Render user message
        const container = document.getElementById('conversation');
        container.appendChild(renderMessage({ role: 'user', content }));

        // Start streaming
        const streamMsg = createStreamingMessage();

        await api.streamChat(
            `/frames/${state.currentFrame.id}/chat/stream`,
            { content },
            // onToken
            (token) => streamMsg.append(token),
            // onToolStart
            (tool) => createToolCard(tool),
            // onToolEnd
            null,
            // onDone
            () => {
                streamMsg.finalize();
                input.disabled = false;
                sendBtn.disabled = false;
                input.focus();
            },
        );
    }

    sendBtn.addEventListener('click', send);
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            send();
        }
    });
}
```

### 6. app.css（核心样式）

```css
:root {
    --bg-primary: #1e1e2e;
    --bg-secondary: #181825;
    --bg-surface: #313244;
    --text-primary: #cdd6f4;
    --text-secondary: #a6adc8;
    --accent: #89b4fa;
    --border: #45475a;
    --user-bubble: #45475a;
    --assistant-bubble: #313244;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    height: 100vh;
    overflow: hidden;
}

#app {
    display: flex;
    height: 100vh;
}

#sidebar {
    width: 260px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
    padding: 16px;
    overflow-y: auto;
}

#main {
    flex: 1;
    display: flex;
    flex-direction: column;
}

#conversation {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
}

.message {
    margin-bottom: 16px;
    max-width: 80%;
}

.message-user {
    margin-left: auto;
}

.message-assistant {
    margin-right: auto;
}

.bubble {
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.5;
    white-space: pre-wrap;
}

.user-bubble {
    background: var(--user-bubble);
}

.assistant-bubble {
    background: var(--assistant-bubble);
}

.streaming .assistant-bubble::after {
    content: '▊';
    animation: blink 1s step-end infinite;
}

@keyframes blink {
    50% { opacity: 0; }
}

#composer {
    padding: 16px 24px;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 12px;
}

#message-input {
    flex: 1;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    color: var(--text-primary);
    resize: none;
    font-family: inherit;
    font-size: 14px;
}

#message-input:focus {
    outline: none;
    border-color: var(--accent);
}

#send-btn {
    background: var(--accent);
    color: var(--bg-primary);
    border: none;
    border-radius: 8px;
    padding: 0 20px;
    cursor: pointer;
    font-weight: 600;
}

#send-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.tool-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 12px;
    cursor: pointer;
    font-size: 13px;
}

.tool-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
}

.tool-status {
    color: var(--text-secondary);
    font-size: 12px;
}
```

### 7. app.js

```javascript
// app.js — Application entry point
document.addEventListener('DOMContentLoaded', async () => {
    setupComposer();

    // Load initial data
    await state.loadProjects();
    if (state.projects.length > 0) {
        state.currentProject = state.projects[0];
        await state.loadFrames(state.currentProject.id);
        if (state.frames.length > 0) {
            state.currentFrame = state.frames[0];
            await state.loadMessages(state.currentFrame.id);
            renderConversation(state.messages);
        }
    }

    // Listen for state changes
    state.on('projects-changed', (projects) => {
        // Re-render project list
    });

    state.on('messages-changed', (messages) => {
        renderConversation(messages);
    });
});
```

## 验证

```bash
# 启动服务
uv run scientex_agent run-server

# 浏览器访问
open http://127.0.0.1:8765
```

预期效果：
- 左侧边栏显示项目列表
- 中间对话区域显示历史消息
- 底部输入框可以发送消息
- 消息流式出现，带闪烁光标
- 工具调用显示为可折叠卡片

## 深入理解

### 前端构建策略

对于本地桌面应用，有两种分发方式：

1. **嵌入式**（当前方案）：静态文件打包在 Python 包里
   ```python
   api.mount("/", StaticFiles(directory=str(web_dir), html=True))
   ```
   优点：一个 `uv run` 全搞定。缺点：前端修改需要了解 Python 包结构。

2. **独立部署**：前端从 CDN 加载或独立开发服务器
   优点：前后端完全分离。缺点：增加部署复杂度。

对于本地工具，嵌入式是最佳选择。

### 为什么流式渲染需要特殊处理

```
同步渲染：  等所有消息下载完 → 一次性渲染
流式渲染：  收到一个 token → 立即追加到 DOM
```

DOM 操作是昂贵的。对于高速流式（100+ tokens/秒），应该：
- 使用 `requestAnimationFrame` 批量更新
- 或者使用 `DocumentFragment` 缓冲

## 当前局限

1. 没有 Markdown 渲染（代码块、表格、公式）
2. 没有文件上传/预览
3. 移动端适配不完整

## 下一步

→ [15-domain-features.md](15-domain-features.md)
