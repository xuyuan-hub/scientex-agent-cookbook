# Step 14: Web 前端（Vue 3 + TypeScript + Vite）

## 目标

构建 Scientex 的单页 Web 应用：项目与 Frame 导航、流式对话、工具/审批状态和 Artifact 入口。前端使用 **Vue 3、TypeScript 与 Vite**，不再使用 vanilla JavaScript 维护手工 DOM、事件总线和可变全局状态。

## 前置条件

- 完成 [13-cli.md](13-cli.md)
- 完成 [11-http-api.md](11-http-api.md) 的 `/api/v1` 与 RunEvent SSE 契约
- 已安装 Node.js LTS 与 pnpm（或 npm）

## 设计思路

### 为什么选择 Vue，而不是原生 JS

前端到这一步已经不只是“显示一段文本”：它需要维护项目选择、Frame 切换、流式消息、取消、错误、工具卡片和审批状态。Vue 的组件模型与响应式状态使这些状态有明确归属，TypeScript 让 API 变更在构建期暴露，Vite 提供快速开发与可复现产物。

```
浏览器
  └── Vue component tree
        ├── Pinia stores：跨视图状态与异步操作
        ├── API client：JSON 请求、Problem Details、SSE parser
        └── router：URL → project / frame 视图
                    │
                    ▼
              /api/v1（Step 11）
```

这不是把所有逻辑移到前端。科研数据、模型调用、工具权限、审批判断和 Markdown 安全策略仍在后端；Vue 只渲染已授权的状态和发起受控请求。

### 组件与文件边界

```
web/
├── package.json
├── vite.config.ts
├── src/
│   ├── main.ts
│   ├── App.vue
│   ├── api/
│   │   ├── client.ts          # fetch、Problem Details、DTO
│   │   └── sse.ts             # POST + ReadableStream SSE parser
│   ├── stores/
│   │   ├── projects.ts
│   │   └── conversation.ts
│   ├── router/
│   │   └── index.ts
│   ├── views/
│   │   ├── ProjectView.vue
│   │   └── FrameView.vue
│   └── components/
│       ├── ConversationList.vue
│       ├── MessageBubble.vue
│       ├── ComposerForm.vue
│       ├── ToolRunCard.vue
│       └── ApprovalCard.vue
└── tests/
```

视图负责路由参数与布局，组件只接收 props / 发出事件，Pinia store 负责调用 API 和生命周期。不要让多个组件同时向同一 SSE stream 追加 DOM。

## 实现

### 1. 创建 Vue 工程

使用官方 `create-vue` 创建 Vite 工程，选择 TypeScript、Vue Router、Pinia、Vitest、ESLint 和 Prettier：

```bash
pnpm create vue@latest web
cd web
pnpm install
pnpm dev
```

`create-vue` 生成的 lockfile 必须提交。不要使用维护模式的 Vue CLI，也不要通过 `<script>` 标签直接在 Python 静态目录中加载未打包模块。

关键依赖如下（实际版本由创建器与 lockfile 锁定）：

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vue-tsc -b && vite build",
    "test:unit": "vitest run",
    "lint": "eslint .",
    "format:check": "prettier --check ."
  },
  "dependencies": {
    "pinia": "^3.0.0",
    "vue": "^3.0.0",
    "vue-router": "^4.0.0"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.0.0",
    "typescript": "^5.0.0",
    "vite": "^6.0.0",
    "vue-tsc": "^2.0.0"
  }
}
```

版本范围仅表达最低代际；发布构建实际使用被提交的 lockfile。升级 Vue/Vite 时，在单独 PR 中运行 typecheck、unit test 和浏览器测试。

### 2. 通过 Vite 代理访问后端

开发时让浏览器始终请求相对 `/api`，由 Vite 代理到 FastAPI；这样避免 CORS、cookie 和生产路径各写一套。

```ts
// web/vite.config.ts
import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: true,
      },
    },
  },
})
```

生产构建中由同源 FastAPI/反向代理提供 `web/dist`，因此 API client 的 base URL 仍为空。静态文件 mount 必须在所有 `/api` router **之后**注册：

```python
# api_server.py（生产模式，所有 API route 注册后）
api.mount("/", StaticFiles(directory=settings.web_dist, html=True), name="web")
```

### 3. 写类型化 HTTP client

```ts
// web/src/api/client.ts
export type ProblemDetail = {
  type: string
  title: string
  status: number
  detail: string
  instance: string
}

export type Message = {
  id: string
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string
  created_at: number
}

export type RunEvent = {
  run_id: string
  seq: number
  type:
    | 'run.started'
    | 'message.delta'
    | 'tool.started'
    | 'tool.completed'
    | 'approval.required'
    | 'run.completed'
    | 'run.failed'
  data: Record<string, unknown>
}

export class ApiError extends Error {
  constructor(public readonly problem: ProblemDetail) {
    super(problem.detail)
  }
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1${path}`, {
    ...init,
    headers: { Accept: 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    const problem = await response.json() as ProblemDetail
    throw new ApiError(problem)
  }
  return response.json() as Promise<T>
}
```

API 类型最终应由 Step 11 的 OpenAPI 文档生成或与之做契约测试。不要在每个组件里复制 URL、`fetch` 和错误处理。

### 4. 正确解析 POST 的 SSE 流

`EventSource` 不支持 POST body，且简单的 `split("\\n")` 会破坏跨 chunk 的 JSON 或多行 `data`。解析器以空行分隔完整 SSE message，保留 `event` 与所有 `data:` 行：

```ts
// web/src/api/sse.ts
import type { RunEvent } from './client'

export async function* runStream(
  frameId: string,
  payload: { content: string; skill_ids: string[]; idempotency_key: string },
  signal: AbortSignal,
): AsyncGenerator<RunEvent> {
  const response = await fetch(`/api/v1/frames/${frameId}/runs/stream`, {
    method: 'POST',
    signal,
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(payload),
  })
  if (!response.ok || !response.body) throw new Error('Unable to start run')

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''

  while (true) {
    const { value = '', done } = await reader.read()
    buffer += value.replace(/\r\n/g, '\n')
    let boundary: number
    while ((boundary = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      const lines = block.split('\n')
      const event = lines.find((line) => line.startsWith('event:'))?.slice(6).trim()
      const data = lines
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trimStart())
        .join('\n')
      if (event && data) yield JSON.parse(data) as RunEvent
    }
    if (done) return
  }
}
```

服务端终态事件是协议的一部分：收到 `run.completed` 或 `run.failed` 后 store 必须清除运行中状态。网络突然关闭而没有终态时，UI 显示“连接已断开”，并允许用户查询 Run 状态或安全重试同一个 idempotency key。

### 5. 用 Pinia 管理一次对话流

```ts
// web/src/stores/conversation.ts
import { defineStore } from 'pinia'
import { ref } from 'vue'
import { apiFetch, type Message, type RunEvent } from '@/api/client'
import { runStream } from '@/api/sse'

export const useConversationStore = defineStore('conversation', () => {
  const messages = ref<Message[]>([])
  const activeRunId = ref<string | null>(null)
  const aborter = ref<AbortController | null>(null)
  const error = ref<string | null>(null)

  async function load(frameId: string) {
    messages.value = await apiFetch<Message[]>(`/frames/${frameId}/messages`)
  }

  async function send(frameId: string, content: string) {
    if (activeRunId.value || !content.trim()) return
    error.value = null
    messages.value.push({ id: crypto.randomUUID(), role: 'user', content, created_at: Date.now() })
    const draft: Message = {
      id: `draft-${crypto.randomUUID()}`, role: 'assistant', content: '', created_at: Date.now(),
    }
    messages.value.push(draft)
    aborter.value = new AbortController()
    activeRunId.value = 'pending'

    try {
      for await (const event of runStream(frameId, {
        content,
        skill_ids: [],
        idempotency_key: crypto.randomUUID(),
      }, aborter.value.signal)) {
        applyEvent(event, draft)
      }
    } catch (cause) {
      if (!aborter.value?.signal.aborted) error.value = String(cause)
    } finally {
      activeRunId.value = null
      aborter.value = null
    }
  }

  function applyEvent(event: RunEvent, draft: Message) {
    activeRunId.value = event.run_id
    if (event.type === 'message.delta') draft.content += String(event.data.text ?? '')
    if (event.type === 'run.failed') error.value = String(event.data.message ?? 'Run failed')
  }

  function cancel() {
    aborter.value?.abort()
  }

  return { messages, activeRunId, error, load, send, cancel }
})
```

真实实现还应把 `tool.*` 和 `approval.required` 写入各自的响应式列表，而不是塞进 Markdown 文本。切换 Frame 前取消旧 stream，或用 run id 验证事件仍属于当前视图，避免慢网络把 A Frame 的 token 写进 B Frame。

### 6. 用组件渲染，而非操作 DOM

```vue
<!-- web/src/components/ComposerForm.vue -->
<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits<{ send: [content: string]; cancel: [] }>()
defineProps<{ busy: boolean }>()
const content = ref('')

function submit() {
  const value = content.value.trim()
  if (!value) return
  emit('send', value)
  content.value = ''
}
</script>

<template>
  <form class="composer" @submit.prevent="submit">
    <label class="sr-only" for="message">发送消息</label>
    <textarea id="message" v-model="content" :disabled="busy"
      placeholder="输入消息；Enter 发送，Shift+Enter 换行"
      @keydown.enter.exact.prevent="submit" />
    <button v-if="busy" type="button" @click="emit('cancel')">停止生成</button>
    <button v-else type="submit">发送</button>
  </form>
</template>
```

所有用户文本默认通过插值 `{{ message.content }}` 渲染。若需要 Markdown，先在前端使用严格 allowlist 的 sanitizer，链接使用安全协议和 `rel="noopener noreferrer"`；绝不能将模型输出直接交给 `v-html`。

## 验证

```bash
# 终端 A：后端
uv run uvicorn scientex_agent.api_server:create_api --factory --reload

# 终端 B：Vue 开发服务器
cd web
pnpm dev

# 发布前
pnpm lint
pnpm build
pnpm test:unit
```

手动验证至少包括：刷新后从 URL 恢复 Project/Frame、流式 token 不丢失、取消按钮停止 UI、工具卡片和审批卡片可操作、断网显示可恢复错误、键盘可完成发送与焦点移动。使用 Vitest 测 SSE parser 和 store；使用 Playwright 覆盖真实代理下的聊天、审批和 Artifact 下载。

## 深入理解

### 响应式状态的归属

组件本地状态（textarea 内容、折叠卡片）留在组件内；跨组件且有副作用的状态（当前 Frame、消息、active run）留在 Pinia；后端是 Project/Frame/Artifact 的权威来源。这个分层让组件可独立测试，也避免“某个模块偷偷改全局对象”造成的流式竞态。

### 前端安全与可访问性

工具调用、运行错误和审批状态不能只用颜色表达，要有文字、图标、ARIA label 和可见焦点。对话自动滚动仅在用户原本位于底部时发生，避免阅读历史时被 token 拉回底部。错误消息采用 `aria-live="polite"`，但 token 逐字输出不要每次都打断屏幕阅读器。

## 当前局限

- 初版是单用户 SPA，不处理离线编辑、多人实时协作或复杂权限路由。
- SSE 断线后还没有按 `Last-Event-ID` 重放；UI 通过查询 Run 终态恢复。
- 前端不负责科学结论验证；它只展示 Step 15 产生的证据与状态。

## 下一步

下一章在稳定的 Agent、Artifact 和 Vue UI 之上加入科研领域能力：可复现分析工具、经过同意的记忆、证据和验证结果。
