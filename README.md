# agent-lab

一个用 Go 写的 LLM 网关 / 工具调用 Agent，后端接自部署的 vLLM（OpenAI 兼容协议），
带流式输出、Redis 会话记忆和可扩展的工具注册表。

```
                   ┌───────────────────────────────┐
   curl / 前端 ───▶ │  Go 网关  :18080              │
                   │  ├─ /ping  /health            │
                   │  ├─ /chat        非流式       │
                   │  └─ /chat/stream SSE 流式     │
                   └───────┬───────────────┬───────┘
                           │               │
                 HTTP(OpenAI 协议)      Redis :6379
                           │               │
                   ┌───────▼───────┐   ┌───▼──────────────┐
                   │ vLLM  :8001   │   │ sess:{id} LIST   │
                   │ Qwen2.5-7B    │   │ 最近 20 条 / 30m │
                   └───────────────┘   └──────────────────┘
```

## 快速开始

```bash
# 0) 依赖：Go 1.22+、Redis、一个 OpenAI 兼容的推理服务
redis-server --port 6379 --daemonize yes --dir /data/ywb/redis

# 1) 编译
go build -o /data/ywb/bin/agent-gw main.go

# 2) 启动
LLM_MODEL=/data/ywb/models/Qwen2.5-7B-Instruct \
  setsid nohup /data/ywb/bin/agent-gw > /data/ywb/logs/gw.log 2>&1 &

# 3) 冒烟测试（13 条）
bash smoke.sh

# 4) 压测
python bench.py -c 1,4,8,16 -n 8
```

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/ping` | 存活探针，返回 `pong` |
| GET | `/health` | 依赖健康（Redis + vLLM） |
| POST | `/chat` | 非流式，返回完整回答 |
| POST | `/chat/stream` | SSE 流式，带工具进度事件 |
| POST | `/reindex` | 重建文档检索索引，返回 chunk 数 |

请求体：

```json
{"session_id": "s1", "message": "现在几点了？"}
```

`session_id` 可省略（服务端生成并回传）。

流式响应帧：

| 帧 | 含义 |
|---|---|
| `data: {"delta":"字"}` | 一个 token |
| `data: {"tool":"now","result":"..."}` | 执行了一次工具 |
| `data: {"done":true,"session_id":"s1"}` | 结束 |

## 一次请求的生命周期

```
chatHandler / chatStreamHandler
  ├─ 解析 JSON，校验 message
  ├─ ensureSession            没有 sid 就生成
  ├─ loadHistory              Redis LRANGE 最近 10 条
  ├─ buildMessages            systemPrompt() + history
  ├─ runAgent / runAgentStream
  │    └─ 循环（最多 4 步）:
  │         chatOnce / streamOnce      调模型
  │         parseToolCall              输出是工具调用吗？
  │           是 → execTool → 追加「工具结果: ...」→ 下一轮
  │           否 → 这就是最终回答
  └─ appendTurn               RPUSH 两条 + LTRIM 20 + EXPIRE 30m
```

## 模块

| 模块 | 位置 | 职责 |
|---|---|---|
| 配置 | `env()` + 包级变量 | `LLM_BASE_URL` / `LLM_MODEL` / `REDIS_ADDR` |
| 会话 | `loadHistory` / `appendTurn` | Redis LIST，按 session 隔离，30 分钟过期 |
| 工具注册表 | `tools` map | 名字 / 描述 / 参数示例 / 执行函数 |
| Prompt | `systemPrompt()` + `toolsPrompt()` | 工具清单由注册表自动生成 |
| 工具循环 | `runAgent` / `runAgentStream` | 非流式 / 流式两种实现 |
| HTTP | `main()` | 路由 + Server（`ReadHeaderTimeout`） |

## 内置工具

| 工具 | 作用 |
|---|---|
| `doc_search` | BM25 检索 `docs/` 下的 Markdown/文本，返回带来源的片段 |
| `now` | 当前时间（RFC3339） |
| `kv_count` | Redis list 长度 |
| `redis_get` | 读取 Redis 字符串值 |

检索用中文 bigram 分词 + BM25，零依赖、可解释；升级成向量检索只需替换 `rag.go` 里的打分函数，索引结构不变。

## 加一个新工具

只改一处 —— 在 `tools` map 里加一项：

```go
"http_get": {
    Name: "http_get",
    Desc: "抓取一个 URL 的文本内容",
    Args: `{"url":"https://example.com"}`,
    Run: func(ctx context.Context, args json.RawMessage) (string, error) {
        // ...
    },
},
```

重新编译启动即可，prompt 会自动多出这个工具。**执行必须在白名单里**，模型只能"请求"，不能"命令"。

## 设计决策

1. **三层分离**：接入（HTTP）→ 业务（Agent 循环）→ 适配（OpenAI 协议）。换模型只改适配层。
2. **错误分层**：400 调用方错 / 502 下游依赖错 / 500 本服务错，便于定位与告警归属。
3. **全链路 context**：请求超时、客户端断开、上游取消三者联动，避免 GPU 空转。
4. **流式首段判定**：第一段以 `{` 开头视为工具调用（静默执行），否则边收边推；工具 JSON 永不闪给用户。
5. **工具白名单**：注册表 + `default` 报错，杜绝模型驱动的任意执行。

## 性能（Qwen2.5-7B / RTX 3090 24G / vLLM 0.31 / eager / ctx 4096）

| 并发 | RPS | tok/s | TTFT p50 | TTFT p95 | 延迟 p95 |
|---|---|---|---|---|---|
| 1 | 1.64 | 48 | 48 ms | 51 ms | 668 ms |
| 4 | 5.98 | 179 | 73 ms | 85 ms | 735 ms |
| 8 | 10.15 | 317 | 101 ms | 108 ms | 778 ms |
| 16 | 10.13 | 299 | 104 ms | 114 ms | 788 ms |

并发 1→8 近线性，8 以后饱和（≈10 RPS / 300 tok/s）。瓶颈是 KV cache 约 2 GB 限制了批大小，且 eager 模式放大了 kernel launch 开销。

## 已知限制

- 工具参数没有 schema 校验（错了才报）
- `redis_get` 可读任意键，缺前缀白名单
- 工具执行期间客户端无进度提示（可加 `thinking` 事件）
- 无鉴权、无限流（仅本地回环，暂不对外）
- 无评测集，成功率未量化

## Roadmap

- [x] 文件检索工具（RAG，BM25）\n- [ ] 向量检索 + 重排
- [ ] 30 条评测集与指标（成功率 / 引用命中 / P95 / 单任务成本）
- [ ] 工具执行进度事件
- [ ] 结构化日志与 Prometheus 指标
