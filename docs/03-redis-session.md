# 会话记忆与 Redis 用法

## 存储结构

每个会话一个 LIST：`sess:{session_id}`，元素是 JSON 序列化的消息：

    {"role":"user","content":"我叫张三"}
    {"role":"assistant","content":"已记录"}

写入时用 pipeline 一次发送三条命令：

    RPUSH sess:{id} <user> <assistant>
    LTRIM sess:{id} -20 -1
    EXPIRE sess:{id} 1800

## 三个设计理由

- **LTRIM**：上下文越长，token 成本与延迟越高，老对话价值递减，只保留最近 20 条。
- **EXPIRE**：会话不是持久数据，不设 TTL 会慢慢撑爆 Redis。
- **pipeline**：三次网络往返合成一次，在“每请求都写”的热路径上能明显降低尾延迟。

## 落库时机

用 `context.WithoutCancel` 派生一个 3 秒超时的 context 来写 Redis：
即使客户端提前断开，已经生成的半截回答也要存下来，否则下一轮上下文就断了。

## 隔离

不同 `session_id` 完全隔离；调试 prompt 时务必换新 session，
否则模型会被自己之前的历史回答带偏（历史污染）。
