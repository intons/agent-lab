# KV cache 与显存计算

KV cache 是推理时缓存历史 token 的 Key/Value 张量，避免每生成一个 token 都重算整个前缀。
它的大小可以按下面的公式估算：

    每个 token 的 KV 大小 = 2 × 层数 × 注意力头数 × head_dim × 精度字节数
    总 KV 显存 = 每 token 大小 × 并发序列数 × 序列长度

以 Qwen2.5-7B 为例，28 层、GQA（4 个 KV 头）、head_dim 128、fp16：
每 token 约 2 × 28 × 4 × 128 × 2 ≈ 57 KB。8k 上下文、单条请求就是约 470 MB。

因此在 24 GB 的 RTX 3090 上，权重占 15 GB 左右，剩下的显存全部用于 KV cache。
vLLM 启动日志会打印 `GPU KV cache size: N tokens` 和
`Maximum concurrency for X tokens per request`，这两个数字才是真实并发能力。
显存不够时优先降 `--max-model-len`，其次降 `--gpu-memory-utilization`。

## 常见误区

- 以为并发只由 GPU 算力决定：实际上多数 7B 服务是 KV cache 容量先到顶。
- 以为 `--gpu-memory-utilization` 是“性能参数”：它是显存预算，不是速度旋钮。
- 忽略 PagedAttention 的分页作用：它把 KV 切成固定大小的块，减少碎片，
  同样显存能装更多并发序列。
