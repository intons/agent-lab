# 实验记录

## W1 L4 流式
- 2026-11-08 模型 Qwen2.5-7B / 3090 / vLLM 0.31
- 输入: "用三句话讲讲什么是 KV cache"
- TTFT 39ms, 总 1.26s, 输出 57 token, ≈47 tok/s
- 观察: 模型把 KV cache 理解成通用键值存储 → 需要 system prompt 限定领域
