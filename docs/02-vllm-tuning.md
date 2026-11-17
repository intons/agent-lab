# vLLM 部署与调优笔记

## 关键启动参数

- `--model`：模型路径；可用 `--served-model-name` 起一个短名字，避免请求里写绝对路径。
- `--max-model-len`：最大上下文。它直接决定 KV cache 的峰值，太大容易 OOM。
- `--gpu-memory-utilization`：显存预算比例。共享机器上按“剩余显存”算，不要用 0.9。
- `--enforce-eager`：关闭 CUDA graph。省 1–2 GB 显存，代价是 kernel launch 开销上升，
  实测吞吐下降约 10%–20%。

## 共享 GPU 的做法

1. 先 `nvidia-smi` 看 Processes 段，确认哪张卡空闲；
2. 用 `CUDA_VISIBLE_DEVICES` 指定显卡；
3. 预算按 `(总显存 - 已用) / 总显存 - 余量` 估算；
4. 用 nohup + 日志文件后台运行，方便排查。

## 观测指标

- TTFT：首 token 延迟，主要由 prefill 决定，随输入长度增长。
- TPOT / 生成速度：decode 阶段，主要由显存带宽和 batch 决定。
- 吞吐（tok/s）：并发上来后趋于饱和，饱和点就是服务容量。
- 队列长度与 GPU 利用率：判断是不是在排队，而不是在算。
