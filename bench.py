#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench.py —— 并发压测：TTFT / 端到端延迟 / 吞吐

用法:
  python bench.py                       # 默认 1,4,8,16 并发，每档 8 个请求
  python bench.py -c 1,4,8 -n 10        # 指定并发和每档请求数
  python bench.py --stream 0            # 用非流式接口测吞吐
"""
import argparse
import csv
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = os.environ.get("BASE", "http://127.0.0.1:18080")
PROMPT = "用三句话解释什么是内存带宽"


def one_stream(i, prompt, max_tokens):
    """发一个流式请求，返回 (ttft, total, tokens)"""
    payload = {"session_id": f"bench-{i}-{time.time_ns()}", "message": prompt}
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    with requests.post(BASE + "/chat/stream", json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "delta" in obj:
                tokens += 1
            elif obj.get("done"):
                break
    total = time.perf_counter() - t0
    return (ttft if ttft is not None else total), total, tokens


def one_plain(i, prompt, max_tokens):
    payload = {"session_id": f"bench-{i}-{time.time_ns()}", "message": prompt}
    t0 = time.perf_counter()
    r = requests.post(BASE + "/chat", json=payload, timeout=300)
    r.raise_for_status()
    total = time.perf_counter() - t0
    body = r.json()
    return total, total, max(1, int(body.get("runes", 0) / 1.6))


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def run_level(conc, n, fn, prompt, max_tokens, warmup=2):
    for w in range(warmup):
        try:
            fn(-1 - w, prompt, 32)
        except Exception as e:
            print("warmup failed:", e)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        results = list(ex.map(lambda i: fn(i, prompt, max_tokens), range(n)))
    wall = time.perf_counter() - t0

    ttfts = [r[0] for r in results]
    totals = [r[1] for r in results]
    toks = [r[2] for r in results]
    total_tokens = sum(toks)

    return {
        "conc": conc,
        "n": n,
        "wall": wall,
        "rps": n / wall,
        "tok_per_s": total_tokens / wall,
        "tok_per_req": total_tokens / n,
        "ttft_p50": pct(ttfts, 50),
        "ttft_p95": pct(ttfts, 95),
        "lat_p50": pct(totals, 50),
        "lat_p95": pct(totals, 95),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--conc", default="1,4,8,16")
    ap.add_argument("-n", "--num", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--stream", type=int, default=1)
    args = ap.parse_args()

    fn = one_stream if args.stream else one_plain
    levels = [int(x) for x in args.conc.split(",")]

    rows = []
    for c in levels:
        r = run_level(c, args.num, fn, PROMPT, args.max_tokens)
        rows.append(r)
        print(
            f"concurrency={c:<3d} rps={r['rps']:6.2f} tok/s={r['tok_per_s']:7.1f} "
            f"ttft_p50={r['ttft_p50']*1000:7.1f}ms ttft_p95={r['ttft_p95']*1000:7.1f}ms "
            f"lat_p95={r['lat_p95']*1000:7.1f}ms",
            flush=True,
        )

    out = os.path.join("/data/ywb/logs", time.strftime("bench_%Y%m%d_%H%M%S.csv"))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nCSV ->", out)

    print("\n| 并发 | RPS | tok/s | TTFT p50 | TTFT p95 | 延迟 p95 |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['conc']} | {r['rps']:.2f} | {r['tok_per_s']:.0f} | "
            f"{r['ttft_p50']*1000:.0f} ms | {r['ttft_p95']*1000:.0f} ms | {r['lat_p95']*1000:.0f} ms |"
        )


if __name__ == "__main__":
    main()
