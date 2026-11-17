#!/usr/bin/env bash
# smoke.sh —— 网关冒烟测试（10 个用例）
# 用法：bash smoke.sh
# 退出码：0 = 全通过，1 = 有失败

set -u
BASE="${BASE:-http://127.0.0.1:18080}"
PASS=0
FAIL=0
FAILED=""

check() { # $1 名称   $2 条件结果(1/0)  $3 失败时附带的信息
  if [ "$2" = "1" ]; then
    PASS=$((PASS + 1))
    printf "  \033[32mPASS\033[0m  %s\n" "$1"
  else
    FAIL=$((FAIL + 1))
    printf "  \033[31mFAIL\033[0m  %s\n" "$1"
    [ -n "${3:-}" ] && printf "        ↳ %s\n" "$3"
  fi
}

post() { # $1 path  $2 json
  curl -s -m 180 -X POST "$BASE$1" -H 'Content-Type: application/json' -d "$2"
}

code() { # $1 path  $2 json
  curl -s -m 180 -o /dev/null -w '%{http_code}' -X POST "$BASE$1" \
    -H 'Content-Type: application/json' -d "$2"
}

echo "== 1. 基础接口"
out=$(curl -s -m 10 "$BASE/ping")
check "/ping 返回 pong" "$([ "$out" = "pong" ] && echo 1 || echo 0)" "实际: $out"

h=$(curl -s -m 10 "$BASE/health")
check "/health 健康" "$(echo "$h" | grep -q '"ok":true' && echo 1 || echo 0)" "实际: $h"

echo "== 2. 普通问答（不需要工具）"
out=$(post /chat '{"session_id":"smoke-q","message":"用一句话说明内存带宽是什么"}')
reply=$(echo "$out" | sed -n 's/.*"reply":"\(.*\)","runes".*/\1/p')
check "返回了非空回答" "$([ ${#reply} -gt 10 ] && echo 1 || echo 0)" "实际: $out"
check "没有把工具 JSON 透出去" "$(echo "$reply" | grep -q '{"tool"' && echo 0 || echo 1)" "实际: $reply"

echo "== 3. 工具：now"
out=$(post /chat '{"session_id":"smoke-now","message":"现在几点了？"}')
check "now 被调用并回答时间" "$(echo "$out" | grep -qE '2026|时间' && echo 1 || echo 0)" "实际: $out"

echo "== 4. 工具：kv_count"
post /chat '{"session_id":"smoke-kv","message":"记住：这是一条测试消息"}' >/dev/null
out=$(post /chat '{"session_id":"smoke-kv","message":"sess:smoke-kv 里有几条消息？"}')
check "kv_count 返回条数" "$(echo "$out" | grep -qE '[0-9]+ ?条|没有' && echo 1 || echo 0)" "实际: $out"

echo "== 5. 工具：redis_get"
/data/ywb/bin/redis-cli set smoke:foo hello-agent >/dev/null 2>&1 || redis-cli set smoke:foo hello-agent >/dev/null 2>&1
out=$(post /chat '{"session_id":"smoke-get","message":"读一下 Redis 里 smoke:foo 这个键"}')
check "redis_get 读到值" "$(echo "$out" | grep -q 'hello-agent' && echo 1 || echo 0)" "实际: $out"

echo "== 6. 多轮记忆"
post /chat '{"session_id":"smoke-mem","message":"我叫张三，记住"}' >/dev/null
out=$(post /chat '{"session_id":"smoke-mem","message":"我叫什么名字？"}')
check "第二轮记得名字" "$(echo "$out" | grep -q '张三' && echo 1 || echo 0)" "实际: $out"

echo "== 7. 错误处理"
c=$(code /chat '{bad json')
check "非法 JSON 返回 400" "$([ "$c" = "400" ] && echo 1 || echo 0)" "实际 HTTP $c"

c=$(code /chat '{"message":""}')
check "空 message 返回 400" "$([ "$c" = "400" ] && echo 1 || echo 0)" "实际 HTTP $c"

c=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$BASE/chat")
check "GET /chat 返回 405" "$([ "$c" = "405" ] && echo 1 || echo 0)" "实际 HTTP $c"

echo "== 8. 流式"
first=$(curl -s -N -m 180 -X POST "$BASE/chat/stream" \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"smoke-s","message":"数到三"}' | head -c 200)
check "流式返回 SSE 帧" "$(echo "$first" | grep -q 'data: ' && echo 1 || echo 0)" "实际: $first"

echo "== 9. 并发 8 个请求"
tmp=$(mktemp -d)
for i in $(seq 1 8); do
  (curl -s -m 180 -o "$tmp/$i" -X POST "$BASE/chat" -H 'Content-Type: application/json' \
    -d "{\"session_id\":\"smoke-c$i\",\"message\":\"用一句话解释第 $i 个概念\"}") &
done
wait
ok=0
for i in $(seq 1 8); do
  grep -q '"reply":"[^"]' "$tmp/$i" && ok=$((ok + 1))
done
rm -rf "$tmp"
check "8 个并发全部返回" "$([ "$ok" = "8" ] && echo 1 || echo 0)" "成功 $ok/8"

echo
echo "----------------------------------------"
printf "通过 %d，失败 %d\n" "$PASS" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  printf "失败项:%s\n" "$FAILED"
  exit 1
fi
exit 0
