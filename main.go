package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

// ---------- 对外接口结构 ----------

type chatReq struct {
	Message   string `json:"message"`
	SessionID string `json:"session_id,omitempty"`
}

type chatResp struct {
	Reply     string `json:"reply"`
	Runes     int    `json:"runes"`
	SessionID string `json:"session_id"`
}

// ---------- OpenAI 兼容结构 ----------

type llmMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type llmReq struct {
	Model       string   `json:"model"`
	Messages    []llmMsg `json:"messages"`
	Temperature float64  `json:"temperature,omitempty"`
	MaxTokens   int      `json:"max_tokens,omitempty"`
	Stream      bool     `json:"stream,omitempty"`
}

type llmResp struct {
	Choices []struct {
		Message llmMsg `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type llmStreamChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
}

const basePrompt = `你是大模型推理与 GPU 性能方向的工程师，回答简洁准确。

【可用工具】
%TOOLS%

【规则】
1. 需要工具时只输出一行 JSON，禁止任何其他字符（不要解释、不要 markdown 代码块、不要句号）：
   {"tool":"now","args":{}}
   {"tool":"kv_count","args":{"key":"sess:s1"}}
2. 收到以「工具结果:」开头的消息后，再用自然语言回答。
3. 不需要工具时直接回答，不要输出 JSON。
4. 只要调用过 doc_search，回答的最后必须单独一行输出：来源: <文件名>（文件名取自检索结果里的「来源=」）。`

// prompt 里的工具清单由注册表生成：加工具只改 tools 一处
func systemPrompt() string {
	return strings.Replace(basePrompt, "%TOOLS%", toolsPrompt(), 1)
}

var (
	llmBaseURL = env("LLM_BASE_URL", "http://127.0.0.1:8001/v1")
	llmModel   = env("LLM_MODEL", "Qwen2.5-7B-Instruct")
	llmClient  = &http.Client{} // 不设 Timeout，统一由每次请求的 context 控制
	rdb        *redis.Client
)

func shorten(s string, n int) string {
	r := []rune(strings.TrimSpace(s))
	if len(r) <= n {
		return string(r)
	}
	return string(r[:n]) + "…"
}

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// ---------- Redis ----------

func initRedis() {
	rdb = redis.NewClient(&redis.Options{
		Addr: env("REDIS_ADDR", "127.0.0.1:6379"),
	})
	if err := rdb.Ping(context.Background()).Err(); err != nil {
		log.Fatalf("redis: %v", err)
	}
	log.Println("redis connected:", rdb.Options().Addr)
}

func sessionKey(sid string) string { return "sess:" + sid }

// 读最近 limit 条
func loadHistory(ctx context.Context, sid string, limit int) ([]llmMsg, error) {
	vals, err := rdb.LRange(ctx, sessionKey(sid), -int64(limit), -1).Result()
	if err != nil {
		return nil, err
	}
	out := make([]llmMsg, 0, len(vals))
	for _, v := range vals {
		var m llmMsg
		if json.Unmarshal([]byte(v), &m) == nil {
			out = append(out, m)
		}
	}
	return out, nil
}

// 一轮问答一次性写入：RPUSH 两条 + 裁剪 + 过期
func appendTurn(ctx context.Context, sid, userMsg, assistantMsg string) error {
	ub, _ := json.Marshal(llmMsg{Role: "user", Content: userMsg})
	ab, _ := json.Marshal(llmMsg{Role: "assistant", Content: assistantMsg})

	key := sessionKey(sid)
	pipe := rdb.Pipeline()
	pipe.RPush(ctx, key, ub, ab)
	pipe.LTrim(ctx, key, -20, -1) // 只留最近 20 条
	pipe.Expire(ctx, key, 30*time.Minute)
	_, err := pipe.Exec(ctx)
	return err
}

// ---------- 工具 ----------

func ensureSession(req *chatReq) string {
	if req.SessionID == "" {
		req.SessionID = fmt.Sprintf("s%d", time.Now().UnixNano())
	}
	return req.SessionID
}

func buildMessages(history []llmMsg) []llmMsg {
	msgs := make([]llmMsg, 0, len(history)+1)
	msgs = append(msgs, llmMsg{Role: "system", Content: systemPrompt()})
	msgs = append(msgs, history...)
	return msgs
}

// 落库用独立 context：客户端断开也要把已生成的半截答案存下来
func storeCtx(r *http.Request) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.WithoutCancel(r.Context()), 3*time.Second)
}

// ---------- 工具注册表 ----------

type toolDef struct {
	Name string
	Desc string
	Args string // 参数示例，用于自动生成 prompt
	Run  func(ctx context.Context, args json.RawMessage) (string, error)
}

var tools = map[string]toolDef{
	"doc_search": {
		Name: "doc_search",
		Desc: "在本地文档库中检索相关段落，返回带来源的片段（知识库问答）",
		Args: `{"query":"KV cache 显存怎么算","k":3}`,
		Run:  docSearch,
	},
	"now": {
		Name: "now",
		Desc: "返回当前时间（RFC3339，UTC）",
		Args: `{}`,
		Run: func(ctx context.Context, _ json.RawMessage) (string, error) {
			return time.Now().UTC().Format(time.RFC3339), nil
		},
	},
	"kv_count": {
		Name: "kv_count",
		Desc: "返回 Redis 中某个键（list）的元素个数",
		Args: `{"key":"sess:s1"}`,
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			var a struct {
				Key string `json:"key"`
			}
			if err := json.Unmarshal(args, &a); err != nil || a.Key == "" {
				return "", fmt.Errorf("key is required")
			}
			n, err := rdb.LLen(ctx, a.Key).Result()
			if err != nil {
				return "", err
			}
			return fmt.Sprintf("%d", n), nil
		},
	},
	"redis_get": {
		Name: "redis_get",
		Desc: "读取 Redis 中某个键的字符串值",
		Args: `{"key":"foo"}`,
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			var a struct {
				Key string `json:"key"`
			}
			if err := json.Unmarshal(args, &a); err != nil || a.Key == "" {
				return "", fmt.Errorf("key is required")
			}
			v, err := rdb.Get(ctx, a.Key).Result()
			if err == redis.Nil {
				return "(键不存在)", nil
			}
			if err != nil {
				return "", err
			}
			return v, nil
		},
	},
}

// 生成给模型看的工具清单（顺序固定，便于排查）
func toolsPrompt() string {
	names := make([]string, 0, len(tools))
	for n := range tools {
		names = append(names, n)
	}
	sort.Strings(names)

	var b strings.Builder
	for _, n := range names {
		t := tools[n]
		fmt.Fprintf(&b, "- %s：%s，参数示例 %s\n", t.Name, t.Desc, t.Args)
	}
	return b.String()
}

// ---------- 工具调用 ----------

type toolCall struct {
	Tool string          `json:"tool"`
	Args json.RawMessage `json:"args"`
}

// 从模型输出里抠出 JSON（容忍代码块、前后废话、多余空白）
func parseToolCall(s string) (*toolCall, bool) {
	s = strings.TrimSpace(s)
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start < 0 || end <= start {
		return nil, false
	}
	var tc toolCall
	if err := json.Unmarshal([]byte(s[start:end+1]), &tc); err != nil {
		return nil, false
	}
	if strings.TrimSpace(tc.Tool) == "" {
		return nil, false
	}
	return &tc, true
}

func execTool(ctx context.Context, tc *toolCall) (string, error) {
	name := strings.TrimSpace(tc.Tool)
	t, ok := tools[name]
	if !ok {
		return "", fmt.Errorf("unknown tool: %s", name)
	}
	return t.Run(ctx, tc.Args)
}

// 单次非流式调用
func chatOnce(ctx context.Context, msgs []llmMsg) (string, error) {
	body, err := json.Marshal(llmReq{
		Model:       llmModel,
		Messages:    msgs,
		Temperature: 0,
		MaxTokens:   512,
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		llmBaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := llmClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return "", fmt.Errorf("llm status %d: %s", resp.StatusCode, b)
	}

	var out llmResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	if out.Error != nil {
		return "", fmt.Errorf("llm error: %s", out.Error.Message)
	}
	if len(out.Choices) == 0 {
		return "", fmt.Errorf("empty choices")
	}
	return out.Choices[0].Message.Content, nil
}

// 带工具循环的对话：模型要工具 → 执行 → 结果回填 → 再问一轮
func runAgent(ctx context.Context, history []llmMsg) (string, error) {
	const maxSteps = 4
	for step := 0; step < maxSteps; step++ {
		r, err := chatOnce(ctx, buildMessages(history))
		if err != nil {
			return "", err
		}
		history = append(history, llmMsg{Role: "assistant", Content: r})

		tc, ok := parseToolCall(r)
		if !ok {
			return r, nil
		}

		res, terr := execTool(ctx, tc)
		if terr != nil {
			res = "工具执行失败: " + terr.Error()
		}
		log.Printf("step %d tool=%s result=%s", step, tc.Tool, shorten(res, 120))
		history = append(history, llmMsg{Role: "user", Content: "工具结果: " + res})
	}
	return "", fmt.Errorf("too many tool steps")
}

// ---------- 非流式 ----------

func chatHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	var req chatReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad json: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Message == "" {
		http.Error(w, "message is required", http.StatusBadRequest)
		return
	}

	sid := ensureSession(&req)

	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()

	history, err := loadHistory(ctx, sid, 10)
	if err != nil {
		http.Error(w, "load history: "+err.Error(), http.StatusInternalServerError)
		return
	}
	history = append(history, llmMsg{Role: "user", Content: req.Message})

	reply, err := runAgent(ctx, history)
	if err != nil {
		http.Error(w, "agent: "+err.Error(), http.StatusBadGateway)
		return
	}

	sctx, scancel := storeCtx(r)
	defer scancel()
	if err := appendTurn(sctx, sid, req.Message, reply); err != nil {
		log.Println("appendTurn:", err) // 存不下来不影响这次回答
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	if err := json.NewEncoder(w).Encode(chatResp{
		Reply:     reply,
		Runes:     len([]rune(reply)),
		SessionID: sid,
	}); err != nil {
		log.Println("encode resp:", err)
	}
}

// ---------- 流式 + 工具 ----------

// 一次流式调用：先判断这段输出是"工具调用"还是"最终回答"
// 返回: 完整文本, 是否已经把内容推给客户端, 错误
func streamOnce(ctx context.Context, msgs []llmMsg, sseSend func(any)) (string, bool, error) {
	body, err := json.Marshal(llmReq{
		Model:       llmModel,
		Messages:    msgs,
		Temperature: 0,
		MaxTokens:   512,
		Stream:      true,
	})
	if err != nil {
		return "", false, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		llmBaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", false, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := llmClient.Do(req)
	if err != nil {
		return "", false, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return "", false, fmt.Errorf("llm status %d: %s", resp.StatusCode, b)
	}

	var sb strings.Builder
	forwarded := false
	decided := false

	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}

		var chunk llmStreamChunk
		if json.Unmarshal([]byte(payload), &chunk) != nil {
			continue
		}
		if len(chunk.Choices) == 0 || chunk.Choices[0].Delta.Content == "" {
			continue
		}

		piece := chunk.Choices[0].Delta.Content
		sb.WriteString(piece)

		if !decided {
			decided = true
			// 首段以 { 开头 → 判定为工具调用，静默攒着，不推给客户端
			if !strings.HasPrefix(strings.TrimSpace(sb.String()), "{") {
				forwarded = true
			}
		}
		if forwarded {
			sseSend(map[string]string{"delta": piece})
		}
	}
	if err := sc.Err(); err != nil {
		return sb.String(), forwarded, err
	}
	return sb.String(), forwarded, nil
}

// 流式版工具循环：中间步骤静默，最终回答逐字推给客户端
func runAgentStream(ctx context.Context, history []llmMsg, sseSend func(any)) (string, error) {
	const maxSteps = 4
	for step := 0; step < maxSteps; step++ {
		out, forwarded, err := streamOnce(ctx, buildMessages(history), sseSend)
		if err != nil {
			return "", err
		}
		history = append(history, llmMsg{Role: "assistant", Content: out})

		tc, ok := parseToolCall(out)
		if !ok {
			if !forwarded && strings.TrimSpace(out) != "" {
				sseSend(map[string]string{"delta": out})
			}
			return out, nil
		}

		res, terr := execTool(ctx, tc)
		if terr != nil {
			// 模型吐了个不存在的工具：当成普通回答，别陷入循环
			if strings.Contains(terr.Error(), "unknown tool") {
				if !forwarded && strings.TrimSpace(out) != "" {
					sseSend(map[string]string{"delta": out})
				}
				return out, nil
			}
			res = "工具执行失败: " + terr.Error()
		}
		log.Printf("stream step %d tool=%s result=%s", step, tc.Tool, shorten(res, 120))
		sseSend(map[string]any{"tool": tc.Tool, "result": res})

		history = append(history, llmMsg{Role: "user", Content: "工具结果: " + res})
	}
	return "", fmt.Errorf("too many tool steps")
}

// ---------- 流式 ----------

func chatStreamHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	var req chatReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Message == "" {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	sid := ensureSession(&req)

	ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second)
	defer cancel()

	history, err := loadHistory(ctx, sid, 10)
	if err != nil {
		http.Error(w, "load history: "+err.Error(), http.StatusInternalServerError)
		return
	}
	history = append(history, llmMsg{Role: "user", Content: req.Message})

	sseSend := func(v any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "data: %s\n\n", b)
		flusher.Flush()
	}

	reply, err := runAgentStream(ctx, history, sseSend)
	if err != nil {
		sseSend(map[string]string{"error": err.Error()})
		return
	}

	sctx, scancel := storeCtx(r)
	defer scancel()
	if err := appendTurn(sctx, sid, req.Message, reply); err != nil {
		log.Println("appendTurn:", err)
	}

	sseSend(map[string]any{"done": true, "session_id": sid})
}

// ---------- 入口 ----------

func main() {
	initRedis()
	if err := index.rebuild(); err != nil {
		log.Println("index rebuild:", err)
	} else {
		log.Printf("doc index ready: %d chunks (dir=%s)", len(index.chunks), index.docDir)
	}

	mux := http.NewServeMux()

	mux.HandleFunc("/ping", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "pong")
	})

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 500*time.Millisecond)
		defer cancel()
		redisOK := rdb.Ping(ctx).Err() == nil

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"ok":           redisOK,
			"redis":        redisOK,
			"llm_base_url": llmBaseURL,
			"model":        llmModel,
		})
	})

	mux.HandleFunc("/reindex", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if err := index.rebuild(); err != nil {
			http.Error(w, "reindex: "+err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"ok": true, "chunks": len(index.chunks)})
	})

	mux.HandleFunc("/chat", chatHandler)
	mux.HandleFunc("/chat/stream", chatStreamHandler)

	srv := &http.Server{
		Addr:              "127.0.0.1:18080",
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Println("listening on 127.0.0.1:18080, llm =", llmBaseURL)
	log.Fatal(srv.ListenAndServe())
}
