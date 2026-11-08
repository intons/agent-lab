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

const systemPrompt = "你是大模型推理与 GPU 性能方向的工程师，回答简洁、准确，优先使用术语的本义。"

var (
	llmBaseURL = env("LLM_BASE_URL", "http://127.0.0.1:8001/v1")
	llmModel   = env("LLM_MODEL", "Qwen2.5-7B-Instruct")
	llmClient  = &http.Client{} // 不设 Timeout，统一由每次请求的 context 控制
	rdb        *redis.Client
)

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
	msgs = append(msgs, llmMsg{Role: "system", Content: systemPrompt})
	msgs = append(msgs, history...)
	return msgs
}

// 落库用独立 context：客户端断开也要把已生成的半截答案存下来
func storeCtx(r *http.Request) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.WithoutCancel(r.Context()), 3*time.Second)
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

	body, err := json.Marshal(llmReq{
		Model:       llmModel,
		Messages:    buildMessages(history),
		Temperature: 0.7,
		MaxTokens:   512,
	})
	if err != nil {
		http.Error(w, "marshal: "+err.Error(), http.StatusInternalServerError)
		return
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		llmBaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		http.Error(w, "new request: "+err.Error(), http.StatusInternalServerError)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := llmClient.Do(httpReq)
	if err != nil {
		http.Error(w, "llm call: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		http.Error(w, fmt.Sprintf("llm status %d: %s", resp.StatusCode, b), http.StatusBadGateway)
		return
	}

	var out llmResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		http.Error(w, "decode llm resp: "+err.Error(), http.StatusBadGateway)
		return
	}
	if out.Error != nil {
		http.Error(w, "llm error: "+out.Error.Message, http.StatusBadGateway)
		return
	}
	if len(out.Choices) == 0 {
		http.Error(w, "llm returned no choices", http.StatusBadGateway)
		return
	}

	reply := out.Choices[0].Message.Content

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

	body, err := json.Marshal(llmReq{
		Model:       llmModel,
		Messages:    buildMessages(history),
		Temperature: 0.7,
		MaxTokens:   512,
		Stream:      true,
	})
	if err != nil {
		http.Error(w, "marshal: "+err.Error(), http.StatusInternalServerError)
		return
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		llmBaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		http.Error(w, "new request: "+err.Error(), http.StatusInternalServerError)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := llmClient.Do(httpReq)
	if err != nil {
		http.Error(w, "llm call: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		http.Error(w, fmt.Sprintf("llm status %d: %s", resp.StatusCode, b), http.StatusBadGateway)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	var sb strings.Builder // 边转发边攒，结束时入库

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

		out, _ := json.Marshal(map[string]string{"delta": piece})
		fmt.Fprintf(w, "data: %s\n\n", out)
		flusher.Flush()
	}
	if err := sc.Err(); err != nil {
		log.Println("scan stream:", err)
	}

	sctx, scancel := storeCtx(r)
	defer scancel()
	if err := appendTurn(sctx, sid, req.Message, sb.String()); err != nil {
		log.Println("appendTurn:", err)
	}

	fmt.Fprintf(w, "data: %s\n\n", fmt.Sprintf(`{"done":true,"session_id":%q}`, sid))
	flusher.Flush()
}

// ---------- 入口 ----------

func main() {
	initRedis()

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