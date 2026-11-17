package main

// rag.go —— 极简检索：中文 bigram 分词 + BM25 + 引用返回
//
// 为什么不用向量检索：跑一个 embedding 模型要么再占几 GB 显存（这台卡已经是共享的），
// 要么走 CPU（慢且要额外依赖）。BM25 在这个规模（几十~几百个 chunk）上够用，
// 而且完全确定、可解释、零依赖。
//
// 升级路径：把 score() 换成余弦相似度即可，Chunk / Index 的结构不用动。

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unicode"
)

type chunk struct {
	Doc    string
	Text   string
	Tokens []string
	TF     map[string]int
	Len    int
}

type docIndex struct {
	mu       sync.RWMutex
	chunks   []chunk
	df       map[string]int // 文档频率
	avgLen   float64
	docDir   string
	chunkLen int
	overlap  int
}

var index = &docIndex{
	df:       map[string]int{},
	docDir:   env("DOC_DIR", "docs"),
	chunkLen: 300,
	overlap:  60,
}

// ---------- 分词 ----------

// 中文按 bigram、英文/数字按词，一起当 token
func tokenize(s string) []string {
	s = strings.ToLower(s)
	var out []string
	var word []rune
	var han []rune

	flushWord := func() {
		if len(word) > 0 {
			out = append(out, string(word))
			word = word[:0]
		}
	}
	flushHan := func() {
		if len(han) >= 2 {
			for i := 0; i+1 < len(han); i++ {
				out = append(out, string(han[i:i+2]))
			}
		} else if len(han) == 1 {
			out = append(out, string(han))
		}
		han = han[:0]
	}

	for _, r := range s {
		switch {
		case unicode.Is(unicode.Han, r):
			flushWord()
			han = append(han, r)
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			flushHan()
			word = append(word, r)
		default:
			flushWord()
			flushHan()
		}
	}
	flushWord()
	flushHan()
	return out
}

// ---------- 建索引 ----------

func (ix *docIndex) rebuild() error {
	files, err := os.ReadDir(ix.docDir)
	if err != nil {
		return err
	}

	var chunks []chunk
	for _, f := range files {
		if f.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(f.Name()))
		if ext != ".md" && ext != ".txt" {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(ix.docDir, f.Name()))
		if err != nil {
			continue
		}
		text := strings.TrimSpace(string(raw))
		if text == "" {
			continue
		}
		for _, part := range splitChunks(text, ix.chunkLen, ix.overlap) {
			toks := tokenize(part)
			tf := map[string]int{}
			for _, t := range toks {
				tf[t]++
			}
			chunks = append(chunks, chunk{
				Doc: f.Name(), Text: part, Tokens: toks, TF: tf, Len: len(toks),
			})
		}
	}

	df := map[string]int{}
	total := 0
	for _, c := range chunks {
		for t := range c.TF {
			df[t]++
		}
		total += c.Len
	}
	avg := 0.0
	if len(chunks) > 0 {
		avg = float64(total) / float64(len(chunks))
	}

	ix.mu.Lock()
	ix.chunks, ix.df, ix.avgLen = chunks, df, avg
	ix.mu.Unlock()
	return nil
}

// 按 rune 切块，带重叠，尽量在段落边界断开
func splitChunks(text string, size, overlap int) []string {
	runes := []rune(text)
	if len(runes) <= size {
		return []string{text}
	}
	var out []string
	step := size - overlap
	if step <= 0 {
		step = size
	}
	for start := 0; start < len(runes); start += step {
		end := start + size
		if end > len(runes) {
			end = len(runes)
		}
		out = append(out, strings.TrimSpace(string(runes[start:end])))
		if end == len(runes) {
			break
		}
	}
	return out
}

// ---------- BM25 ----------

type searchHit struct {
	Doc   string
	Text  string
	Score float64
}

func (ix *docIndex) search(query string, k int) []searchHit {
	qTokens := tokenize(query)
	ix.mu.RLock()
	defer ix.mu.RUnlock()

	const k1, b = 1.5, 0.75
	n := float64(len(ix.chunks))

	var hits []searchHit

	for _, c := range ix.chunks {
		var s float64
		for _, t := range qTokens {
			tf := c.TF[t]
			if tf == 0 {
				continue
			}
			df := float64(ix.df[t])
			idf := math.Log(1 + (n-df+0.5)/(df+0.5))
			denom := float64(tf) + k1*(1-b+b*float64(c.Len)/ix.avgLen)
			s += idf * (float64(tf) * (k1 + 1)) / denom
		}
		if s > 0 {
			hits = append(hits, searchHit{Doc: c.Doc, Text: c.Text, Score: s})
		}
	}

	sort.Slice(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
	if k <= 0 || k > len(hits) {
		k = len(hits)
	}
	out := make([]searchHit, 0, k)
	out = append(out, hits[:k]...)
	return out
}

// ---------- 工具实现 ----------

func docSearch(ctx context.Context, args json.RawMessage) (string, error) {
	var a struct {
		Query string `json:"query"`
		K     int    `json:"k"`
	}
	if err := json.Unmarshal(args, &a); err != nil || strings.TrimSpace(a.Query) == "" {
		return "", fmt.Errorf("query is required")
	}
	if a.K <= 0 {
		a.K = 3
	}
	if a.K > 5 {
		a.K = 5
	}

	hits := index.search(a.Query, a.K)
	if len(hits) == 0 {
		return "没有检索到相关内容", nil
	}

	var b strings.Builder
	for i, h := range hits {
		text := strings.TrimSpace(h.Text)
		if len([]rune(text)) > 400 {
			text = string([]rune(text)[:400]) + "…"
		}
		fmt.Fprintf(&b, "[%d] 来源=%s 相关度=%.2f\n%s\n\n", i+1, h.Doc, h.Score, text)
	}
	return b.String(), nil
}
