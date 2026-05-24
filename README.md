# Repo-Query

A Retrieval-Augmented Generation (RAG) backend for querying codebases, PDFs, and web pages via natural language. Ingest any GitHub repository, local codebase, PDF document, or URL and ask questions about it through a streaming chat interface.

---

## Features

- **Multi-source ingestion** — GitHub repos (via Git Trees API), local paths, PDFs, and web URLs in a single pipeline run
- **Python AST chunking** — parses source files into class, method, and function symbols; extracts signatures, docstrings, and dependency edges
- **Hybrid search** — FAISS dense retrieval fused with BM25 sparse retrieval using Reciprocal Rank Fusion (RRF)
- **Query rewriting** — four strategies: `multi_query`, `HyDE`, `step_back`, and `contextual` standalone rewrite
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` reranker with sentence-level context compression
- **Sliding-window memory** — token-budget-aware conversation memory with automatic LLM summarization on eviction
- **Streaming responses** — token-level SSE streaming via `/query/stream`
- **Background ingestion jobs** — async job queue with status polling via `/ingest/status/{job_id}`
- **Token tracking** — per-request JSONL ledger with cost estimation via LiteLLM
- **Streamlit UI** — full chat interface with ingest controls, source chips, and memory reset

---

## Architecture

```
Ingestion Layer
├── CodeIngestor      — GitHub Git Trees API + local filesystem walk
├── PdfIngestor       — pymupdf native text + OCR fallback (pytesseract) + table extraction
└── WebCrawler        — crawl4ai with LLM extraction strategy + content pruning

Chunking Layer
├── PythonASTParser   — class/method/function symbols → SymbolIR → Chunk
├── ChunkPolicyEngine — filters trivial symbols (min_lines, merge_small_methods)
├── TextChunker       — recursive / token / semantic strategies via LangChain
└── FallbackTextChunker — single-blob fallback for unsupported languages

Embedding Layer
├── Embedder          — SentenceTransformer BAAI/bge-m3 (1024-dim, normalized)
├── CodeEmbeddingProjector — enriches with symbol type, name, signature, docstring
└── TextEmbeddingProjector — plain text pass-through

Vector Store
├── FAISS             — IndexFlatIP (cosine via L2-normalized vectors) or HNSW
├── BM25Okapi         — sparse keyword index (rank_bm25)
└── RRF fusion        — Reciprocal Rank Fusion over dense + sparse ranked lists

Query Layer
├── QueryRewriter     — multi_query / HyDE / step_back / contextual rewrite
├── CrossEncoderReranker — sentence-transformer cross-encoder, scores chunk pairs
├── ContextCompressor — sentence extraction + Jaccard dedup + token budget fitting
└── QueryEngine       — orchestrates rewrite → retrieve → rerank → compress → generate

Memory
└── ConversationMemory — sliding window with async LLM summarization, lineage tracking

LLM Client
└── LiteLLMClient     — model pool rotation, rate-limit retry, async streaming

API
├── POST /ingest/              — trigger background ingestion job
├── GET  /ingest/status/{id}   — poll job status and summary
├── POST /query/               — structured response with sources
├── POST /query/stream         — token-level streaming response
├── POST /query/reset          — clear conversation memory
└── GET  /query/debug/chunks   — paginated vector store inspection
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- A LLM API key
- Optional: GitHub token for private repo ingestion

### Install

```bash
git clone https://github.com/Nagzz4vr/Repo-Query.git
cd Repo-Query
pip install -r requirements.txt
```

### Environment

```bash
export LLM_API_KEY=your_groq_api_key
export GITHUB_TOKEN=your_github_token   # optional, raises rate limit to 5000/hr
```

### Run the API server

```bash
uvicorn Api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run the Streamlit UI

```bash
streamlit run App/streamlit_app.py
```

The UI connects to `http://localhost:8000` by default. Override with the `RAG_API_URL` environment variable.

---

## Usage

### Ingest a GitHub repository

```bash
curl -X POST http://localhost:8000/ingest/ \
  -H "Content-Type: application/json" \
  -d '{
    "github_repo": "https://github.com/Nagzz4vr/Repo-Query",
    "github_token": "ghp_..."
  }'
```

Response:
```json
{ "job_id": "3f2a...", "status": "running" }
```

### Poll job status

```bash
curl http://localhost:8000/ingest/status/3f2a...
```

### Query

```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{ "question": "How does the AST chunker extract method symbols?" }'
```

### Streaming query

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{ "question": "Explain the RRF fusion logic in the vector store." }'
```

---

## Configuration

Key configuration points are constructor arguments on each component. Notable defaults:

| Component | Parameter | Default |
|---|---|---|
| `Embedder` | `model_name` | `BAAI/bge-m3` |
| `WritePipeline` | `embedding_dim` | `1024` |
| `VectorStore` | `index_type` | `flat_ip` |
| `ChunkPolicyEngine` | `min_lines` | `4` |
| `TextChunker` | `strategy` | `recursive` |
| `TextChunker` | `recursive_chunk_size` | `1200` |
| `QueryRewriter` | `strategy` | `multi_query` |
| `QueryRewriter` | `n_variants` | `3` |
| `QueryConfig` | `top_k_retrieval` | `10` |
| `QueryConfig` | `top_k_final` | `5` |
| `ConversationMemory` | `max_verbatim_tokens` | `4000` |
| `LiteLLMClient` | `model_pool` | `["groq/llama-3.3-70b-versatile"]` |

---

## Supported File Types

Code files are AST-chunked (Python) or text-chunked (all others):

`py` `js` `ts` `jsx` `tsx` `java` `go` `rb` `rs` `cpp` `c` `cs` `php` `swift` `kt` `scala` `sh` `lua` `r` `sql` `html` `css` `yaml` `json` `toml` `xml` `md`

Files inside `.git`, `__pycache__`, `node_modules`, `.venv`, `dist`, `build` are skipped automatically. Files over 500 KB are skipped.

---

## Project Structure

```
├── Api/                  FastAPI app, routes, schemas
├── App/                  Streamlit chat interface
├── Chunker/              AST chunker, text chunker, shared models
├── Embedder/             SentenceTransformer embedding pipeline
├── Ingestion/            Code, PDF, and web ingestors
├── LLM/                  LiteLLM async client with model rotation
├── Memory/               Conversation memory and context compressor
├── Pipelines/            Ingestion, chunking, and write pipeline orchestrators
├── Query/                Query engine, rewriter, reranker
├── Tracker/              Token ledger and trace logger (JSONL)
└── Vector_Store/         FAISS + BM25 hybrid vector store
```

---

## Token Tracking

Every LLM call is recorded to `token_ledger/<session_id>.jsonl`. Each entry includes `request_id`, `model`, `prompt_tokens`, `completion_tokens`, `cost`, `latency_ms`, `retry_count`, and a deterministic `event_id` (SHA-256 of the entry contents).

---

## License

See [LICENSE.txt](LICENSE.txt).
