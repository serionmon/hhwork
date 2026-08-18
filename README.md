# Pregunta — Multilingual Voice RAG

A voice-enabled, multilingual retrieval-augmented generation (RAG) system over
the [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) parallel
corpus (Hindi + Marathi). Ask questions by voice or by text and receive grounded,
extractive answers backed by real retrieved passages.

**Live demo:** https://pregunta-backend.onrender.com

---

## Features

- **Text and voice question answering** — type a question or speak it; the same
  retrieval pipeline handles both.
- **Multilingual corpus** — 203,757 indexed chunks from Hindi (`hin_Deva`) and
  Marathi (`mar_Deva`) passages drawn from MSMARCO-XI.
- **English language routing** — Latin-script (English) queries are automatically
  routed to a separate English BM25 index (`english_256`, 98,812 passages), so
  English questions reach English text.
- **Hybrid retrieval** — BM25 sparse retrieval (always on) combined with HNSW
  dense retrieval via `multilingual-e5-small` (optional, off by default in
  production to stay within Render Free RAM limits).
- **Extractive answering** — the fast path picks the best-supported sentence span
  from retrieved chunks with no LLM call; P50 latency is ~30 ms on aarch64.
- **Optional LLM rewrite** — when `ANSWER_MODE=llm` and an LLM key is configured,
  the extractive answer is rewritten and verified for grounding before being served.
- **Dual guardrails** — an intent check before retrieval (unsafe input) and a
  grounding check after extraction (unsupported answer).
- **Sarvam STT** — speech-to-text via the Sarvam `saaras:v3` model; supports
  Hindi, Marathi, and English-Indian speech, up to 30 seconds of audio.
- **Live benchmark UI** — a browser-side telemetry strip runs 100 live queries
  and displays P50/P70/P100 latency when the `/benchmark` API endpoint is
  available (see [Known Limitations](#known-limitations)).
- **Ablation panel** — the UI can compare all three chunking strategies
  (`fixed_256`, `semantic_128`, `metadata_128`) on the current question via
  `POST /compare`.

---

## Architecture

```
Browser (web/)
  │
  ├─ text input / mic recording (WAV, 16 kHz mono)
  │
  ▼
FastAPI server (api/main.py)
  │
  ├── POST /ask ────────────────────────────────► core/harness.py::RAGHarness.answer()
  │                                                    │
  ├── POST /voice ──► core/stt.py::SarvamSTT           │
  │                   (Sarvam API → transcript)         │
  │                         │                           │
  │                         └──────────────────────────►│
  │                                                    │
  │                                               1. check_input (guardrails)
  │                                               2. script routing:
  │                                                    Latin → english_256 (BM25)
  │                                                    Devanagari → metadata_128
  │                                               3. embed_query (multilingual-e5-small,
  │                                                    ONNX, optional)
  │                                               4. retrieve (BM25 sparse, ±HNSW dense)
  │                                               5. extract_answer (best span)
  │                                               6. check_output (grounding gate)
  │                                               7. [optional] LLM generate + verify
  │                                                    │
  └──────────────────────────────────────────────────◄─┘
                                          JSON response
```

**Data flow — voice path:**

```
Mic → MediaRecorder → OfflineAudioContext resample to 16 kHz → WAV blob
    → POST /voice (multipart file=<wav>)
    → SarvamSTT.transcribe() → transcript string
    → RAGHarness.answer(transcript) → same pipeline as /ask
    → JSON + stt_ok, transcript, stt_ms, stt_language
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI 0.115+ |
| ASGI server | Uvicorn |
| Data validation | Pydantic v2 |
| Sparse retrieval | bm25s 0.2+ |
| Dense retrieval | hnswlib 0.8+ (optional) |
| Embedder | multilingual-e5-small via ONNX Runtime 1.20+ |
| Tokenizer | HuggingFace `tokenizers` (XLM-R vocab) |
| Speech-to-text | Sarvam AI `saaras:v3` |
| LLM providers | AWS Bedrock, Google Gemini, Anthropic, OpenRouter, NVIDIA NIM (all optional) |
| Data format | Apache Arrow / Parquet (ingest only) |
| HTTP client | httpx |
| Python | ≥3.11, <3.13 |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Container | Docker (CPU: `Dockerfile`; GPU ingest: `Dockerfile.gpu`) |
| Frontend | Vanilla HTML + CSS + JavaScript (no framework) |

---

## Project Structure

```
.
├── api/
│   └── main.py              # FastAPI application, all HTTP routes
├── core/
│   ├── embedder.py          # multilingual-e5-small ONNX wrapper
│   ├── extractive.py        # span extraction and support scoring
│   ├── guardrails.py        # input/output safety checks
│   ├── harness.py           # RAGHarness: orchestrates the full pipeline
│   ├── index.py             # ChunkIndex: HNSW + BM25 per strategy
│   ├── retriever.py         # AdaptiveRetriever, RRF fusion, language routing
│   ├── stt.py               # Sarvam STT client
│   └── text.py              # Devanagari-safe tokenisation for BM25
├── ingest/
│   ├── chunkers.py          # fixed / sentence / semantic / metadata strategies
│   ├── download.py          # MSMARCO-XI parquet streaming
│   ├── pipeline.py          # end-to-end ingest: download → chunk → embed → index
│   ├── build_english.py     # builds english_256 with dense HNSW (needs GPU)
│   ├── build_english_bm25.py# builds english_256 BM25-only (~5 s, no GPU)
│   └── SCHEMA.md            # documented corpus schema
├── bench/
│   ├── fastpath.py          # end-to-end fast-path latency benchmark
│   ├── latency.py           # single ChunkIndex latency
│   ├── profile_extract.py   # extractive answer profiling
│   └── tune_extract.py      # extractive parameter tuning
├── eval/
│   ├── evaluate.py          # MRR@10, R@10, R@20 over labelled queries
│   ├── ablate.py / ablate_full.py  # chunking strategy ablation
│   ├── significance.py      # paired bootstrap confidence intervals
│   └── behaviour.py         # qualitative answer behaviour analysis
├── tests/
│   └── test_direct_retrieval.py  # 12 pytest tests
├── web/
│   ├── index.html           # single-page UI
│   ├── main.js              # API calls, voice recording, telemetry
│   └── styles.css           # visual design
├── data/
│   ├── raw/                 # downloaded JSONL corpus (gitignored)
│   ├── index/full/          # built indexes (gitignored)
│   │   ├── metadata_128/    # served Indic index — 203,757 chunks
│   │   └── english_256/     # English BM25 index — 98,812 chunks
│   └── reports/             # benchmark output JSON + Markdown (committed)
├── docs/
│   ├── BUILD_LOG.md         # chronological development log
│   ├── HANDOVER.md          # deployment and operational notes
│   └── GPU_SETUP.md         # GPU ingest setup
├── Dockerfile               # CPU serving image (multi-arch)
├── Dockerfile.gpu           # CUDA image for ingest only
├── docker-compose.yml       # local development and benchmark services
├── pyproject.toml           # dependencies (uv / pip)
└── .env.example             # all environment variables documented
```

---

## Backend API

All endpoints are served by `api/main.py` via uvicorn. The static frontend is
mounted at `/` from `web/`.

### `POST /ask`

Submit a text question and receive an extractive (or LLM-generated) answer.

**Request body (JSON):**
```json
{
  "question": "What is the capital of India?",
  "generate": null
}
```

| Field | Type | Description |
|---|---|---|
| `question` | `string` (1–1000 chars) | The question to answer |
| `generate` | `bool \| null` | `true` forces LLM generation; `null` follows `ANSWER_MODE` env var |

**Response (JSON):**
```json
{
  "success": true,
  "mode": "direct",
  "query": "What is the capital of India?",
  "answer": "The capital of India is New Delhi.",
  "decision": "allow",
  "reason": "",
  "answer_source": "extractive",
  "extractive_answer": "The capital of India is New Delhi.",
  "generated_answer": "",
  "route": "english",
  "support": 0.7234,
  "grounding": 0.8100,
  "grounded": true,
  "threshold": 0.45,
  "results": [{ "source": "eng:...", "text": "...", "score": 0.016 }],
  "citations": [...],
  "fast_path_ms": 32.1,
  "total_ms": 32.1,
  "timings_ms": { "guardrail_in": 0.01, "retrieve_sparse": 2.3, ... }
}
```

`answer_source` is one of: `extractive`, `generated`, `abstain`, `refusal`, `greeting`.

---

### `POST /voice`

Submit a WAV audio file; returns transcript plus the same answer payload as `/ask`.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | `UploadFile` | WAV audio (16 kHz mono recommended, ≤30 s) |
| `generate` | `bool \| null` | Optional form field; same as `/ask` |

**Response (JSON):**

All fields from `/ask`, plus:

| Field | Type | Description |
|---|---|---|
| `stt_ok` | `bool` | `true` if transcription succeeded |
| `transcript` | `string` | The transcribed text |
| `stt_ms` | `float` | STT round-trip time in milliseconds |
| `stt_language` | `string` | Detected language code (e.g. `hi-IN`) |

Returns `HTTP 400` if STT fails; `HTTP 503` if STT or index is unavailable.

---

### `POST /compare`

Run the current question against all three Indic chunking strategies
(`fixed_256`, `semantic_128`, `metadata_128`) and return per-strategy hits and
support scores. Used by the ablation panel in the UI.

**Request body:** same as `/ask` (`question` required).

**Response:** `{ "query": "...", "strategies": { "<name>": { "extractive_answer", "support", "hits", "ms" } } }`

---

### `GET /health`

Readiness probe and status summary.

**Response:**
```json
{
  "status": "ready",
  "serving": ["metadata_128"],
  "n_indexes": 2,
  "total_chunks": 203757,
  "embedding_runtime": "unavailable",
  "started_ms": 1842.3,
  "stt_configured": true,
  "mode": "direct",
  "embedder_variant": "int8_arm",
  "index_tag": "full"
}
```

`total_chunks` reflects only the **serving** index (not all loaded indexes).
`embedding_runtime` is `"unavailable"` when `ENABLE_DENSE_RETRIEVAL=false` (the default).

---

### `GET /diag/health`

Lightweight diagnostic; responds before the index finishes loading.

```json
{
  "status": "ok",
  "pure_python": true,
  "retrieval": "available",
  "embedding_runtime": "unavailable",
  "dense_retrieval_enabled": false,
  "index": "available",
  "llm_required": false
}
```

---

### `GET /diag/memory`

Process memory and runtime configuration.

```json
{
  "status": "ok",
  "pid": 1,
  "rss_mb": 412.0,
  "answer_mode": "direct",
  "embedder_initialized": false,
  "dense_retrieval_enabled": false,
  "indexes_loaded": 2
}
```

---

### `GET /diag/imports`

Tests that each native binary dependency can be imported in an isolated subprocess
(prevents `SIGILL` / `Illegal instruction` from crashing the main process on hosts
without the required CPU instructions).

Returns per-package `safe: bool`, return code, stdout/stderr, and latency.

---

### `GET /diag/embedding`

Reports whether the ONNX embedder initialised successfully and which provider and
model variant were selected.

---

### `GET /diag/ort`

Full ONNX Runtime diagnostic: version, selected provider, variant, and a live
inference test. Useful for debugging `SIGILL` or provider fall-through issues.

---

> **Note:** The frontend calls `GET /benchmark?n=100` to populate the live
> telemetry strip. This endpoint is **not implemented** in the current backend.
> Clicking "▶ Run 100 Live Test" will fail with a network error in the UI.
> Use `python -m bench.fastpath` locally instead (see [Benchmarking](#benchmarking)).

---

## Voice / STT

Audio is recorded in the browser using the `MediaRecorder` API, resampled to
16 kHz mono WAV via `OfflineAudioContext`, then sent to `POST /voice` as a
`multipart/form-data` upload.

The backend uses `core/stt.py::SarvamSTT`:

- **Endpoint:** `https://api.sarvam.ai/speech-to-text`
- **Model:** `saaras:v3` (replaces the deprecated `saarika:v2.5`)
- **Mode:** `transcribe` — preserves the source language rather than translating
- **Auth:** `api-subscription-key: <SARVAM_API_KEY>` header
- **Limit:** 30 seconds of audio; longer recordings are rejected
- **Retry:** up to 2 retries with backoff on `5xx` and `429` responses; `4xx`
  (bad key, unsupported format) are not retried

Supported language codes: `hi-IN`, `mr-IN`, `en-IN`, `bn-IN`, `kn-IN`,
`ml-IN`, `od-IN`, `pa-IN`, `ta-IN`, `te-IN`, `gu-IN`.

If `SARVAM_API_KEY` is not set, the microphone button is disabled in the UI and
`stt_configured: false` is returned by `/health`.

---

## Retrieval / Knowledge Base

### Corpus

Source: `ai4bharat/MSMARCO-XI` (HuggingFace), Hindi and Marathi train shards.
Each shard contains parallel English + translated passages with gold relevance
labels (`is_selected`).

Raw data is stored in `data/raw/{hin,mar}_train_passages.jsonl` after ingestion.

### Indexes

| Index | Language | Chunks | BM25 | HNSW |
|---|---|---|---|---|
| `metadata_128` | Hindi + Marathi (`text_translated`) | 203,757 | ✅ | Optional |
| `english_256` | English (`text_eng`, deduped) | 98,812 | ✅ | Not built |

`metadata_128` is the **served** index by default (`SERVE_ENSEMBLE=metadata_128`).
`english_256` is loaded alongside it for English-query routing.

> **Dense retrieval is disabled by default** (`ENABLE_DENSE_RETRIEVAL=false`) to
> stay within Render Free's 512 MB RAM limit. The BM25 sparse path is always active.

### Chunking strategies (used during ingest)

Four strategies are defined in `ingest/chunkers.py` and measured in `eval/`:

| Name | Strategy | Max tokens |
|---|---|---|
| `fixed_128` | Fixed token window | 128 |
| `fixed_256` | Fixed token window | 256 |
| `sentence_128` | Whole-sentence packing | 128 |
| `semantic_128` | Similarity-based split | 128 |
| `metadata_128` | Passage-native + query-type tag | 128 |

Only `metadata_128` is shipped and served. An ablation over 1,500 queries showed
the three-index ensemble is not significantly better than `metadata_128` alone on
MRR@10 or R@10, while costing 2.8× the memory and 2.6× the search time (see
`data/reports/full_significance.md`).

### Query routing

`core/retriever.py::is_latin_query()` classifies the input script:

- ≥90% Latin characters → routed to `english_256` (English BM25)
- otherwise → routed to `metadata_128` (Indic BM25)

### Retrieval pipeline

1. **Input guardrail** — regex patterns check for unsafe intent (credentials,
   self-harm, weapon synthesis) in both English and Hindi/Marathi.
2. **Embedding** (if `ENABLE_DENSE_RETRIEVAL=true`) — `multilingual-e5-small`
   encodes the query with the mandatory `"query: "` prefix.
3. **Retrieval** — BM25 sparse search (default) or HNSW+BM25 hybrid with
   Reciprocal Rank Fusion if dense is enabled.
4. **Extractive answer** — the best-supported 1–2 sentence span is selected by
   blending cosine similarity (75%) and lexical overlap (25%).
5. **Output guardrail** — grounding gate: if `support < RETRIEVAL_THRESHOLD`
   (default 0.45), the system abstains rather than returning a low-confidence span.
6. **LLM generation** (optional) — when `ANSWER_MODE=llm`, the extractive span is
   rewritten; the result is verified against the retrieved context and kept only if
   it passes the grounding check.

### Tokenisation

`core/text.py` defines a Devanagari-safe tokeniser used by BM25. It splits on
whitespace and separators (including the Hindi danda `।` and double danda `॥`)
rather than on character class, preserving whole words in any script.

---

## Measured Performance

From `data/reports/full_final_metadata_128_latency.md` (300 queries, aarch64,
`metadata_128` only, dense retrieval enabled, `multilingual-e5-small int8_arm`):

| Stage | P50 | P70 | P100 |
|---|---:|---:|---:|
| guardrail_in | 0.008 ms | 0.009 ms | 9.861 ms |
| embed_query | 1.749 ms | 1.954 ms | 86.903 ms |
| retrieve | 2.282 ms | 2.611 ms | 10.254 ms |
| extract | 26.214 ms | 28.509 ms | 44.757 ms |
| guardrail_out | 0.107 ms | 0.118 ms | 1.190 ms |
| **fast_path_total** | **30.5 ms** | **32.9 ms** | **129.5 ms** |

300/300 queries under the 200 ms budget (dense retrieval enabled on aarch64).
STT and LLM generation are excluded from these measurements.

> These numbers were measured during development on an Oracle aarch64 instance
> with dense retrieval enabled. The production Render deployment runs BM25-only
> (`ENABLE_DENSE_RETRIEVAL=false`), which skips the embedding step entirely.

---

## Diagnostics

```bash
# Readiness (use as health check)
curl https://pregunta-backend.onrender.com/health

# Lightweight check (responds before index loads)
curl https://pregunta-backend.onrender.com/diag/health

# Process memory and config
curl https://pregunta-backend.onrender.com/diag/memory

# Dependency import check (subprocess isolation)
curl https://pregunta-backend.onrender.com/diag/imports

# ONNX Runtime / embedder status
curl https://pregunta-backend.onrender.com/diag/ort
curl https://pregunta-backend.onrender.com/diag/embedding
```

---

## Configuration

Copy `.env.example` to `.env` and set the values you need. The application reads
these at runtime via `os.getenv()`.

### Required

| Variable | Description |
|---|---|
| *(none strictly required)* | The server starts without any key. Voice is disabled and answers are text-only. |

### Optional — Voice (STT)

| Variable | Description |
|---|---|
| `SARVAM_API_KEY` | Sarvam AI key. Without it the `/voice` endpoint returns 503. |

### Optional — LLM Generation

Set `ANSWER_MODE=llm` and one of:

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `gemini`, `anthropic`, `openrouter`, `bedrock`, `nvidia` |
| `GEMINI_API_KEY` + `GEMINI_MODEL` | Google Gemini |
| `ANTHROPIC_API_KEY` + `ANTHROPIC_MODEL` | Anthropic Claude |
| `OPENROUTER_API_KEY` + `OPENROUTER_MODEL` | OpenRouter |
| `NVIDIA_API_KEY` + `NVIDIA_MODEL` | NVIDIA NIM |
| `BEDROCK_MODEL` + `BEDROCK_REGION` | AWS Bedrock (uses credential chain, no explicit key) |
| `LLM_TIMEOUT_S` | Per-call timeout in seconds (default: 30) |

### Optional — Runtime / Index

| Variable | Default | Description |
|---|---|---|
| `ANSWER_MODE` | `direct` | `direct` (BM25 extractive only) or `llm` |
| `PORT` | `8000` | Uvicorn listen port |
| `DATA_ROOT` | `/data` or `./data` | Root of the data directory |
| `INDEX_PATH` | `$DATA_ROOT/index` | Parent of index tag directories |
| `INDEX_TAG` | `full` | Which index build to serve |
| `SERVE_ENSEMBLE` | `metadata_128` | Comma-separated list of indexes to serve |
| `ENABLE_DENSE_RETRIEVAL` | `false` | Set `true` to load HNSW and embed queries |
| `E5_VARIANT` | `int8_arm` | ONNX model variant: `fp32`, `int8_x86`, `int8_arm` |
| `ORT_PROVIDERS` | `CPUExecutionProvider` | ONNX Runtime execution provider(s) |
| `ORT_THREADS` | `0` (auto) | Thread count for ONNX Runtime inference |
| `RETRIEVAL_THRESHOLD` | `0.45` | Minimum support score; below this the system abstains |
| `CONTEXT_PASSAGES` | `4` | Number of retrieved passages passed to generation |
| `INDEX_DOWNLOAD_URL` | *(unset)* | HTTPS URL to a `.tar.gz` index artifact; downloaded on cold start if set |
| `INDEX_SHA256` | *(unset)* | SHA-256 hex of the artifact; required when `INDEX_DOWNLOAD_URL` is set |
| `HF_TOKEN` | *(unset)* | HuggingFace token for corpus download; lifts anonymous rate limit |

### Development-only

| Variable | Description |
|---|---|
| `AWS_PROFILE` | Named AWS profile for Bedrock |
| `AWS_DEFAULT_REGION` | AWS region for Bedrock |
| `INDEX_S3_URI` | S3 path for index storage (not used by the application at runtime) |

---

## Local Development

### Prerequisites

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
git clone https://github.com/serionmon/hhwork.git
cd hhwork

# Install dependencies (creates .venv automatically)
uv sync

# Copy and edit environment variables
cp .env.example .env
# Set SARVAM_API_KEY if you want voice, and a LLM key if you want generation.
```

### Running the backend

The backend requires a built index. See [Building the index](#building-the-index)
first if `data/index/full/` is empty.

```bash
# Start the API server (serves frontend from web/ automatically)
uv run uvicorn api.main:app --reload --port 8000
```

The frontend is served at `http://localhost:8000/`.

### Building the index

**Download the corpus first:**

```bash
uv run python -m ingest.pipeline --langs hin mar --max-queries 25000 --stream
```

This streams the parquet shards without storing them and runs chunking +
embedding + indexing. Requires ~1–4 hours (embedding dominates). Dense
embedding requires `ENABLE_DENSE_RETRIEVAL=true` and a compatible ONNX Runtime.

**Build the English BM25 index** (fast, no GPU, no embedding):

```bash
uv run python -m ingest.build_english_bm25
```

This reads `text_eng` from the already-downloaded `data/raw/` files and builds
`data/index/full/english_256/` in ~5 seconds. No GPU or ONNX Runtime needed.

### Running tests

```bash
uv run pytest tests/ -v
```

12 tests covering: answer mode defaults, direct retrieval, relevance thresholding,
API schema, error handling, diagnostic endpoints, sparse retrieval fallback,
`ChunkIndex` HNSW skip, and dense retrieval flag.

### Benchmarking

```bash
# End-to-end fast-path latency (300 queries, requires built index and ONNX)
uv run python -m bench.fastpath --tag full --n 300

# Single-index latency
uv run python -m bench.latency --tag full

# Chunking strategy ablation
uv run python -m eval.ablate_full --tag full
```

Results are written to `data/reports/`.

---

## Deployment

The live service runs on [Render](https://render.com) Free tier.

### Key constraints on Render Free

- **No persistent disk** — the index is re-downloaded on every cold start via
  `INDEX_DOWNLOAD_URL`. The download URL and `INDEX_SHA256` must be set in Render
  environment variables.
- **512 MB RAM** — dense retrieval (`ENABLE_DENSE_RETRIEVAL=true`) is disabled in
  production to avoid OOM. Only BM25 sparse retrieval is active.
- **Cold starts** — the first request after inactivity triggers index download
  (≥60 MB) + BM25 load. `GET /diag/health` responds immediately; `GET /health`
  returns `"status": "unready"` until index loading completes.

### Required Render environment variables

```
SARVAM_API_KEY=<key>
INDEX_DOWNLOAD_URL=<https://...tar.gz>
INDEX_SHA256=<sha256hex>
INDEX_TAG=full
SERVE_ENSEMBLE=metadata_128
ANSWER_MODE=direct
PORT=8000
```

### Index artifact

The deployment tarball must contain at least the `metadata_128/` strategy
directory (with `meta.pkl` and `bm25/`). To enable English routing, include
`english_256/` as well. The combined tarball can be generated locally:

```bash
python -c "
import tarfile
with tarfile.open('data/metadata_128_english_256_bm25.tar.gz', 'w:gz') as t:
    t.add('data/index/full/metadata_128', arcname='metadata_128')
    t.add('data/index/full/english_256',  arcname='english_256')
"
```

Upload this file to object storage (S3, R2, GCS, etc.) and set `INDEX_DOWNLOAD_URL`
to the public HTTPS URL, plus `INDEX_SHA256` to its SHA-256 hex digest.

### Docker (local or self-hosted)

```bash
# Build and start the API server
docker compose up api

# Run the ingest pipeline (streams corpus, no local parquet download)
docker compose run --rm ingest

# Run benchmarks against a local index
docker compose run --rm bench
```

The GPU image (`Dockerfile.gpu`) is for the ingest stage only and requires
CUDA 12.x + cuDNN 9. The serving container (`Dockerfile`) is CPU-only and
multi-arch (amd64 + aarch64).

---

## Testing

```
tests/
└── test_direct_retrieval.py   (12 tests)
```

| Test | What it covers |
|---|---|
| `test_answer_mode_defaults_to_direct` | `ANSWER_MODE` defaults to `direct` |
| `test_direct_retrieval_no_llm_required` | BM25 extractive path needs no LLM key |
| `test_relevance_threshold_no_result_behavior` | Low-support queries abstain cleanly |
| `test_api_structured_response_schema` | `/ask` response schema is correct |
| `test_api_no_result_structured_schema` | `/ask` response when nothing found |
| `test_empty_and_malformed_queries` | Empty/missing `question` → 422 |
| `test_unhandled_exception_returns_controlled_json` | Uncaught errors → 500 JSON |
| `test_diag_health_without_onnx` | `/diag/health` works without ONNX Runtime |
| `test_sparse_retrieval_fallback_without_onnx` | BM25 path works without embedder |
| `test_diag_memory_endpoint` | `/diag/memory` response structure |
| `test_dense_retrieval_disabled_by_default` | `ENABLE_DENSE_RETRIEVAL` defaults to `false` |
| `test_chunk_index_load_does_not_initialize_hnsw_by_default` | `load_hnsw=False` skips HNSW |

Run with:

```bash
uv run pytest tests/ -v
```

---

## Known Limitations

- **`/benchmark` endpoint is not implemented.** The UI button "▶ Run 100 Live Test"
  calls `GET /benchmark?n=100`, which is not present in `api/main.py`. It will
  always fail with a network error. Use `python -m bench.fastpath` locally.

- **Dense retrieval is off in production.** Render Free has 512 MB RAM. Loading
  the HNSW index plus the BM25 corpus exceeds this. Queries are answered by BM25
  sparse retrieval only (`ENABLE_DENSE_RETRIEVAL=false`).

- **English routing requires the `english_256` index.** If the deployment tarball
  does not include `english_256/`, English questions fall back to the Devanagari
  BM25 index and may return Hindi-language passages.

- **Index is not persistent on Render Free.** Every cold start re-downloads and
  re-loads the full index (~60–90 MB download + several seconds to deserialise).

- **Corpus is multilingual (Hindi/Marathi), not English-native.** The Indic index
  (`metadata_128`) contains `text_translated` — Hindi and Marathi passages. English
  questions are served from the separate `english_256` index when it is available.

- **Voice is limited to 30 seconds.** The Sarvam API rejects audio longer than
  30 seconds.

- **LLM generation is optional and not configured by default.** Without a LLM key,
  `ANSWER_MODE=direct` is enforced and only extractive answers are produced.

---

## Troubleshooting

### Backend does not start / index unavailable

```
"status": "unready"   # from GET /health
```

- Check that `INDEX_DOWNLOAD_URL` and `INDEX_SHA256` are set in the Render
  environment (or that `data/index/full/metadata_128/` exists locally).
- On Render, monitor logs for `INDEX_DOWNLOAD:` and `LIFESPAN: READY` lines.
- `GET /diag/health` responds immediately and shows `"retrieval": "unavailable"`
  until the index finishes loading.

### Voice button is greyed out / disabled

- `SARVAM_API_KEY` is not set; check `/health` → `stt_configured: false`.

### STT returns an error

- Audio over 30 seconds is rejected by the Sarvam API.
- Unsupported audio formats return a 4xx from Sarvam; the backend translates this
  to `HTTP 400`. The UI converts audio to 16 kHz mono WAV before sending.

### ONNX Runtime crash (`SIGILL` / `Illegal instruction`)

- The `int8_x86` variant uses AVX-512 VNNI instructions not available on all CPUs.
  Set `E5_VARIANT=int8_arm` (ARM) or `E5_VARIANT=fp32` (safe on any x86-64).
- `GET /diag/imports` tests each native package in a subprocess to catch this
  before it kills the main process.

### Dense retrieval not activating

- Set `ENABLE_DENSE_RETRIEVAL=true`. Verify with `GET /diag/embedding` →
  `"available": true`. On Render Free this will likely OOM; use locally or on a
  larger instance.

### "Knowledge base index is currently loading or unavailable" on `/ask`

- The index has not finished loading yet (cold start). Retry after a few seconds.
- If it persists, the index download may have failed (check `INDEX_SHA256` matches
  the actual file).

### `/benchmark` button always fails

- This endpoint is not implemented. Use `python -m bench.fastpath --tag full` locally.

---

## Security

- **API keys are read from environment variables** (`SARVAM_API_KEY`, LLM keys,
  `HF_TOKEN`). They are never committed to the repository. `.env` is in `.gitignore`.
- **Uploaded audio** is read into memory, transcribed, and discarded. No audio is
  written to disk.
- **Question length** is capped at 1000 characters by the Pydantic model and at
  512 characters inside the harness (the 512-char cap bounds embedding latency on
  pathologically long machine-translation artifacts).
- **Retrieved answers** are extractive spans from the indexed corpus. The system
  never invents facts; it abstains when the corpus does not cover the query.
- **Source link** in the UI opens `https://github.com/serionmon/hhwork` in a new
  tab — no proxying or credential forwarding.

---

## License

No license file is present in this repository.
