"""HTTP surface for the voice RAG pipeline -- pure-Python startup surface.

Startup is 100% pure Python: no top-level imports of numpy, onnxruntime, bm25s, or hnswlib.
FastAPI binds to 0.0.0.0:$PORT immediately, enabling /diag/health to respond in <1ms.
Native binary dependencies are lazily loaded inside methods or tested via subprocess isolation.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

print("STARTUP_PHASE_1: Initializing pure Python FastAPI surface", flush=True)
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

print("STARTUP_PHASE_2: Defining pure Python schema models and constants", flush=True)
from core.guardrails import Decision
from core.stt import SarvamSTT

DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data" if Path("/data").is_dir() else "./data"))
INDEX_TAG = os.getenv("INDEX_TAG", "full")
INDEX_ROOT = Path(os.getenv("INDEX_PATH", str(DATA_ROOT / "index"))) / INDEX_TAG
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

DEFAULT_ENSEMBLE = ["metadata_128"]
FULL_ENSEMBLE = ["fixed_256", "semantic_128", "metadata_128"]
ENGLISH_INDEX = "english_256"

SERVE = os.getenv("SERVE_ENSEMBLE", ",".join(DEFAULT_ENSEMBLE)).split(",")
COMPARE = FULL_ENSEMBLE

STATE: dict = {}


def _percentiles(values: list[float]) -> dict[str, float]:
    import numpy as np
    a = np.array(values, dtype=np.float32)
    if len(a) == 0:
        return {"p50": 0.0, "p70": 0.0, "p90": 0.0, "p99": 0.0, "p100": 0.0, "mean": 0.0}
    return {
        "p50": round(float(np.percentile(a, 50)), 2),
        "p70": round(float(np.percentile(a, 70)), 2),
        "p90": round(float(np.percentile(a, 90)), 2),
        "p99": round(float(np.percentile(a, 99)), 2),
        "p100": round(float(a.max()), 2),
        "mean": round(float(a.mean()), 2),
    }


def _maybe_download_index(index_root: Path) -> None:
    """Download and atomically install the BM25 index artifact if missing.

    Render Free has no persistent disk, so this runs on every cold start.
    It's idempotent (skips if .artifact_complete is already present) and
    safe to call unconditionally.
    """
    marker = index_root / ".artifact_complete"
    if marker.exists():
        print(f"INDEX_DOWNLOAD: already installed ({marker})", flush=True)
        return

    url = os.getenv("INDEX_DOWNLOAD_URL", "").strip()
    expected_sha = os.getenv("INDEX_SHA256", "").strip().lower()
    if not url:
        print("INDEX_DOWNLOAD: INDEX_DOWNLOAD_URL not set, skipping", flush=True)
        return
    if not expected_sha:
        print("INDEX_DOWNLOAD: INDEX_SHA256 not set, refusing to download without a checksum", flush=True)
        return

    import hashlib
    import shutil
    import tarfile
    import tempfile
    import urllib.request

    index_root.parent.mkdir(parents=True, exist_ok=True)
    index_root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"INDEX_DOWNLOAD: starting download from {url}", flush=True)

    with tempfile.TemporaryDirectory(dir=str(index_root.parent)) as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / "artifact.tar.gz"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pregunta-backend/1.0"})
            with urllib.request.urlopen(req, timeout=180) as resp, archive_path.open("wb") as f:
                shutil.copyfileobj(resp, f, length=1024 * 1024)
        except Exception as exc:
            print(f"INDEX_DOWNLOAD ERROR: download failed: {exc}", flush=True)
            return

        dt = round((time.perf_counter() - t0) * 1000, 1)
        size_mb = round(archive_path.stat().st_size / (1024 * 1024), 2)
        print(f"INDEX_DOWNLOAD: downloaded {size_mb}MB in {dt}ms", flush=True)

        sha256 = hashlib.sha256()
        with archive_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha256.update(chunk)
        actual_sha = sha256.hexdigest()
        if actual_sha != expected_sha:
            print(
                f"INDEX_DOWNLOAD ERROR: SHA-256 mismatch "
                f"(expected {expected_sha}, got {actual_sha}); aborting",
                flush=True,
            )
            return
        print("INDEX_DOWNLOAD: SHA-256 verified", flush=True)

        stage_dir = tmp_path / "stage"
        stage_dir.mkdir()
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                for member in tar.getmembers():
                    member_path = (stage_dir / member.name).resolve()
                    if not str(member_path).startswith(str(stage_dir.resolve())):
                        raise ValueError(f"unsafe path in archive: {member.name}")
                tar.extractall(stage_dir)
        except Exception as exc:
            print(f"INDEX_DOWNLOAD ERROR: extraction failed: {exc}", flush=True)
            return

        extracted_dirs = [p for p in stage_dir.iterdir() if p.is_dir()]
        if not extracted_dirs:
            print("INDEX_DOWNLOAD ERROR: archive contained no strategy directory", flush=True)
            return

        installed_any = False
        for strategy_dir in extracted_dirs:
            if not (strategy_dir / "meta.pkl").exists():
                print(f"INDEX_DOWNLOAD ERROR: {strategy_dir.name} missing meta.pkl, skipping", flush=True)
                continue
            dest = index_root / strategy_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(strategy_dir), str(dest))
            print(f"INDEX_DOWNLOAD: installed {strategy_dir.name} -> {dest}", flush=True)
            installed_any = True

        if not installed_any:
            print("INDEX_DOWNLOAD ERROR: nothing valid installed, not writing marker", flush=True)
            return

        marker.write_text("ok\n")
        total_dt = round((time.perf_counter() - t0) * 1000, 1)
        print(f"INDEX_DOWNLOAD: complete in {total_dt}ms, marker written", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    print("LIFESPAN: START", flush=True)

    _maybe_download_index(INDEX_ROOT)

    enable_dense = os.getenv("ENABLE_DENSE_RETRIEVAL", "false").lower() == "true"
    embedder = None

    if enable_dense:
        print("LIFESPAN: DENSE_RETRIEVAL_ENABLED - Creating Embedder", flush=True)
        try:
            from core.embedder import Embedder, EmbedderConfig
            threads = int(os.getenv("ORT_THREADS", "0"))
            embedder = Embedder(EmbedderConfig(threads=threads))
        except Exception as exc:
            print(f"LIFESPAN WARNING: Embedder creation failed: {exc}", flush=True)
    else:
        print("LIFESPAN: EMBEDDER_DISABLED", flush=True)

    indexes: dict[str, ChunkIndex] = {}
    if INDEX_ROOT.exists():
        print("LIFESPAN: LOADING_DIRECT_INDEX", flush=True)
        from core.index import ChunkIndex
        for name in [*COMPARE, ENGLISH_INDEX]:
            if (INDEX_ROOT / name / "meta.pkl").exists() or (INDEX_ROOT / name / "hnsw.bin").exists():
                try:
                    indexes[name] = ChunkIndex.load(INDEX_ROOT, name, load_hnsw=enable_dense)
                except Exception as exc:
                    print(f"LIFESPAN WARNING: Index {name} load failed: {exc}", flush=True)
        print("LIFESPAN: DIRECT_INDEX_LOADED", flush=True)

    from core.harness import RAGHarness
    from core.retriever import AdaptiveRetriever

    serve_names = [n for n in SERVE if n in indexes] or (list(indexes)[:1] if indexes else [])
    english = (
        AdaptiveRetriever({ENGLISH_INDEX: indexes[ENGLISH_INDEX]})
        if ENGLISH_INDEX in indexes else None
    )

    harness = None
    if indexes:
        harness = RAGHarness(
            INDEX_ROOT,
            embedder=embedder,
            retriever=AdaptiveRetriever({n: indexes[n] for n in serve_names}),
            context_passages=int(os.getenv("CONTEXT_PASSAGES", "4")),
            english_retriever=english,
        )

    STATE.update(
        embedder=embedder,
        indexes=indexes,
        harness=harness,
        serve_names=serve_names,
        stt=SarvamSTT(),
        sample_queries=[],
        started_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    print("LIFESPAN: READY", flush=True)
    print(
        f"ready in {STATE['started_ms']}ms | serving {serve_names} | "
        f"{sum(len(i) for i in indexes.values()):,} chunks across {len(indexes)} indexes",
        flush=True
    )
    yield
    STATE.get("stt") and STATE["stt"].close()
    STATE.clear()


print("STARTUP_PHASE_3: Creating FastAPI application instance", flush=True)
app = FastAPI(title="Voice RAG — HH Goa Task 2", version="1.0", lifespan=lifespan)
print("FASTAPI_APP_CREATED: Application instance initialized", flush=True)


@app.get("/diag/health")
def diag_health():
    indexes = STATE.get("indexes")
    embedder = STATE.get("embedder")
    embed_avail = embedder.is_available() if embedder is not None else False
    enable_dense = os.getenv("ENABLE_DENSE_RETRIEVAL", "false").lower() == "true"
    return {
        "status": "ok",
        "pure_python": True,
        "retrieval": "available" if indexes else "unavailable",
        "embedding_runtime": "available" if embed_avail else ("unavailable" if embedder else "not_initialized"),
        "dense_retrieval_enabled": enable_dense,
        "index": "available" if indexes else "unavailable",
        "llm_required": False
    }


@app.get("/diag/memory")
def diag_memory():
    pid = os.getpid()
    rss_mb = 0.0

    try:
        if os.path.exists("/proc/self/statm"):
            with open("/proc/self/statm", "r") as f:
                parts = f.read().split()
                if len(parts) >= 2:
                    pages = int(parts[1])
                    rss_mb = round((pages * 4096) / (1024 * 1024), 2)
    except Exception:
        pass

    embedder = STATE.get("embedder")
    embed_avail = embedder.is_available() if embedder is not None else False
    enable_dense = os.getenv("ENABLE_DENSE_RETRIEVAL", "false").lower() == "true"
    mode = os.getenv("ANSWER_MODE", "direct").lower()

    return {
        "status": "ok",
        "pid": pid,
        "rss_mb": rss_mb,
        "answer_mode": mode,
        "embedder_initialized": embed_avail,
        "dense_retrieval_enabled": enable_dense,
        "indexes_loaded": len(STATE.get("indexes", {})),
    }


@app.get("/diag/imports")
def diag_imports():
    import subprocess

    packages = ["onnxruntime", "hnswlib", "numpy", "bm25s", "tokenizers", "pyarrow", "pystemmer"]
    results = []

    for pkg in packages:
        code = f"import {pkg}; print('{pkg}_OK', flush=True)"
        if pkg == "pystemmer":
            code = "import Stemmer; print('pystemmer_OK', flush=True)"

        try:
            t0 = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=5,
            )
            dt_ms = round((time.perf_counter() - t0) * 1000, 2)
            safe = proc.returncode == 0 and f"{pkg}_OK" in proc.stdout
            results.append({
                "package": pkg,
                "safe": safe,
                "returncode": proc.returncode,
                "signal": -proc.returncode if proc.returncode < 0 else (132 if proc.returncode == 132 else 0),
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "took_ms": dt_ms
            })
        except Exception as exc:
            results.append({
                "package": pkg,
                "safe": False,
                "error": str(exc)
            })

    return {"status": "ok", "package_diagnostics": results}


@app.get("/diag/embedding")
def diag_embedding():
    embedder = STATE.get("embedder")
    if embedder is None:
        return {"available": False, "status": "not_initialized"}
    avail = embedder.is_available()
    return {
        "available": avail,
        "provider": getattr(embedder, "provider", None),
        "variant": getattr(embedder, "variant", None),
        "status": "available" if avail else "unavailable"
    }


@app.get("/diag/ort")
def diag_ort():
    res = {
        "python_version": sys.version,
        "configured_provider": os.getenv("ORT_PROVIDERS"),
        "e5_variant": os.getenv("E5_VARIANT"),
        "ort_version": None,
        "ort_import_ok": False,
        "session_created": False,
        "inference_ok": False,
        "error": None
    }
    try:
        print("DIAG: PRE_ORT_IMPORT", flush=True)
        import onnxruntime as ort
        print("DIAG: POST_ORT_IMPORT", flush=True)
        res["ort_version"] = getattr(ort, "__version__", "unknown")
        res["ort_import_ok"] = True

        print("DIAG: PRE_SESSION_CREATE", flush=True)
        from core.embedder import Embedder, EmbedderConfig
        threads = int(os.getenv("ORT_THREADS", "0"))
        e = Embedder(EmbedderConfig(threads=threads))
        print("DIAG: POST_SESSION_CREATE", flush=True)
        res["session_created"] = True
        res["selected_provider"] = e.provider
        res["selected_variant"] = e.variant

        print("DIAG: PRE_INFERENCE", flush=True)
        vec = e.encode_query("test query")
        print("DIAG: POST_INFERENCE", flush=True)
        res["inference_ok"] = True
        res["vector_shape"] = list(vec.shape)
    except Exception as exc:
        print(f"DIAG_ERROR: {exc}", flush=True)
        res["error"] = str(exc)
    return res


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    code = "KNOWLEDGE_BASE_UNAVAILABLE" if exc.status_code == 503 else "HTTP_ERROR"
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": code,
                "message": str(exc.detail),
            }
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred while processing the request."
            }
        }
    )


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    generate: bool | None = None


def _answer_payload(r) -> dict:
    mode = os.getenv("ANSWER_MODE", "direct").lower()
    threshold = float(os.getenv("RETRIEVAL_THRESHOLD", os.getenv("MIN_GROUNDING_SCORE", "0.45")))
    grounded = r.answered and r.support >= threshold and r.decision != Decision.ABSTAIN_UNGROUNDED.value

    return {
        "success": True,
        "mode": mode,
        "query": r.question,
        "answer": r.answer or r.extractive_answer,
        "decision": r.decision,
        "reason": r.reason,
        "answer_source": r.answer_source,
        "extractive_answer": r.extractive_answer,
        "generated_answer": r.generated_answer if mode == "llm" else "",
        "route": r.route,
        "support": r.support,
        "grounding": r.grounding,
        "grounded": grounded,
        "threshold": threshold,
        "results": [
            {
                "source": s.unit_id,
                "passage_id": s.unit_id,
                "text": s.text,
                "score": s.score,
                "contributors": s.contributors,
            }
            for s in r.sources
        ],
        "citations": [
            {
                "source": s.unit_id,
                "passage_id": s.unit_id,
                "text": s.text,
                "score": s.score,
            }
            for s in r.sources[:2]
        ],
        "fast_path_ms": r.fast_path_ms,
        "total_ms": r.total_ms,
        "timings_ms": r.timings_ms,
    }


@app.post("/ask")
def ask(req: AskRequest):
    harness = STATE.get("harness")
    if harness is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base index is currently loading or unavailable."
        )
    gen_flag = req.generate
    if gen_flag is None:
        gen_flag = (os.getenv("ANSWER_MODE", "direct").lower() == "llm")
    r = harness.answer(req.question, generate=gen_flag)
    return _answer_payload(r)


@app.post("/voice")
async def voice(file: UploadFile = File(...), generate: bool | None = Form(None)):
    stt = STATE.get("stt")
    harness = STATE.get("harness")
    if not stt or not harness:
        raise HTTPException(
            status_code=503,
            detail="Voice service or knowledge base index is unavailable."
        )
    audio = await file.read()
    tr = stt.transcribe(audio, filename=file.filename or "audio.wav")
    if not tr.ok:
        raise HTTPException(status_code=400, detail=f"STT error: {tr.error}")

    gen_flag = generate
    if gen_flag is None:
        gen_flag = (os.getenv("ANSWER_MODE", "direct").lower() == "llm")

    res = harness.answer(tr.text, generate=gen_flag)
    payload = _answer_payload(res)
    payload["stt_ok"] = True
    payload["transcript"] = tr.text
    payload["stt_ms"] = tr.took_ms
    payload["stt_language"] = tr.language_code
    return payload


@app.post("/compare")
def compare(req: AskRequest):
    from core.extractive import extract_answer
    from core.retriever import AdaptiveRetriever

    indexes = STATE.get("indexes")
    embedder = STATE.get("embedder")
    if not indexes:
        raise HTTPException(status_code=503, detail="Indexes not loaded")

    embed_avail = embedder is not None and getattr(embedder, "is_available", lambda: False)()
    qvec = embedder.encode_query(req.question) if embed_avail else None

    out: dict[str, dict] = {}
    for name in COMPARE:
        if name not in indexes:
            continue
        ar = AdaptiveRetriever({name: indexes[name]})
        t0 = time.perf_counter()
        if qvec is not None:
            res = ar.search(qvec, req.question, k=5)
            ext = extract_answer(req.question, qvec, res.hits, embedder)
        else:
            res = ar.search_sparse(req.question, k=5)
            ext = extract_answer(req.question, None, res.hits, None)
        dt = round((time.perf_counter() - t0) * 1000, 2)
        out[name] = {
            "extractive_answer": ext.text,
            "support": round(ext.support, 4),
            "hits": [
                {"unit_id": h.unit_id, "score": round(h.score, 4), "text": h.text}
                for h in res.hits[:3]
            ],
            "ms": dt,
        }
    return {"query": req.question, "strategies": out}


@app.get("/health")
def health():
    indexes = STATE.get("indexes", {})
    embedder = STATE.get("embedder")
    stt = STATE.get("stt")
    embed_avail = embedder.is_available() if embedder is not None else False
    serve_names = STATE.get("serve_names", [])
    serving_chunks = sum(len(indexes[n]) for n in serve_names if n in indexes)
    return {
        "status": "ready" if indexes else "unready",
        "serving": serve_names,
        "n_indexes": len(indexes),
        "total_chunks": serving_chunks,
        "embedding_runtime": "available" if embed_avail else "unavailable",
        "started_ms": STATE.get("started_ms", 0.0),
        "stt_configured": bool(stt and stt.configured),
        "mode": os.getenv("ANSWER_MODE", "direct").lower(),
        "embedder_variant": os.getenv("E5_VARIANT", "multilingual-e5-small"),
        "index_tag": INDEX_TAG,
    }


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
