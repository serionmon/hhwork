"""Build a sparse-only English BM25 index from text_eng already on disk.

MSMARCO-XI ships every passage twice: the original English (English_passages,
stored as text_eng in raw/) and the Indic translation. Only the translation was
ever indexed, so an English question falls back to the Devanagari index and gets
random phonetic BM25 false positives.

This script builds english_256/ with:
  - meta.pkl   (chunk_ids, passage_ids, texts, langs, query_types)
  - bm25/      (BM25 corpus)
  - info.json  (n_chunks)

No HNSW is built (no embedder needed, no GPU, takes ~30s not hours).
When load_hnsw=False, ChunkIndex.load() uses BM25-only retrieval, which is
exactly what the harness calls via search_sparse() when no embedder is
available -- which is the current deployment mode.

Usage:
    python -m ingest.build_english_bm25
    python -m ingest.build_english_bm25 --tag full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data" if Path("/data").is_dir() else "./data"))
VARIANT = "english_256"


def load_english_passages() -> tuple[list[str], list[str], list[str], list[str]]:
    """Return (chunk_ids, passage_ids, texts, query_types) from text_eng, deduped."""
    seen: set[bytes] = set()
    chunk_ids: list[str] = []
    passage_ids: list[str] = []
    texts: list[str] = []
    query_types: list[str] = []

    for lang in ("hin", "mar"):
        p = DATA_ROOT / "raw" / f"{lang}_train_passages.jsonl"
        if not p.exists():
            print(f"  skip {lang}: {p} not found")
            continue
        print(f"  reading {p}...")
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = (r.get("text_eng") or "").strip()
                if not text:
                    continue
                # Dedup by content hash -- English source is shared across hin/mar shards.
                h = hashlib.blake2b(text.encode(), digest_size=16).digest()
                if h in seen:
                    continue
                seen.add(h)
                pid = f"eng:{r['passage_id']}"
                chunk_ids.append(pid)
                passage_ids.append(pid)
                texts.append(text)
                query_types.append(r.get("query_type", "DESCRIPTION"))

    return chunk_ids, passage_ids, texts, query_types


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    t0 = time.perf_counter()
    dest = DATA_ROOT / "index" / args.tag / VARIANT

    # Skip if already built
    if (dest / "meta.pkl").exists() and (dest / "bm25").exists():
        info = dest / "info.json"
        n = json.loads(info.read_text()).get("n_chunks", 0) if info.exists() else 0
        print(f"english_256 already built ({n:,} chunks). Delete {dest} to rebuild.")
        return

    print(f"Loading English passages from {DATA_ROOT / 'raw'} ...")
    chunk_ids, passage_ids, texts, query_types = load_english_passages()
    print(f"  {len(texts):,} unique English passages")

    # Build BM25 index.
    print("Building BM25 index...")
    import bm25s
    from core.text import BM25_TOKEN_PATTERN

    tokens = bm25s.tokenize(
        texts, stopwords=None, show_progress=False, token_pattern=BM25_TOKEN_PATTERN
    )
    bm25 = bm25s.BM25()
    bm25.index(tokens, show_progress=False)
    print(f"  BM25 indexed {len(texts):,} passages")

    # Save.
    dest.mkdir(parents=True, exist_ok=True)
    bm25.save(str(dest / "bm25"))

    meta = {
        "strategy": VARIANT,
        "chunk_ids": chunk_ids,
        "passage_ids": passage_ids,
        "texts": texts,
        "query_types": query_types,
        "langs": ["eng_Latn"] * len(texts),
    }
    with (dest / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    info = {"strategy": VARIANT, "n_chunks": len(texts)}
    (dest / "info.json").write_text(json.dumps(info, indent=2))

    dt = time.perf_counter() - t0
    print(f"\nWrote {dest}  ({len(texts):,} chunks)  in {dt:.1f}s")


if __name__ == "__main__":
    main()
