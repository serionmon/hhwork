"""Hybrid index: dense HNSW + sparse BM25, fused with Reciprocal Rank Fusion.

One of these is built per chunking strategy, so the eval can compare them on
identical queries and the router can pick between them at serve time.

Why hybrid: dense retrieval handles paraphrase and cross-lingual matching;
BM25 handles exact tokens -- numbers, names, transliterated entities -- which is
exactly where dense embeddings are weakest and where NUMERIC/PERSON/ENTITY
queries live. RRF fuses them without needing score calibration between the two.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

from core.text import BM25_TOKEN_PATTERN

DIM = 384


@dataclass(slots=True)
class Hit:
    chunk_id: str
    passage_id: str
    text: str
    score: float
    rank: int
    query_type: str
    lang: str
    source: str  # "dense" | "sparse" | "rrf"


class ChunkIndex:
    """Dense + sparse index over the chunks produced by one strategy."""

    def __init__(self, strategy: str):
        self.strategy = strategy
        self.hnsw: hnswlib.Index | None = None
        self.bm25: bm25s.BM25 | None = None
        # Row i in every array below corresponds to internal id i.
        self.chunk_ids: list[str] = []
        self.passage_ids: list[str] = []
        self.texts: list[str] = []
        self.query_types: list[str] = []
        self.langs: list[str] = []

    # -- build -------------------------------------------------------------

    def build(
        self,
        vectors: np.ndarray,
        chunks: list[dict],
        *,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 96,
        bm25_texts: list[str] | None = None,
        display_texts: list[str] | None = None,
    ) -> None:
        """Build both indexes. `vectors` must be L2-normalised, row-aligned to `chunks`.

        Three views of the same chunk, because they have different jobs:

          embedded   `c["text"]` -- what produced `vectors`. The metadata strategy
                     prepends a query-type hint here on purpose.
          bm25_texts lexical matching, where that hint would be noise.
          display_texts what the user is shown and what the extractive answer is
                     drawn from. The hint must not appear in an answer -- it is an
                     indexing device, not content.

        `display_texts` defaults to the embedded text, which is correct for every
        strategy that does not rewrite it.
        """
        import bm25s
        try:
            import hnswlib
        except ImportError:
            hnswlib = None
        import numpy as np

        n = len(chunks)
        if vectors.shape != (n, DIM):
            raise ValueError(f"vectors {vectors.shape} != ({n}, {DIM})")

        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.passage_ids = [c["passage_id"] for c in chunks]
        self.texts = display_texts if display_texts is not None else [c["text"] for c in chunks]
        self.query_types = [c["query_type"] for c in chunks]
        self.langs = [c["lang"] for c in chunks]

        if hnswlib is not None:
            self.hnsw = hnswlib.Index(space="ip", dim=DIM)
            self.hnsw.init_index(max_elements=n, ef_construction=ef_construction, M=M)
            self.hnsw.add_items(vectors, np.arange(n))
            self.hnsw.set_ef(ef_search)

        corpus = bm25_texts if bm25_texts is not None else self.texts
        tokens = bm25s.tokenize(
            corpus, stopwords=None, show_progress=False, token_pattern=BM25_TOKEN_PATTERN
        )
        self.bm25 = bm25s.BM25()
        self.bm25.index(tokens, show_progress=False)

    # -- search ------------------------------------------------------------

    def search_dense(self, qvec, k: int) -> list[tuple[int, float]]:
        try:
            import hnswlib
        except ImportError:
            return []
        if self.hnsw is None:
            return []
        labels, dists = self.hnsw.knn_query(qvec.reshape(1, -1), k=min(k, len(self.chunk_ids)))
        return [(int(i), 1.0 - float(d)) for i, d in zip(labels[0], dists[0], strict=True)]

    def search_sparse(self, query: str, k: int) -> list[tuple[int, float]]:
        import bm25s
        if self.bm25 is None:
            return []
        tokens = bm25s.tokenize(
            [query], stopwords=None, show_progress=False, token_pattern=BM25_TOKEN_PATTERN
        )
        idx, scores = self.bm25.retrieve(tokens, k=min(k, len(self.chunk_ids)), show_progress=False)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0], strict=True)]

    def search(
        self,
        qvec,
        query_text: str,
        k: int = 50,
        *,
        rrf_k: int = 60,
        dense_only: bool = False,
    ) -> list[Hit]:
        dense = self.search_dense(qvec, k)
        if dense_only:
            fused = {i: s for i, s in dense}
            ordered = sorted(fused.items(), key=lambda kv: -kv[1])
            source = "dense"
        else:
            sparse = self.search_sparse(query_text, k)
            scores: dict[int, float] = {}
            for rank, (i, _) in enumerate(dense):
                scores[i] = scores.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
            for rank, (i, _) in enumerate(sparse):
                scores[i] = scores.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
            ordered = sorted(scores.items(), key=lambda kv: -kv[1])
            source = "rrf"

        return [
            Hit(
                chunk_id=self.chunk_ids[i],
                passage_id=self.passage_ids[i],
                text=self.texts[i],
                score=float(s),
                rank=r,
                query_type=self.query_types[i],
                lang=self.langs[i],
                source=source,
            )
            for r, (i, s) in enumerate(ordered[:k])
        ]

    # -- persistence -------------------------------------------------------

    def save(self, root: Path) -> None:
        import bm25s
        d = root / self.strategy
        d.mkdir(parents=True, exist_ok=True)
        if self.hnsw is not None:
            self.hnsw.save_index(str(d / "hnsw.bin"))
        if self.bm25 is not None:
            self.bm25.save(str(d / "bm25"))
        with (d / "meta.pkl").open("wb") as f:
            pickle.dump(
                {
                    "strategy": self.strategy,
                    "chunk_ids": self.chunk_ids,
                    "passage_ids": self.passage_ids,
                    "texts": self.texts,
                    "query_types": self.query_types,
                    "langs": self.langs,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        (d / "info.json").write_text(
            json.dumps({"strategy": self.strategy, "n_chunks": len(self.chunk_ids)}, indent=2)
        )

    @classmethod
    def load(cls, root: Path, strategy: str, ef_search: int = 96, load_hnsw: bool = False) -> ChunkIndex:
        import bm25s

        hnswlib = None
        if load_hnsw:
            try:
                import hnswlib
            except ImportError:
                hnswlib = None
            except Exception as exc:
                # Some hnswlib wheels crash with SIGILL (illegal instruction)
                # on hosts whose CPU lacks the AVX512 features the wheel was
                # built with. That's a hard process kill, not a catchable
                # Python exception -- so this except only catches import-time
                # failures that *are* catchable (e.g. missing shared libs).
                print(f"[ChunkIndex] hnswlib unavailable, dense retrieval disabled: {exc}", flush=True)
                hnswlib = None

        d = root / strategy
        with (d / "meta.pkl").open("rb") as f:
            meta = pickle.load(f)

        ix = cls(strategy)
        ix.chunk_ids = meta["chunk_ids"]
        ix.passage_ids = meta["passage_ids"]
        ix.texts = meta["texts"]
        ix.query_types = meta["query_types"]
        ix.langs = meta["langs"]

        if load_hnsw and hnswlib is not None and (d / "hnsw.bin").exists():
            try:
                ix.hnsw = hnswlib.Index(space="ip", dim=DIM)
                ix.hnsw.load_index(str(d / "hnsw.bin"), max_elements=len(ix.chunk_ids))
                ix.hnsw.set_ef(ef_search)
            except Exception as e:
                print(f"[ChunkIndex] HNSW load skipped: {e}", flush=True)

        if (d / "bm25").exists() or (d / "bm25.json").exists():
            try:
                ix.bm25 = bm25s.BM25.load(str(d / "bm25"))
            except Exception as e:
                print(f"[ChunkIndex] BM25 load skipped: {e}", flush=True)

        return ix

    def __len__(self) -> int:
        return len(self.chunk_ids)
