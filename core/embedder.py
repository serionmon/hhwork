"""ONNX embedder for multilingual-e5-small (384-dim, XLM-R vocab).

Two things about e5 that are easy to get wrong and quietly cost retrieval
quality:

  1. **Prefixes are mandatory.** e5 was trained with "query: " on queries and
     "passage: " on documents. Dropping them, or using the same prefix for
     both, measurably degrades results. `encode_query` / `encode_passages`
     exist so the call site cannot forget.
  2. **Mean-pool over the attention mask, then L2-normalise.** Not CLS pooling.
     Normalising lets us use a plain dot product as cosine downstream, which is
     what hnswlib's inner-product space expects.

Model variants -- pick by target architecture:
  official fp32   intfloat/multilingual-e5-small : onnx/model.onnx
  official int8   intfloat/multilingual-e5-small : onnx/model_qint8_avx512_vnni.onnx  (x86 only)
  generic int8    Xenova/multilingual-e5-small   : onnx/model_int8.onnx               (ARM-safe)
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DIM = 384
MAX_LEN = 512

# (repo_id, onnx_filename)
VARIANTS = {
    "fp32": ("intfloat/multilingual-e5-small", "onnx/model.onnx"),
    "int8_x86": ("intfloat/multilingual-e5-small", "onnx/model_qint8_avx512_vnni.onnx"),
    "int8_arm": ("Xenova/multilingual-e5-small", "onnx/model_int8.onnx"),
}
TOKENIZER_REPO = "intfloat/multilingual-e5-small"


def available_providers() -> list[str]:
    """Execution providers to try, best first. Override with ORT_PROVIDERS.

    Defaults to CPUExecutionProvider to ensure container safety and prevent unnecessary
    GPU device discovery calls on CPU-only environments.
    """
    if env := os.getenv("ORT_PROVIDERS"):
        return [p.strip() for p in env.split(",") if p.strip()]
    return ["CPUExecutionProvider"]


def default_variant() -> str:
    """Pick the ONNX build that suits the execution provider."""
    if v := os.getenv("E5_VARIANT"):
        return v
    if "CUDAExecutionProvider" in available_providers():
        return "fp32"
    return "int8_arm"


def check_ort_compatibility() -> bool:
    """Run an isolated subprocess probe to verify ONNX Runtime compatibility on this host.

    Prevents SIGILL (Exit status 132) hardware instruction traps from crashing the main process.
    """
    if os.getenv("DISABLE_ONNX") == "1":
        print("[Embedder] DISABLE_ONNX=1 set; skipping ONNX Runtime initialization.", flush=True)
        return False

    probe_code = (
        "import os, sys\n"
        "try:\n"
        "    import onnxruntime as ort\n"
        "    so = ort.SessionOptions()\n"
        "    p = [os.getenv('ORT_PROVIDERS', 'CPUExecutionProvider')]\n"
        "    print('ORT_PROBE_OK', flush=True)\n"
        "except Exception as e:\n"
        "    print(f'ORT_PROBE_FAIL:{e}', flush=True)\n"
        "    sys.exit(1)\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and "ORT_PROBE_OK" in proc.stdout:
            return True
        print(
            f"[Embedder] ONNX probe failed (returncode {proc.returncode}). "
            f"Stderr: {proc.stderr.strip()} Stdout: {proc.stdout.strip()}",
            flush=True,
        )
        return False
    except Exception as err:
        print(f"[Embedder] ONNX probe exception: {err}", flush=True)
        return False


@dataclass(slots=True)
class EmbedderConfig:
    variant: str = ""
    max_len: int = MAX_LEN
    batch_size: int = 64
    threads: int = 0  # 0 -> onnxruntime picks


class Embedder:
    def __init__(self, cfg: EmbedderConfig | None = None, cache_dir: Path | None = None):
        self.cfg = cfg or EmbedderConfig()
        self._available = check_ort_compatibility()
        self.session = None
        self.tokenizer = None
        self.provider = "CPUExecutionProvider"
        self.variant = self.cfg.variant or default_variant()

        if not self._available:
            print("[Embedder] ONNX Runtime unavailable or disabled on this host. Using pure Python BM25 mode.", flush=True)
            return

        # Lazy import of ONNX & HuggingFace tokenizers inside instance initialization
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}; pick from {list(VARIANTS)}")
        repo, fname = VARIANTS[self.variant]

        model_path = hf_hub_download(repo, fname, cache_dir=str(cache_dir) if cache_dir else None)
        tok_path = hf_hub_download(
            TOKENIZER_REPO, "tokenizer.json", cache_dir=str(cache_dir) if cache_dir else None
        )

        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_truncation(max_length=self.cfg.max_len)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")  # XLM-R pad id

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.cfg.threads > 0:
            so.intra_op_num_threads = self.cfg.threads

        self.providers = available_providers()
        try:
            self.session = ort.InferenceSession(model_path, so, providers=self.providers)
        except Exception as err:
            if self.variant != "fp32":
                print(f"Warning: failed to load model variant {self.variant} ({err}), falling back to fp32", flush=True)
                self.variant = "fp32"
                repo_fb, fname_fb = VARIANTS["fp32"]
                model_path_fb = hf_hub_download(
                    repo_fb, fname_fb, cache_dir=str(cache_dir) if cache_dir else None
                )
                self.session = ort.InferenceSession(model_path_fb, so, providers=self.providers)
            else:
                self._available = False
                raise

        self.provider = self.session.get_providers()[0]
        self._input_names = {i.name for i in self.session.get_inputs()}

        print("=== ONNX Runtime Diagnostic ===", flush=True)
        print(f"  Python version      : {sys.version.split()[0]}", flush=True)
        print(f"  ONNX Runtime version: {getattr(ort, '__version__', 'unknown')}", flush=True)
        print(f"  Selected provider   : {self.provider}", flush=True)
        print(f"  Selected variant    : {self.variant}", flush=True)
        print("================================", flush=True)

        if "CUDA" in self.provider and cfg is None:
            self.cfg.batch_size = int(os.getenv("EMBED_BATCH", "256"))

    def is_available(self) -> bool:
        return self._available and self.session is not None

    # -- internals ---------------------------------------------------------

    def _forward(self, texts: list[str]) -> np.ndarray:
        if not self.is_available():
            return np.zeros((len(texts), DIM), dtype=np.float32)

        enc = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)

        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}

        hidden = self.session.run(None, feeds)[0]  # (B, T, DIM)

        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-12, None)).astype(np.float32)

    def _encode(self, texts: list[str], prefix: str, batch_size: int | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        if not self.is_available():
            return np.zeros((len(texts), DIM), dtype=np.float32)

        prefixed = [prefix + t for t in texts]
        order = sorted(range(len(prefixed)), key=lambda i: len(prefixed[i]))

        out = np.empty((len(prefixed), DIM), dtype=np.float32)
        bs = batch_size or self.cfg.batch_size
        for i in range(0, len(order), bs):
            idx = order[i : i + bs]
            out[idx] = self._forward([prefixed[j] for j in idx])
        return out

    # -- public ------------------------------------------------------------

    def encode_query(self, text: str) -> np.ndarray:
        """Single query -> (DIM,) normalised vector."""
        if not self.is_available():
            return np.zeros((DIM,), dtype=np.float32)
        return self._encode([text], "query: ")[0]

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if not self.is_available():
            return np.zeros((len(texts), DIM), dtype=np.float32)
        return self._encode(texts, "query: ")

    def encode_passages(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        if not self.is_available():
            return np.zeros((len(texts), DIM), dtype=np.float32)
        return self._encode(texts, "passage: ", batch_size)
