#!/usr/bin/env python3
"""
FastEmbed wrapper: all-MiniLM-L6-v2 (384-dim, ONNX, runs on CPU).
No GPU and no API key required. Model cache: ~/.cache/fastembed/ (~23 MB, downloaded once).
"""
import sys
import warnings
from pathlib import Path
from functools import lru_cache
from typing import List

import numpy as np

# FastEmbed / onnxruntime / urllib3 emit import-time warnings we cannot act on.
# Scope the suppression to this import rather than silencing the whole process.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastembed import TextEmbedding

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE = Path.home() / ".cache" / "fastembed"


def is_model_cached() -> bool:
    """Return True if the ONNX model file is already on disk."""
    if not MODEL_CACHE.exists():
        return False
    # fastembed stores models under ~/.cache/fastembed/<model-name>/
    return any(MODEL_CACHE.rglob("*.onnx"))


def ensure_model_available() -> None:
    """
    Check that the model is cached. If not, attempt to download it and
    raise a clear RuntimeError on failure (network down, etc.).
    """
    if is_model_cached():
        return

    print(
        f"  First run: downloading embedding model ({MODEL_NAME})...\n"
        f"  Cache destination: {MODEL_CACHE}\n"
        f"  This is a one-time ~23 MB download.",
        flush=True,
    )
    try:
        # Instantiating TextEmbedding triggers the download
        TextEmbedding(model_name=MODEL_NAME, cache_dir=str(MODEL_CACHE))
    except Exception as exc:
        cause = exc.__cause__ or exc
        raise RuntimeError(
            f"\n\n  Could not download the embedding model.\n"
            f"      Model : {MODEL_NAME}\n"
            f"      Error : {cause}\n\n"
            f"  The first run needs network access to download the ONNX model\n"
            f"  (~23 MB). Once cached at {MODEL_CACHE}, no network is needed.\n\n"
            f"  Fix: connect to the internet and re-run."
        ) from exc


@lru_cache(maxsize=1)
def get_model() -> TextEmbedding:
    """Load the ONNX model once and keep it in memory for the process lifetime."""
    return TextEmbedding(model_name=MODEL_NAME, cache_dir=str(MODEL_CACHE))


def embed(texts: List[str]) -> List[np.ndarray]:
    """
    Embed a batch of texts. Returns a list of 384-dim float32 arrays.
    Always batch, even a single text should be passed as a list.
    """
    model = get_model()
    return list(model.embed(texts))


def embed_one(text: str) -> np.ndarray:
    """Embed a single query string."""
    return embed([text])[0]


if __name__ == "__main__":
    test = sys.argv[1] if len(sys.argv) > 1 else "how does reciprocal rank fusion work"
    print(f"Embedding: '{test}'")
    vec = embed_one(test)
    print(f"Shape: {vec.shape}, dtype: {vec.dtype}, norm: {np.linalg.norm(vec):.4f}")
