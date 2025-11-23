from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..config import config

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from sentence_transformers import SentenceTransformer as SentenceTransformerType
else:  # Fallback to Any to keep runtime import optional
    SentenceTransformerType = Any

DEFAULT_MODEL = config.EMBEDDING_MODEL
CACHE_DIR = Path(os.path.expanduser(config.EMBEDDING_CACHE_DIR))


def _load_sentence_transformer_cls() -> type[SentenceTransformerType]:
    """Import SentenceTransformer lazily to avoid heavy deps at module import."""
    try:
        from sentence_transformers import SentenceTransformer as cls
    except Exception as exc:  # pragma: no cover - exercised when deps missing
        raise ImportError(
            "Embeddings support requires the 'tinyvecdb[server]' extra."
        ) from exc
    return cls


def _load_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - exercised when deps missing
        raise ImportError(
            "Embeddings support requires the 'tinyvecdb[server]' extra."
        ) from exc
    return snapshot_download


def load_default_model() -> SentenceTransformerType:
    """
    Load the embedding model specified in the config.

    Returns:
        Loaded SentenceTransformer model.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    snapshot = _load_snapshot_download()
    st_cls = _load_sentence_transformer_cls()

    model_path = snapshot(
        repo_id=DEFAULT_MODEL,
        cache_dir=CACHE_DIR,
        local_files_only=False,  # auto-download first time
    )

    # Quantized + CPU-friendly
    model = st_cls(
        model_path,
        model_kwargs={"dtype": "auto", "file_name": "model.onnx"},
        tokenizer_kwargs={"padding": True, "truncation": True, "max_length": 512},
        backend="onnx",
    )
    # Optional: enable memory-efficient attention if flash-attn available (no-op otherwise)
    # Modern PyTorch (2.0+) uses SDPA by default, so explicit BetterTransformer conversion
    # is often unnecessary or deprecated in newer transformers versions.

    return model


# Global singleton
_default_model: SentenceTransformerType | None = None


def get_embedder() -> SentenceTransformerType:
    """
    Get the global singleton embedding model.

    Returns:
        SentenceTransformer instance.
    """
    global _default_model
    if _default_model is None:
        _default_model = load_default_model()
    return _default_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using the default model.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (list of floats).
    """
    model = get_embedder()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
    )
    return embeddings.tolist()
