from __future__ import annotations

from .types import Document, DistanceStrategy
from .core import VectorDB, VectorCollection, Quantization, get_optimal_batch_size
from .config import config
from .integrations.langchain import SimpleVecDBVectorStore
from .integrations.llamaindex import SimpleVecDBLlamaStore

__version__ = "1.0.0"
__all__ = [
    "VectorDB",
    "VectorCollection",
    "Quantization",
    "Document",
    "DistanceStrategy",
    "SimpleVecDBVectorStore",
    "SimpleVecDBLlamaStore",
    "config",
    "get_optimal_batch_size",
]
