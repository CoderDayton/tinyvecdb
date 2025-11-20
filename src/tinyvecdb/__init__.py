from __future__ import annotations

import dataclasses
from dataclasses import field
from enum import StrEnum

__version__ = "0.0.1"
__all__ = ["VectorDB", "Document", "DistanceStrategy"]


@dataclasses.dataclass(frozen=True, slots=True)
class Document:
    """Simple document with text content and arbitrary metadata."""

    page_content: str
    metadata: dict = field(default_factory=dict)


class DistanceStrategy(StrEnum):
    """Supported distance metrics (matches sqlite-vec exactly)"""

    COSINE = "cosine"
    L2 = "l2"  # euclidean
    IP = "inner"  # inner product (negative distance = higher similarity)


from .core import VectorDB  # noqa: E402
