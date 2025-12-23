from __future__ import annotations

import dataclasses
from dataclasses import field
from enum import Enum


class StrEnum(str, Enum):
    """Enum where members are also (and must be) strings"""

    def __str__(self) -> str:
        return str(self.value)


@dataclasses.dataclass(frozen=True, slots=True)
class Document:
    """Simple document with text content and arbitrary metadata."""

    page_content: str
    metadata: dict = field(default_factory=dict)


class DistanceStrategy(StrEnum):
    """Supported distance metrics for usearch backend."""

    COSINE = "cosine"
    L2 = "l2"  # euclidean (squared L2 internally)
    # Note: L1 (manhattan) was removed in v2.0.0 - usearch doesn't support it


class Quantization(StrEnum):
    FLOAT = "float"
    INT8 = "int8"
    BIT = "bit"
