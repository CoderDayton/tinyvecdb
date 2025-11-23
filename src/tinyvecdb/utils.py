from __future__ import annotations

import importlib
import sys
from typing import Any


def _import_optional(name: str) -> Any:
    """Attempt to import a module while honoring tests that stub sys.modules."""
    sentinel = object()
    existing = sys.modules.get(name, sentinel)
    if existing is None:
        return None
    if existing is not sentinel:
        return existing
    try:
        return importlib.import_module(name)
    except Exception:
        return None
