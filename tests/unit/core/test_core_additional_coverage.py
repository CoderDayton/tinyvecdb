"""Additional coverage tests for VectorDB core."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simplevecdb.core import (
    VectorDB,
    _batched,
    get_optimal_batch_size,
)


def test_batched_handles_sequence():
    """Ensure _batched slices sequences without iterator fallback."""
    batches = list(_batched([1, 2, 3, 4], 3))
    assert batches == [[1, 2, 3], [4]]


def test_get_optimal_batch_size_arm_many_cores_returns_16():
    """ARM machines with many cores should return the largest branch."""
    mock_psutil = MagicMock()
    mock_psutil.cpu_count.return_value = 12
    mock_psutil.virtual_memory.return_value.available = 16 * 1024**3

    with (
        patch.dict(
            sys.modules,
            {"onnxruntime": None, "torch": None, "psutil": mock_psutil},
        ),
        patch("platform.machine", return_value="arm64"),
    ):
        assert get_optimal_batch_size() == 16


def test_add_texts_uses_local_embedder_numpy(tmp_path):
    """When embeddings are missing, local embedder should run and accept numpy."""
    db_path = tmp_path / "auto_embed.db"
    embed_returns = [
        [np.array([1.0, 0.0, 0.0], dtype=np.float32)],
        [np.array([0.5, 0.5, 0.5], dtype=np.float32)],
    ]

    with patch("simplevecdb.embeddings.models.embed_texts", side_effect=embed_returns):
        db = VectorDB(str(db_path))
        collection = db.collection("default")
        first_ids = collection.add_texts(["alpha"], metadatas=[{"idx": 1}])
        second_ids = collection.add_texts(["beta"], metadatas=[{"idx": 2}])

    assert len(first_ids) == 1
    assert len(second_ids) == 1
    assert collection._dim == 3
    db.close()


def test_remove_texts_requires_criteria(tmp_path):
    """remove_texts should demand either texts or filters."""
    db = VectorDB(str(tmp_path / "remove_none.db"))
    collection = db.collection("default")
    with pytest.raises(ValueError):
        collection.remove_texts()
    db.close()


def test_remove_texts_combines_text_and_filter(tmp_path):
    """Removal should deduplicate IDs gathered from texts and filters."""
    db = VectorDB(str(tmp_path / "remove.db"))
    collection = db.collection("default")
    collection.add_texts(
        ["dup", "filter", "keep"],
        metadatas=[{"topic": "target"}, {"topic": "filter"}, {"topic": "keep"}],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

    removed = collection.remove_texts(texts=["dup"], filter={"topic": "filter"})
    assert removed == 2

    remaining = db.conn.execute(
        f"SELECT text FROM {collection._table_name} ORDER BY id"
    ).fetchall()
    assert [row[0] for row in remaining] == ["keep"]
    db.close()


def test_vector_db_del_swallows_close_error(tmp_path):
    """__del__ should ignore close failures."""
    db = VectorDB(str(tmp_path / "del.db"))
    db.close = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[assignment]

    db.__del__()  # Should not raise
    assert db.close.called
