"""Similarity search and query tests."""

from unittest.mock import patch
import pytest

from simplevecdb import VectorDB
from simplevecdb.core import Quantization


def test_similarity_search_text_query_error():
    """Test similarity search with text query when embeddings unavailable."""
    import sys

    db = VectorDB(":memory:")
    collection = db.collection("default")
    collection.add_texts(["test"], embeddings=[[0.1, 0.2]])

    # Mock embeddings module to be unavailable
    with patch.dict(sys.modules, {"simplevecdb.embeddings.models": None}):
        # Try text query - should raise ValueError
        with pytest.raises((ValueError, AttributeError)):
            collection.similarity_search("query text", k=1)


def test_similarity_search_with_int8_quantization():
    """Test similarity search uses vec_int8 placeholder for INT8."""
    db = VectorDB(":memory:", quantization=Quantization.INT8)
    collection = db.collection("default")
    texts = ["a", "b"]
    embs = [[0.1, 0.2], [0.3, 0.4]]
    collection.add_texts(texts, embeddings=embs)

    results = collection.similarity_search([0.1, 0.2], k=1)
    assert len(results) == 1


def test_similarity_search_with_bit_quantization():
    """Test similarity search uses vec_bit placeholder for BIT."""
    db = VectorDB(":memory:", quantization=Quantization.BIT)
    collection = db.collection("default")
    texts = ["a", "b"]
    # BIT requires dimensions divisible by 8
    embs = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] for _ in range(2)]
    collection.add_texts(texts, embeddings=embs)

    results = collection.similarity_search([0.1] * 8, k=1)
    assert len(results) == 1
