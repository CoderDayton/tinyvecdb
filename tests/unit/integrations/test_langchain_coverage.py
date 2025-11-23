"""LangChain integration coverage helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_langchain_delete(tmp_path):
    """Test LangChain integration delete method."""
    from tinyvecdb.integrations.langchain import TinyVecDBVectorStore

    db_path = tmp_path / "lc_del.db"

    # Mock the embedding function to return valid embeddings
    mock_embedding = MagicMock()
    mock_embedding.embed_documents.return_value = [np.random.rand(384).tolist()]

    store = TinyVecDBVectorStore(str(db_path), embedding=mock_embedding)

    # Add some dummy data to delete with pre-computed embeddings
    ids = store.add_texts(["text1"])

    # Delete - LangChain's delete returns None
    store.delete(ids)
    # No assertion needed, just verify it doesn't crash


def test_langchain_similarity_search_requires_embedding(tmp_path):
    """Ensure text queries fail when no embedding model is configured."""
    from tinyvecdb.integrations.langchain import TinyVecDBVectorStore

    store = TinyVecDBVectorStore(str(tmp_path / "lc_no_embed.db"))

    with pytest.raises(ValueError):
        store.similarity_search("query")


def test_langchain_similarity_search_with_score_returns_scores(tmp_path):
    """Validate similarity_search_with_score uses embeddings and returns tuples."""
    from tinyvecdb.integrations.langchain import TinyVecDBVectorStore

    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.5] * 3
    store = TinyVecDBVectorStore(
        str(tmp_path / "lc_with_score.db"), embedding=mock_embedding
    )

    mock_doc = SimpleNamespace(page_content="content", metadata={"source": "unit"})

    with patch.object(store, "_collection") as mock_col:
        mock_col.similarity_search.return_value = [(mock_doc, 0.25)]

        results = store.similarity_search_with_score("query", k=1)

    assert len(results) == 1
    doc, score = results[0]
    assert doc.page_content == "content"
    assert score == 0.25
    mock_embedding.embed_query.assert_called_once_with("query")
    mock_col.similarity_search.assert_called_once_with(query=[0.5] * 3, k=1, filter=None)


def test_langchain_mmr_requires_embedding(tmp_path):
    """MMR search should raise when no embedding model is provided."""
    from tinyvecdb.integrations.langchain import TinyVecDBVectorStore

    store = TinyVecDBVectorStore(str(tmp_path / "lc_mmr_no_embed.db"))

    with pytest.raises(ValueError):
        store.max_marginal_relevance_search("query")
