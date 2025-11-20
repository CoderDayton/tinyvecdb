# tests/unit/test_search.py
import pytest
import numpy as np
from tinyvecdb import VectorDB, DistanceStrategy


@pytest.fixture
def db():
    db = VectorDB(":memory:")
    texts = ["apple", "banana", "orange", "grape"]
    embeddings = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.9, 0.9, 0.9],
            [0.85, 0.85, 0.85],
        ],
        dtype=np.float32,
    )
    metadatas = [
        {"type": "fruit"},
        {"type": "fruit"},
        {"type": "fruit"},
        {"type": "fruit"},
    ]
    db.add_texts(texts, embeddings=embeddings.tolist(), metadatas=metadatas)
    return db


def test_similarity_search_basic(db):
    results = db.similarity_search([0.95, 0.95, 0.95], k=2)
    assert len(results) == 2
    assert results[0][0].page_content == "grape"  # closest
    assert results[1][0].page_content == "orange"
    assert results[0][1] >= 0  # non-negative distance
    assert results[0][1] <= results[1][1]  # first result closer


def test_similarity_search_filter(db):
    results = db.similarity_search([0.95, 0.95, 0.95], k=2, filter={"type": "fruit"})
    assert len(results) == 2  # all match, but test filtering logic


def test_brute_force_fallback():
    db = VectorDB(":memory:", distance_strategy=DistanceStrategy.L2)
    texts = ["a", "b"]
    embs = [[1.0, 0.0], [0.0, 1.0]]
    db.add_texts(texts, embeddings=embs)
    # Simulate no-extension by dropping vec_index
    db.conn.execute("DROP TABLE vec_index")
    # This should fall back to brute force if implemented
    results = db.similarity_search([1.0, 0.0], k=1)
    assert results[0][0].page_content == "a"
    assert pytest.approx(results[0][1], 0.001) == 0.0


def test_recall_toy_dataset(db):
    # Gold standard: full scan
    texts = ["apple", "banana", "orange", "grape"]
    query = np.array([0.95, 0.95, 0.95])
    all_embs = np.array(
        [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.9, 0.9, 0.9], [0.85, 0.85, 0.85]]
    )
    all_embs /= np.linalg.norm(all_embs, axis=1, keepdims=True)
    query /= np.linalg.norm(query)
    sims = np.dot(all_embs, query)
    gold_topk = np.argsort(-sims)[:2]  # indices 3,2 (grape, orange)

    results = db.similarity_search(query, k=2)
    result_indices = [
        texts.index(r[0].page_content) for r in results
    ]  # assuming texts list order
    recall = len(set(gold_topk) & set(result_indices)) / 2
    assert recall >= 0.9
