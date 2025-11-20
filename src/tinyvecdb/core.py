from __future__ import annotations

import sqlite3
import struct
import numpy as np
from typing import List, Sequence, Union, Optional, Tuple, Any, Iterable
import sqlite_vec
from pathlib import Path
from typing import Dict
import json

from . import Document, DistanceStrategy


def _serialize_float32(vector: Sequence[float]) -> bytes:
    """Pack a list/array of floats into little-endian float32 blob."""
    return struct.pack("<%sf" % len(vector), *(float(x) for x in vector))


def _normalize_l2(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


class VectorDB:
    """
    Dead-simple local vector database powered by sqlite-vec.
    One SQLite file = one collection. Chroma-style API.
    """

    def __init__(
        self,
        path: Union[str, Path] = ":memory:",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
    ):
        self.path = str(path)
        self.distance_strategy = distance_strategy

        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.enable_load_extension(True)
        try:
            sqlite_vec.load(self.conn)
        except sqlite3.OperationalError:
            # Some platforms (WASM, restricted envs) can't load – we'll fallback later
            pass
        self.conn.enable_load_extension(False)

        self._dim: Optional[int] = None
        self._table_name = "tinyvec_items"
        self._create_table()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _create_table(self) -> None:
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                metadata TEXT,                     -- JSON string
                embedding BLOB NOT NULL
            )
            """
        )
        # Virtual table is created lazily on first insert once we know dimension

    def _ensure_virtual_table(self, dim: int) -> None:
        if self._dim is not None and self._dim != dim:
            raise ValueError(f"Dimension mismatch: existing {self._dim}, got {dim}")
        if self._dim is None:
            # First insert – recreate virtual table with correct dimension
            self._dim = dim
            self.conn.execute("DROP TABLE IF EXISTS vec_index")
            self.conn.execute(
                f"""
                CREATE VIRTUAL TABLE vec_index USING vec0(
                    embedding float[{dim}]
                )
                """
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[dict]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
        ids: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """
        Add texts with optional pre-computed embeddings.
        Returns the assigned integer IDs.
        """
        if not texts:
            raise ValueError("No texts provided")

        import json

        if metadatas is None:
            metadatas = [{} for _ in texts]

        if embeddings is None:
            try:
                from .embeddings.models import embed_texts

                embeddings = embed_texts(list(texts))
            except Exception as e:
                raise ValueError(
                    "No embeddings provided and local embedder failed – install with [server] extra"
                ) from e

        first_emb = embeddings[0]
        dim = len(first_emb)
        self._ensure_virtual_table(dim)

        # Normalise for cosine if needed
        if self.distance_strategy == DistanceStrategy.COSINE:
            embeddings = [
                _normalize_l2(np.array(e, dtype=np.float32)) for e in embeddings
            ]

        rows = []
        vec_rows = []
        for i, (text, meta, emb) in enumerate(zip(texts, metadatas, embeddings)):
            rowid = ids[i] if ids and ids[i] is not None else None
            rows.append((rowid, text, json.dumps(meta), _serialize_float32(emb)))
            vec_rows.append(
                (rowid or i + 999999999, _serialize_float32(emb))
            )  # temporary rowid

        with self.conn:
            # Insert main rows (id may be NULL → AUTOINCREMENT)
            self.conn.executemany(
                f"""
                INSERT INTO {self._table_name}(id, text, metadata, embedding)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    text=excluded.text,
                    metadata=excluded.metadata,
                    embedding=excluded.embedding
                """,
                rows,
            )
            # Sync to virtual table using real rowids
            actual_ids = [
                row[0]
                for row in self.conn.execute(
                    "SELECT id FROM tinyvec_items WHERE rowid IN (SELECT max(rowid) FROM tinyvec_items GROUP BY id)"
                )
            ]
            # Simpler: just re-insert into vec_index with correct rowid
            self.conn.execute("DELETE FROM vec_index")
            real_vec_rows = []
            for real_id, (_, _, _, blob) in zip(actual_ids, rows):
                real_vec_rows.append((real_id, blob))
            self.conn.executemany(
                "INSERT INTO vec_index(rowid, embedding) VALUES (?, ?)",
                real_vec_rows,
            )

        return actual_ids or [
            row[0]
            for row in self.conn.execute(
                "SELECT id FROM tinyvec_items ORDER BY id DESC LIMIT ?", (len(texts),)
            )
        ]

    def upsert_texts(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        ids: Sequence[int],
        metadatas: Optional[Sequence[dict]] = None,
    ) -> None:
        """Upsert by explicit IDs (add_texts handles ON CONFLICT)."""
        # Reuse add_texts logic – it's already upsert-capable
        self.add_texts(texts, metadatas, embeddings, ids=ids)

    def delete_by_ids(self, ids: Iterable[int]) -> None:
        """Delete documents by IDs."""
        with self.conn:
            placeholders = ",".join("?" for _ in ids)
            self.conn.execute(
                f"DELETE FROM {self._table_name} WHERE id IN ({placeholders})",
                list(ids),
            )
            self.conn.execute(
                f"DELETE FROM vec_index WHERE rowid IN ({placeholders})",
                list(ids),
            )

    def similarity_search(
        self,
        query: Union[str, Sequence[float]],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Return top-k documents with distances.
        Supports vector queries (text queries require embeddings integration).
        Optional metadata filter as dict (e.g., {"category": "fruit"}).
        """
        import json

        if self._dim is None:
            return []  # empty collection

        if isinstance(query, str):
            try:
                from .embeddings.models import embed_texts

                query_embedding = embed_texts([query])[0]
                query_vec = np.array(query_embedding, dtype=np.float32)
            except Exception as e:
                raise ValueError(
                    "Text queries require embeddings – install with [server] extra or provide vector query"
                ) from e
        else:
            query_vec = np.array(query, dtype=np.float32)
        if len(query_vec) != self._dim:
            raise ValueError(
                f"Query dim {len(query_vec)} != collection dim {self._dim}"
            )

        if self.distance_strategy == DistanceStrategy.COSINE:
            query_vec = _normalize_l2(query_vec)

        blob = _serialize_float32(query_vec)

        try:
            # Fast path: use sqlite-vec if available
            sql = """
                SELECT
                    rowid,
                    distance
                FROM vec_index
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            """
            params = (blob, k)
            candidates = self.conn.execute(sql, params).fetchall()

        except sqlite3.OperationalError:
            # Fallback: pure NumPy brute-force scan (slow but works everywhere)
            candidates = self._brute_force_search(query_vec, k)

        # Apply metadata filter if any
        if filter:
            candidates = [c for c in candidates if self._matches_filter(c[0], filter)]

        # Fetch full docs (limit to final k after filter)
        results = []
        for rowid, distance in candidates[:k]:
            text, meta_json = self.conn.execute(
                f"SELECT text, metadata FROM {self._table_name} WHERE id = ?",
                (rowid,),
            ).fetchone()
            meta = json.loads(meta_json) if meta_json else {}
            results.append(
                (Document(page_content=text, metadata=meta), float(distance))
            )

        return results

    def _brute_force_search(
        self, query_vec: np.ndarray, k: int
    ) -> List[Tuple[int, float]]:
        """Pure-NumPy fallback for no-extension environments."""
        rows = self.conn.execute(
            f"SELECT id, embedding FROM {self._table_name}"
        ).fetchall()

        if not rows:
            return []

        ids, blobs = zip(*rows)
        vectors = np.array([np.frombuffer(blob, dtype=np.float32) for blob in blobs])

        if self.distance_strategy == DistanceStrategy.COSINE:
            dots = np.dot(vectors, query_vec)
            norms = np.linalg.norm(vectors, axis=1)
            similarities = dots / (norms * np.linalg.norm(query_vec))
            distances = 1 - similarities  # cosine distance = 1 - sim
        elif self.distance_strategy == DistanceStrategy.L2:
            distances = np.linalg.norm(vectors - query_vec, axis=1)
        else:  # IP
            distances = -np.dot(vectors, query_vec)  # negative for sorting

        # Sort and take top-k
        indices = np.argsort(distances)[:k]
        return [(ids[i], distances[i]) for i in indices]

    def _matches_filter(self, rowid: int, filter_dict: Dict[str, Any]) -> bool:
        """Simple JSON filter matcher (expand for complex in v1)."""
        meta_json = self.conn.execute(
            f"SELECT metadata FROM {self._table_name} WHERE id = ?",
            (rowid,),
        ).fetchone()[0]
        if not meta_json:
            return False
        meta = json.loads(meta_json)

        for key, value in filter_dict.items():
            if meta.get(key) != value:  # exact match for now
                return False
        return True

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self.conn.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
