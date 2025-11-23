from __future__ import annotations

import sqlite3
import struct
import json
import re
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Any, TYPE_CHECKING
import sqlite_vec  # type: ignore
from pathlib import Path
import platform
import multiprocessing
import itertools

from .types import Document, DistanceStrategy, StrEnum
from .utils import _import_optional

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from .integrations.langchain import TinyVecDBVectorStore
    from .integrations.llamaindex import TinyVecDBLlamaStore


class Quantization(StrEnum):
    FLOAT = "float"
    INT8 = "int8"
    BIT = "bit"


def _serialize_vector(vector: np.ndarray, quant: Quantization) -> bytes:
    """Serialize a normalized float vector according to quantization mode."""
    if quant == Quantization.FLOAT:
        return struct.pack("<%sf" % len(vector), *(float(x) for x in vector))

    elif quant == Quantization.INT8:
        # Scalar quantization: scale to [-128, 127]
        scaled = np.clip(np.round(vector * 127), -128, 127).astype(np.int8)
        return scaled.tobytes()

    elif quant == Quantization.BIT:
        # Binary quantization: threshold at 0 → pack bits
        bits = (vector > 0).astype(np.uint8)
        packed = np.packbits(bits)
        return packed.tobytes()

    raise ValueError(f"Unsupported quantization: {quant}")


def _dequantize_vector(blob: bytes, dim: int | None, quant: Quantization) -> np.ndarray:
    """Reverse serialization for fallback path."""
    if quant == Quantization.FLOAT:
        return np.frombuffer(blob, dtype=np.float32)

    elif quant == Quantization.INT8:
        return np.frombuffer(blob, dtype=np.int8).astype(np.float32) / 127.0

    elif quant == Quantization.BIT and dim is not None:
        unpacked = np.unpackbits(np.frombuffer(blob, dtype=np.uint8))
        v = unpacked[:dim].astype(np.float32)
        return np.where(v == 1, 1.0, -1.0)

    raise ValueError(f"Unsupported quantization: {quant} or unknown dim {dim}")


def _normalize_l2(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def _batched(iterable: Iterable[Any], n: int) -> Iterable[Sequence[Any]]:
    """Batch data into lists of length n. The last batch may be shorter."""
    if isinstance(iterable, Sequence):
        for i in range(0, len(iterable), n):
            yield iterable[i : i + n]
    else:
        it = iter(iterable)
        while True:
            batch = list(itertools.islice(it, n))
            if not batch:
                return
            yield batch


def get_optimal_batch_size() -> int:
    """
    Automatically determine optimal batch size based on hardware.

    Detection hierarchy:
    1. CUDA GPU (NVIDIA) - High batch sizes for desktop/server GPUs
    2. ROCm GPU (AMD) - Similar to CUDA for high-end cards
    3. MPS (Apple Metal Performance Shaders) - Apple Silicon optimization
    4. ONNX Runtime GPU (CUDA/TensorRT/DirectML)
    5. CPU - Scale with cores and architecture

    Returns:
        Optimal batch size for the detected hardware.
    """
    # 1. Try PyTorch detection first
    torch = _import_optional("torch")
    if torch is not None:
        # Check for NVIDIA CUDA GPU
        if torch.cuda.is_available():
            # Get GPU properties
            gpu_props = torch.cuda.get_device_properties(0)
            vram_gb = gpu_props.total_memory / (1024**3)

            if vram_gb >= 20:
                return 512  # RTX 4090, A100, H100
            elif vram_gb >= 12:
                return 256  # RTX 4070 Ti, 3090, A10
            elif vram_gb >= 8:
                return 128  # RTX 4060 Ti, 3070
            else:
                return 64  # GTX 1660, RTX 3050

        # Check for AMD ROCm GPU
        if hasattr(torch, "hip") and torch.hip.is_available():  # type: ignore
            return 256

        # Check for Apple Metal (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            machine = platform.machine().lower()
            if "arm" in machine or "aarch64" in machine:
                try:
                    import subprocess

                    chip_info = subprocess.check_output(
                        ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                    ).lower()

                    if "m3" in chip_info or "m4" in chip_info:
                        return 64
                    elif "max" in chip_info or "ultra" in chip_info:
                        return 128
                    else:
                        return 32
                except Exception:
                    return 32

    # 2. Try ONNX Runtime detection
    ort = _import_optional("onnxruntime")
    if ort is not None:
        providers = ort.get_available_providers()
        if (
            "CUDAExecutionProvider" in providers
            or "TensorrtExecutionProvider" in providers
        ):
            # Hard to get VRAM from ORT directly without other libs, assume mid-range
            return 128
        if "DmlExecutionProvider" in providers:
            # DirectML (Windows AMD/Intel/NVIDIA)
            return 64
        if "CoreMLExecutionProvider" in providers:
            # Apple CoreML
            return 32

    # 3. CPU fallback - scale with available cores and RAM
    psutil = _import_optional("psutil")
    if psutil is not None:
        # Physical cores are better for dense math
        cpu_count = psutil.cpu_count(logical=False) or multiprocessing.cpu_count()
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
    else:
        cpu_count = multiprocessing.cpu_count()
        available_ram_gb = 8.0  # Assume decent machine

    machine = platform.machine().lower()

    # Check for ARM architecture (mobile/embedded)
    if "arm" in machine or "aarch64" in machine:
        if cpu_count <= 4:
            return 4
        elif cpu_count <= 8:
            return 8
        else:
            return 16

    # x86/x64 CPU
    base_batch = 16
    if cpu_count >= 32:
        base_batch = 64
    elif cpu_count >= 16:
        base_batch = 48
    elif cpu_count >= 8:
        base_batch = 32

    # Constrain by available RAM to avoid swapping
    # Rough heuristic: reduce batch size if RAM is tight
    if available_ram_gb < 2.0:
        return min(base_batch, 4)
    elif available_ram_gb < 4.0:
        return min(base_batch, 8)
    elif available_ram_gb < 8.0:
        return min(base_batch, 16)

    return base_batch


class VectorCollection:
    """
    Represents a single vector collection within the database.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        name: str,
        distance_strategy: DistanceStrategy,
        quantization: Quantization,
    ):
        self.conn = conn
        self.name = name
        self.distance_strategy = distance_strategy
        self.quantization = quantization

        # Sanitize name to prevent SQL injection
        if not re.match(r"^[a-zA-Z0-9_]+$", name):
            raise ValueError(
                f"Invalid collection name '{name}'. Must be alphanumeric + underscores."
            )

        # Table names
        if name == "default":
            self._table_name = "tinyvec_items"
            self._vec_table_name = "vec_index"
        else:
            self._table_name = f"items_{name}"
            self._vec_table_name = f"vectors_{name}"

        self._fts_table_name = f"{self._table_name}_fts"
        self._fts_enabled = False
        self._dim: int | None = None

        self._create_table()
        self._recover_dim()

    def _recover_dim(self) -> None:
        """Attempt to recover dimension from existing virtual table schema."""
        try:
            row = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = ?", (self._vec_table_name,)
            ).fetchone()
            if row and row[0]:
                match = re.search(r"(?:float|int8|bit)\[(\d+)\]", row[0])
                if match:
                    self._dim = int(match.group(1))
        except Exception:
            pass

    def _create_table(self) -> None:
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        self._ensure_fts_table()

    def _ensure_fts_table(self) -> None:
        try:
            self.conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self._fts_table_name}
                USING fts5(text)
                """
            )
            self._fts_enabled = True
        except sqlite3.OperationalError:
            self._fts_enabled = False

    def _upsert_fts_rows(self, ids: Sequence[int], texts: Sequence[str]) -> None:
        if not self._fts_enabled or not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self.conn.execute(
            f"DELETE FROM {self._fts_table_name} WHERE rowid IN ({placeholders})",
            tuple(ids),
        )
        rows = list(zip(ids, texts))
        self.conn.executemany(
            f"INSERT INTO {self._fts_table_name}(rowid, text) VALUES (?, ?)", rows
        )

    def _delete_fts_rows(self, ids: Sequence[int]) -> None:
        if not self._fts_enabled or not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self.conn.execute(
            f"DELETE FROM {self._fts_table_name} WHERE rowid IN ({placeholders})",
            tuple(ids),
        )

    def _ensure_virtual_table(self, dim: int) -> None:
        if self._dim is not None and self._dim != dim:
            raise ValueError(f"Dimension mismatch: existing {self._dim}, got {dim}")
        if self._dim is None:
            self._dim = dim
            self.conn.execute(f"DROP TABLE IF EXISTS {self._vec_table_name}")

            storage_dim = dim
            if self.quantization == Quantization.BIT:
                storage_dim = ((dim + 7) // 8) * 8

            vec_type = {
                Quantization.FLOAT: f"float[{storage_dim}]",
                Quantization.INT8: f"int8[{storage_dim}]",
                Quantization.BIT: f"bit[{storage_dim}]",
            }[self.quantization]

            # Custom SQL creation with dynamic table name
            sql = f"CREATE VIRTUAL TABLE {self._vec_table_name} USING vec0(embedding {vec_type}"
            if (
                self.distance_strategy
                and not vec_type.startswith("bit")
                and self.distance_strategy != DistanceStrategy.COSINE
            ):
                # Note: sqlite-vec defaults to cosine/l2 depending on usage, but we can enforce metric in table def
                # Actually, sqlite-vec 0.1.1+ supports distance_metric param
                sql += f" distance_metric={self.distance_strategy.value}"
            sql += ")"
            self.conn.execute(sql)

    def _build_filter_clause(
        self, filter_dict: dict[str, Any] | None, metadata_column: str = "metadata"
    ) -> tuple[str, list[Any]]:
        if not filter_dict:
            return "", []
        clauses = []
        params = []
        for key, value in filter_dict.items():
            json_path = f"$.{key}"
            if isinstance(value, (int, float)):
                clauses.append(f"json_extract({metadata_column}, ?) = ?")
                params.extend([json_path, value])
            elif isinstance(value, str):
                clauses.append(f"json_extract({metadata_column}, ?) LIKE ?")
                params.extend([json_path, f"%{value}%"])
            elif isinstance(value, list):
                placeholders = ",".join("?" for _ in value)
                clauses.append(
                    f"json_extract({metadata_column}, ?) IN ({placeholders})"
                )
                params.extend([json_path] + value)
            else:
                raise ValueError(f"Unsupported filter value type for {key}")
        where = " AND ".join(clauses)
        return f"AND ({where})" if where else "", params

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
        ids: Sequence[int | None] | None = None,
    ) -> list[int]:
        if not texts:
            return []

        embed_func = None
        if embeddings is None:
            try:
                from tinyvecdb.embeddings.models import embed_texts

                embed_func = embed_texts
            except Exception as e:
                raise ValueError(
                    "No embeddings provided and local embedder failed – install with [server] extra"
                ) from e

        from tinyvecdb import config

        all_ids = []
        n_total = len(texts)
        batch_size = config.EMBEDDING_BATCH_SIZE

        metas_it = metadatas if metadatas else ({} for _ in range(n_total))
        ids_it = ids if ids else (None for _ in range(n_total))

        combined: Iterable[Any]
        if embeddings:
            combined = zip(texts, metas_it, ids_it, embeddings)
        else:
            combined = zip(texts, metas_it, ids_it)

        for batch in _batched(combined, batch_size):
            unzipped = list(zip(*batch))
            batch_texts = list(unzipped[0])
            batch_metadatas = list(unzipped[1])
            batch_ids = list(unzipped[2])

            batch_embeddings: Sequence[float] | Sequence[list[float]] | Any
            if embeddings:
                batch_embeddings = list(unzipped[3])
            else:
                assert embed_func is not None
                batch_embeddings = embed_func(list(batch_texts))

            if self._dim is None:
                first_emb = batch_embeddings[0]
                if isinstance(first_emb, (list, tuple)):
                    dim = len(first_emb)
                elif isinstance(first_emb, np.ndarray):
                    dim = len(first_emb)
                else:
                    dim = len(list(first_emb))  # type: ignore
                self._ensure_virtual_table(dim)
            else:
                first_emb = batch_embeddings[0]
                if isinstance(first_emb, (list, tuple)):
                    first_dim = len(first_emb)
                elif isinstance(first_emb, np.ndarray):
                    first_dim = len(first_emb)
                else:
                    first_dim = len(list(first_emb))  # type: ignore
                if first_dim != self._dim:
                    raise ValueError(
                        f"Dimension mismatch: existing {self._dim}, got {first_dim}"
                    )

            emb_np = np.array(batch_embeddings, dtype=np.float32)
            if self.distance_strategy == DistanceStrategy.COSINE:
                norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
                emb_np = emb_np / np.maximum(norms, 1e-12)

            serialized = [_serialize_vector(vec, self.quantization) for vec in emb_np]

            rows = []
            for txt, meta, uid in zip(batch_texts, batch_metadatas, batch_ids):
                rows.append((uid, txt, json.dumps(meta)))

            with self.conn:
                self.conn.executemany(
                    f"""
                    INSERT INTO {self._table_name}(id, text, metadata)
                    VALUES (?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        text=excluded.text,
                        metadata=excluded.metadata
                    """,
                    rows,
                )
                batch_real_ids = [
                    r[0]
                    for r in self.conn.execute(
                        f"SELECT id FROM {self._table_name} ORDER BY id DESC LIMIT ?",
                        (len(batch_texts),),
                    )
                ]
                batch_real_ids.reverse()

                real_vec_rows = [
                    (real_id, ser) for real_id, ser in zip(batch_real_ids, serialized)
                ]

                insert_placeholder = "?"
                if self.quantization == Quantization.INT8:
                    insert_placeholder = "vec_int8(?)"
                elif self.quantization == Quantization.BIT:
                    insert_placeholder = "vec_bit(?)"

                placeholders = ",".join("?" for _ in batch_real_ids)
                self.conn.execute(
                    f"DELETE FROM {self._vec_table_name} WHERE rowid IN ({placeholders})",
                    tuple(batch_real_ids),
                )

                self.conn.executemany(
                    f"INSERT INTO {self._vec_table_name}(rowid, embedding) VALUES (?, {insert_placeholder})",
                    real_vec_rows,
                )

                self._upsert_fts_rows(batch_real_ids, batch_texts)
                all_ids.extend(batch_real_ids)

        return all_ids

    def _vector_search_candidates(
        self,
        query: str | Sequence[float],
        k: int,
        filter: dict[str, Any] | None,
    ) -> list[tuple[int, float]]:
        if self._dim is None:
            return []

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

        blob = _serialize_vector(query_vec, self.quantization)

        filter_clause, filter_params = self._build_filter_clause(
            filter, metadata_column="ti.metadata"
        )

        match_placeholder = "?"
        if self.quantization == Quantization.INT8:
            match_placeholder = "vec_int8(?)"
        elif self.quantization == Quantization.BIT:
            match_placeholder = "vec_bit(?)"

        try:
            sql = f"""
                SELECT ti.id, distance
                FROM {self._vec_table_name} vi
                JOIN {self._table_name} ti ON vi.rowid = ti.id
                WHERE embedding MATCH {match_placeholder}
                AND k = ?
                {filter_clause}
                ORDER BY distance
            """
            rows = self.conn.execute(
                sql, (blob,) + (k,) + tuple(filter_params)
            ).fetchall()
        except sqlite3.OperationalError:
            rows = self._brute_force_search(query_vec, k, filter)

        return [(int(cid), float(dist)) for cid, dist in rows[:k]]

    def _hydrate_documents(
        self, candidates: Sequence[tuple[int, float]]
    ) -> list[tuple[Document, float]]:
        results: list[tuple[Document, float]] = []
        for cid, score in candidates:
            row = self.conn.execute(
                f"SELECT text, metadata FROM {self._table_name} WHERE id = ?", (cid,)
            ).fetchone()
            if not row:
                continue
            text, meta_json = row
            meta = json.loads(meta_json) if meta_json else {}
            results.append((Document(page_content=text, metadata=meta), score))
        return results

    def similarity_search(
        self,
        query: str | Sequence[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        candidates = self._vector_search_candidates(query, k, filter)
        return self._hydrate_documents(candidates)

    def _keyword_search_candidates(
        self, query: str, k: int, filter: dict[str, Any] | None
    ) -> list[tuple[int, float]]:
        if not self._fts_enabled:
            raise RuntimeError(
                "keyword_search requires SQLite compiled with FTS5 support"
            )
        if not query.strip():
            return []

        filter_clause, filter_params = self._build_filter_clause(
            filter, metadata_column="ti.metadata"
        )

        sql = f"""
            SELECT ti.id, bm25({self._fts_table_name}) as score
            FROM {self._fts_table_name} f
            JOIN {self._table_name} ti ON ti.id = f.rowid
            WHERE {self._fts_table_name} MATCH ?
            {filter_clause}
            ORDER BY score ASC
            LIMIT ?
        """
        params = (query,) + tuple(filter_params) + (k,)
        rows = self.conn.execute(sql, params).fetchall()
        return [(int(row[0]), float(row[1])) for row in rows]

    def keyword_search(
        self, query: str, k: int = 5, filter: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        candidates = self._keyword_search_candidates(query, k, filter)
        return self._hydrate_documents(candidates)

    def _reciprocal_rank_fusion(
        self,
        dense: Sequence[tuple[int, float]],
        sparse: Sequence[tuple[int, float]],
        rrf_k: int,
    ) -> list[tuple[int, float]]:
        rank_scores: dict[int, float] = {}

        def _accumulate(items: Sequence[tuple[int, float]]):
            for rank, (doc_id, _) in enumerate(items):
                rank_scores[doc_id] = rank_scores.get(doc_id, 0.0) + 1.0 / (
                    rrf_k + rank + 1
                )

        _accumulate(dense)
        _accumulate(sparse)

        fused = sorted(rank_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(doc_id, score) for doc_id, score in fused]

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        *,
        query_vector: Sequence[float] | None = None,
        vector_k: int | None = None,
        keyword_k: int | None = None,
        rrf_k: int = 60,
    ) -> list[tuple[Document, float]]:
        if not self._fts_enabled:
            raise RuntimeError(
                "hybrid_search requires SQLite compiled with FTS5 support"
            )

        if not query.strip():
            return []

        dense_k = vector_k or max(k, 10)
        sparse_k = keyword_k or max(k, 10)

        vector_input: str | Sequence[float]
        if query_vector is not None:
            vector_input = query_vector
        else:
            vector_input = query

        dense_candidates = self._vector_search_candidates(vector_input, dense_k, filter)
        sparse_candidates = self._keyword_search_candidates(query, sparse_k, filter)

        fused_candidates = self._reciprocal_rank_fusion(
            dense_candidates, sparse_candidates, rrf_k
        )

        return self._hydrate_documents(fused_candidates[:k])

    def max_marginal_relevance_search(
        self,
        query: str | Sequence[float],
        k: int = 5,
        fetch_k: int = 20,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        candidates_with_scores = self.similarity_search(query, k=fetch_k, filter=filter)
        candidates = [doc for doc, _ in candidates_with_scores]

        if len(candidates) <= k:
            return candidates

        selected = []
        unselected = candidates.copy()
        selected.append(unselected.pop(0))

        while len(selected) < k:
            mmr_scores = []
            for candidate in unselected:
                relevance = next(
                    score for doc, score in candidates_with_scores if doc == candidate
                )
                diversity = min(
                    next(
                        score
                        for doc, score in candidates_with_scores
                        if doc == selected_doc
                    )
                    for selected_doc in selected
                )
                mmr_score = 0.5 * relevance - 0.5 * diversity
                mmr_scores.append((mmr_score, candidate))

            mmr_scores.sort(key=lambda x: x[0], reverse=True)
            selected.append(mmr_scores[0][1])
            unselected.remove(mmr_scores[0][1])

        return selected

    def _brute_force_search(
        self,
        query_vec: np.ndarray,
        k: int,
        filter: dict[str, Any] | None,
    ) -> list[tuple[int, float]]:
        batch_size = get_optimal_batch_size()

        try:
            cursor = self.conn.execute(f"SELECT rowid, embedding FROM {self._vec_table_name}")
        except sqlite3.OperationalError:
            return []

        top_k_candidates: list[tuple[int, float]] = []

        for batch in _batched(cursor, batch_size):
            if not batch:
                continue

            ids, blobs = zip(*batch)

            metas = []
            if filter:
                placeholders = ",".join("?" for _ in ids)
                meta_rows = self.conn.execute(
                    f"SELECT id, metadata FROM {self._table_name} WHERE id IN ({placeholders})",
                    ids,
                ).fetchall()
                meta_map = {r[0]: r[1] for r in meta_rows}
                metas = [meta_map.get(i) for i in ids]
            else:
                metas = [None] * len(ids)

            vectors = np.array(
                [_dequantize_vector(b, self._dim, self.quantization) for b in blobs]
            )

            if self.distance_strategy == DistanceStrategy.COSINE:
                dots = np.dot(vectors, query_vec)
                norms = np.linalg.norm(vectors, axis=1)
                similarities = dots / (norms * np.linalg.norm(query_vec) + 1e-12)
                distances = 1 - similarities
            elif self.distance_strategy == DistanceStrategy.L2:
                distances = np.linalg.norm(vectors - query_vec, axis=1)
            elif self.distance_strategy == DistanceStrategy.L1:
                distances = np.sum(np.abs(vectors - query_vec), axis=1)
            else:
                raise ValueError(
                    f"Unsupported distance strategy: {self.distance_strategy}"
                )

            batch_candidates = []
            for _, (cid, dist, meta_json) in enumerate(zip(ids, distances, metas)):
                if filter:
                    meta = json.loads(meta_json) if meta_json else {}
                    if not all(
                        meta.get(k) == v
                        if not isinstance(v, list)
                        else meta.get(k) in v
                        for k, v in filter.items()
                    ):
                        continue
                batch_candidates.append((int(cid), float(dist)))

            top_k_candidates.extend(batch_candidates)
            top_k_candidates.sort(key=lambda x: x[1])
            top_k_candidates = top_k_candidates[:k]

        return top_k_candidates

    def delete_by_ids(self, ids: Iterable[int]) -> None:
        ids = list(ids)
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        params = tuple(ids)
        with self.conn:
            self.conn.execute(
                f"DELETE FROM {self._table_name} WHERE id IN ({placeholders})",
                params,
            )
            self.conn.execute(
                f"DELETE FROM {self._vec_table_name} WHERE rowid IN ({placeholders})", params
            )
            self._delete_fts_rows(ids)
        self.conn.execute("VACUUM")

    def remove_texts(
        self,
        texts: Sequence[str] | None = None,
        filter: dict[str, Any] | None = None,
    ) -> int:
        if texts is None and filter is None:
            raise ValueError("Must provide either texts or filter to remove")

        ids_to_delete: list[int] = []

        if texts:
            placeholders = ",".join("?" for _ in texts)
            rows = self.conn.execute(
                f"SELECT id FROM {self._table_name} WHERE text IN ({placeholders})",
                tuple(texts),
            ).fetchall()
            ids_to_delete.extend(r[0] for r in rows)

        if filter:
            filter_clause, filter_params = self._build_filter_clause(filter)
            filter_clause = filter_clause.replace("AND ", "", 1)
            where_clause = f"WHERE {filter_clause}" if filter_clause else ""
            rows = self.conn.execute(
                f"SELECT id FROM {self._table_name} {where_clause}",
                tuple(filter_params),
            ).fetchall()
            ids_to_delete.extend(r[0] for r in rows)

        unique_ids = list(set(ids_to_delete))
        if unique_ids:
            self.delete_by_ids(unique_ids)

        return len(unique_ids)


class VectorDB:
    """
    Dead-simple local vector database powered by sqlite-vec.
    One SQLite file = multiple collections. Chroma-style API with quantization.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        quantization: Quantization = Quantization.FLOAT,
    ):
        self.path = str(path)
        self.distance_strategy = distance_strategy
        self.quantization = quantization

        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.enable_load_extension(True)
        try:
            sqlite_vec.load(self.conn)
            self._extension_available = True
        except sqlite3.OperationalError:
            self._extension_available = False

        self.conn.enable_load_extension(False)

    def collection(
        self,
        name: str,
        distance_strategy: DistanceStrategy | None = None,
        quantization: Quantization | None = None,
    ) -> VectorCollection:
        """
        Get or create a named collection.

        Args:
            name: Collection name (alphanumeric + underscores).
            distance_strategy: Override default distance strategy.
            quantization: Override default quantization.

        Returns:
            VectorCollection instance.
        """
        return VectorCollection(
            self.conn,
            name,
            distance_strategy or self.distance_strategy,
            quantization or self.quantization,
        )

    # ------------------------------------------------------------------ #
    # Integrations
    # ------------------------------------------------------------------ #
    def as_langchain(
        self, embeddings: Embeddings | None = None, collection_name: str = "default"
    ) -> TinyVecDBVectorStore:
        """
        Return a LangChain-compatible vector store interface.

        Args:
            embeddings: LangChain Embeddings model (optional).
            collection_name: Name of the collection to use.

        Returns:
            TinyVecDBVectorStore instance.
        """
        from .integrations.langchain import TinyVecDBVectorStore

        return TinyVecDBVectorStore(
            db_path=self.path, embedding=embeddings, collection_name=collection_name
        )

    def as_llama_index(self, collection_name: str = "default") -> TinyVecDBLlamaStore:
        """
        Return a LlamaIndex-compatible vector store interface.

        Args:
            collection_name: Name of the collection to use.

        Returns:
            TinyVecDBLlamaStore instance.
        """
        from .integrations.llamaindex import TinyVecDBLlamaStore

        return TinyVecDBLlamaStore(db_path=self.path, collection_name=collection_name)

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
