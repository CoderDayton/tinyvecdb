from __future__ import annotations

import sqlite3
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

from .types import Document, DistanceStrategy, Quantization
from .utils import _import_optional
from .quantization import QuantizationStrategy
from .search import SearchEngine

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from .integrations.langchain import SimpleVecDBVectorStore
    from .integrations.llamaindex import SimpleVecDBLlamaStore


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
        self._quantizer = QuantizationStrategy(quantization)

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

        # Initialize SearchEngine after table names are set
        self._search = SearchEngine(
            conn=self.conn,
            table_name=self._table_name,
            vec_table_name=self._vec_table_name,
            fts_table_name=self._fts_table_name,
            fts_enabled=False,  # Will be updated in _ensure_fts_table
            distance_strategy=self.distance_strategy,
            quantization=self.quantization,
            quantizer=self._quantizer,
            dim_getter=lambda: self._dim,
        )

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
            self._search._fts_enabled = True
        except sqlite3.OperationalError:
            self._fts_enabled = False
            self._search._fts_enabled = False

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
                from simplevecdb.embeddings.models import embed_texts

                embed_func = embed_texts
            except Exception as e:
                raise ValueError(
                    "No embeddings provided and local embedder failed – install with [server] extra"
                ) from e

        from simplevecdb import config

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

            serialized = [self._quantizer.serialize(vec) for vec in emb_np]

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

    def similarity_search(
        self,
        query: str | Sequence[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        return self._search.similarity_search(
            query, k, filter, filter_builder=self._build_filter_clause
        )

    def keyword_search(
        self, query: str, k: int = 5, filter: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        return self._search.keyword_search(
            query, k, filter, filter_builder=self._build_filter_clause
        )

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
        return self._search.hybrid_search(
            query,
            k,
            filter,
            query_vector=query_vector,
            vector_k=vector_k,
            keyword_k=keyword_k,
            rrf_k=rrf_k,
            filter_builder=self._build_filter_clause,
        )

    def max_marginal_relevance_search(
        self,
        query: str | Sequence[float],
        k: int = 5,
        fetch_k: int = 20,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        return self._search.max_marginal_relevance_search(
            query, k, fetch_k, filter, filter_builder=self._build_filter_clause
        )

    def _brute_force_search(
        self,
        query_vec: np.ndarray,
        k: int,
        filter: dict[str, Any] | None,
    ) -> list[tuple[int, float]]:
        return self._search._brute_force_search(
            query_vec, k, filter, self._build_filter_clause, get_optimal_batch_size()
        )

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
                f"DELETE FROM {self._vec_table_name} WHERE rowid IN ({placeholders})",
                params,
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
    ) -> SimpleVecDBVectorStore:
        """
        Return a LangChain-compatible vector store interface.

        Args:
            embeddings: LangChain Embeddings model (optional).
            collection_name: Name of the collection to use.

        Returns:
            SimpleVecDBVectorStore instance.
        """
        from .integrations.langchain import SimpleVecDBVectorStore

        return SimpleVecDBVectorStore(
            db_path=self.path, embedding=embeddings, collection_name=collection_name
        )

    def as_llama_index(self, collection_name: str = "default") -> SimpleVecDBLlamaStore:
        """
        Return a LlamaIndex-compatible vector store interface.

        Args:
            collection_name: Name of the collection to use.

        Returns:
            SimpleVecDBLlamaStore instance.
        """
        from .integrations.llamaindex import SimpleVecDBLlamaStore

        return SimpleVecDBLlamaStore(db_path=self.path, collection_name=collection_name)

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
