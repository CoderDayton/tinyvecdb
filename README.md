# TinyVecDB

[![CI](https://github.com/coderdayton/tinyvecdb/actions/workflows/ci.yml/badge.svg)](https://github.com/coderdayton/tinyvecdb/actions)
[![PyPI](https://img.shields.io/pypi/v/tinyvecdb?color=blue)](https://pypi.org/project/tinyvecdb/)
[![License: MIT](https://img.shields.io/github/license/coderdayton/tinyvecdb)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/coderdayton/tinyvecdb?style=social)](https://github.com/coderdayton/tinyvecdb)

**The dead-simple, local-first vector database.**

TinyVecDB brings **Chroma-like simplicity** to a single **SQLite file**. Built on `sqlite-vec`, it offers high-performance vector search, quantization, and zero infrastructure headaches. Perfect for local RAG, offline agents, and indie hackers.

---

## 🚀 Why TinyVecDB?

- **Zero Infra**: Just a `.db` file. No Docker, no Redis, no cloud bills.
- **Fast**: ~2ms queries on consumer hardware.
- **Efficient**: 4x-32x storage reduction with INT8/BIT quantization.
- **Universal**: Runs anywhere SQLite runs (Linux, macOS, Windows, WASM).
- **Batteries Included**: Optional FastAPI embeddings server & LangChain/LlamaIndex integrations.

## 📦 Installation

```bash
# Core only (lightweight)
pip install tinyvecdb

# With server & local models
pip install "tinyvecdb[server]"
```

## ⚡ Quickstart

TinyVecDB is **just a vector storage layer**—it doesn't include an LLM or generate embeddings for you. You can use it in three ways:

### Option 1: With OpenAI (Simplest)

```python
from tinyvecdb import VectorDB
from openai import OpenAI

# Initialize TinyVecDB
db = VectorDB("knowledge.db")
collection = db.collection("knowledge_base")

# Generate embeddings using OpenAI
client = OpenAI()
texts = ["Paris is the capital of France.", "The mitochondria is the powerhouse of the cell."]

embeddings = [
    client.embeddings.create(model="text-embedding-3-small", input=t).data[0].embedding
    for t in texts
]

# Store in TinyVecDB
collection.add_texts(
    texts=texts,
    embeddings=embeddings,
    metadatas=[{"category": "geography"}, {"category": "biology"}]
)

# Search (you still need to embed your query)
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input="capital of France"
).data[0].embedding

results = collection.similarity_search(query_embedding, k=1)
print(f"Result: {results[0][0].page_content}")

# Search with metadata filter
geo_results = collection.similarity_search(
    query_embedding,
    k=3,
    filter={"category": "geography"},
)
print(f"Geography results: {len(geo_results)}")
```

### Option 2: Fully Local (with `[server]` extras)

```bash
# Install with local embedding support
pip install "tinyvecdb[server]"
```

```python
from tinyvecdb import VectorDB
from tinyvecdb.embeddings.models import embed_texts

db = VectorDB("local.db")
collection = db.collection("local_docs")

texts = ["Paris is the capital of France.", "The mitochondria is the powerhouse of the cell."]

# Generate embeddings locally using HuggingFace models
embeddings = embed_texts(texts)

collection.add_texts(
    texts=texts,
    embeddings=embeddings,
    metadatas=[{"category": "geography"}, {"category": "biology"}]
)

# Search
query_embeddings = embed_texts(["capital of France"])
results = collection.similarity_search(query_embeddings[0], k=1)
print(f"Result: {results[0][0].page_content}")

# Hybrid BM25 + Vector search
hybrid_results = collection.hybrid_search("yellow fruit", k=2)
for doc, score in hybrid_results:
    print(doc.page_content, score)
```

**Local Embeddings Server** (Optional):

If you prefer an OpenAI-compatible API running 100% locally:

```bash
tinyvecdb-server --port 8000
# Now use http://localhost:8000/v1/embeddings with any OpenAI-compatible client
```

See the [Setup Guide](ENV_SETUP.md) for advanced configuration, including model registry locking, throughput limits, and API key authentication.

### Option 3: With LangChain or LlamaIndex

TinyVecDB integrates directly with popular frameworks:

```python
from tinyvecdb.integrations.langchain import TinyVecDBVectorStore
from langchain_openai import OpenAIEmbeddings

# Use LangChain's embedding models
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = TinyVecDBVectorStore(db_path="langchain.db", embedding=embeddings)

# Now use standard LangChain methods
store.add_texts(["Paris is the capital of France."])
results = store.similarity_search("capital of France", k=1)
```

For complete RAG workflows with Ollama, LangChain, and LlamaIndex, see the **[Examples](https://coderdayton.github.io/tinyvecdb/examples/)** page.

## Multi-Collection Support

TinyVecDB supports multiple collections within a single database file.

```python
# Create/Get a named collection
users = db.collection("users", quantization=Quantization.INT8)
products = db.collection("products", quantization=Quantization.FLOAT)

users.add_texts(["User A"], embeddings=[[0.1]*384])
products.add_texts(["Product X"], embeddings=[[0.9]*384])
```

## 🔍 Search Features

### Keyword & Hybrid Search

TinyVecDB ships with first-class sparse search. Call `keyword_search("exact term")` for BM25/FTS-only scoring or `hybrid_search("concept", k=10)` to blend BM25 scores with vector distances using Reciprocal Rank Fusion—all inside a single SQLite file. Metadata filters still work, and no extra services are required.

> **Note:** If you compile SQLite yourself, ensure `-DSQLITE_ENABLE_FTS5` is enabled so the FTS virtual table is available.

Already have your own embeddings? Pass `query_vector=my_vector` to `hybrid_search` to reuse them (saves a second embedding pass and keeps dimensions aligned).

> Both integrations surface `keyword_search(...)` and `hybrid_search(...)`, so you can rerank sparse + dense results without leaving your favorite framework.

## 🛠️ Features

| Feature           | Status | Description                                                 |
| :---------------- | :----- | :---------------------------------------------------------- |
| **Storage**       | ✅     | Single SQLite file or in-memory.                            |
| **Collections**   | ✅     | Multiple named collections per database file.               |
| **Search**        | ✅     | Cosine, Euclidean, and IP distance metrics.                 |
| **Hybrid Search** | ✅     | BM25 keyword search + dense fusion via `hybrid_search`.     |
| **Quantization**  | ✅     | FLOAT32, INT8, and BIT (1-bit) support.                     |
| **Filtering**     | ✅     | Metadata filtering with SQL `WHERE` clauses.                |
| **Integrations**  | ✅     | LangChain/LlamaIndex shims expose keyword + hybrid helpers. |
| **Hardware**      | ✅     | Auto-detects CUDA/MPS/CPU for optimal batching.             |

## 📊 Benchmarks

_Tested on i9-13900K & RTX 4090 with `sqlite-vec` v0.1.6 (10k vectors, 384-dim)_

| Type      | Storage  | Insert Speed | Query Time (k=10) |
| :-------- | :------- | :----------- | :---------------- |
| **FLOAT** | 15.50 MB | 15,585 vec/s | 3.55 ms           |
| **INT8**  | 4.23 MB  | 27,893 vec/s | 3.93 ms           |
| **BIT**   | 0.95 MB  | 32,321 vec/s | 0.27 ms           |

## 📚 Documentation

- **[Setup Guide](ENV_SETUP.md)**: Configuration and environment variables.
- **[API Reference](api/core/)**: Full class and method documentation.
- **[Benchmarks](benchmarks/)**: Performance comparisons.
- **[Examples](examples/)**: RAG notebooks and integration demos.
- **[Contributing](CONTRIBUTING.md)**: How to build and test.

## 🗺️ Roadmap

- [x] Hybrid Search (BM25 + Vector)
- [x] Multi-collection support
- [ ] HNSW Indexing (via sqlite-vec updates)
- [ ] Built-in Encryption (SQLCipher)

## 🤝 Contributing

I welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on setting up your development environment.

## ❤️ Sponsors

TinyVecDB is an independent, open‑source project, built and maintained transparently. If it’s useful to you and you’d like to support it, there are a few simple ways to contribute.

### Company Sponsors

_Become my first company sponsor! [Support me on GitHub](https://github.com/sponsors/coderdayton)_

### Individual Supporters

_Join the list of supporters! [Support me on GitHub](https://github.com/sponsors/coderdayton)_

<!-- sponsors --><!-- sponsors -->

**Want to support the project?**

- 🍵 [Buy me a coffee](https://www.buymeacoffee.com/coderdayton) (One-time donation)
- 💎 [Get the Pro Pack](https://tinyvecdb.lemonsqueezy.com/) (Deployment templates & production recipes)
- 💖 [GitHub Sponsors](https://github.com/sponsors/coderdayton) (Monthly support)

## 📄 License

[MIT](LICENSE)
