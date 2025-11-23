# Examples

## RAG with LangChain

[View Notebook](https://github.com/coderdayton/simplevecdb/blob/main/examples/rag/langchain_rag.ipynb)

## RAG with LlamaIndex

[View Notebook](https://github.com/coderdayton/simplevecdb/blob/main/examples/rag/llama_rag.ipynb)

## RAG with Ollama LLM

[View Notebook](https://github.com/coderdayton/simplevecdb/blob/main/examples/rag/ollama_rag.ipynb)

## Keyword & Hybrid Search (SQLite-only)

```python
from simplevecdb import VectorDB

db = VectorDB("local.db")
collection = db.collection("default")
collection.add_texts(["banana is yellow", "grapes are purple"], embeddings=[[0.1,0.2],[0.3,0.4]])

# BM25
bm25 = collection.keyword_search("banana", k=1)

# Blend BM25 + vectors
hybrid = collection.hybrid_search("yellow fruit", k=2)
```

> Requires SQLite builds with FTS5 enabled (default in CPython).
