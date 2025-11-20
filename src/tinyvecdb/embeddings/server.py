from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Literal
import uvicorn

from .models import embed_texts, DEFAULT_MODEL

app = FastAPI(
    title="TinyVecDB Embeddings",
    description="OpenAI-compatible /v1/embeddings endpoint – 100% local",
    version="0.1.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
)


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str], List[int], List[List[int]]]
    model: Optional[str] = DEFAULT_MODEL
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict = Field(default_factory=lambda: {"prompt_tokens": 0, "total_tokens": 0})


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    if isinstance(request.input, str):
        texts = [request.input]
    elif isinstance(request.input, list) and all(
        isinstance(i, int) for i in request.input
    ):
        texts = [str(i) for i in request.input]  # token arrays – just stringify
    else:
        texts = [str(item) for item in request.input]

    try:
        embeddings = embed_texts(texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    # Fake token usage (optional – some tools expect it)
    total_tokens = sum(len(t.split()) for t in texts)

    return EmbeddingResponse(
        data=[
            EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)
        ],
        model=request.model or DEFAULT_MODEL,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    )


@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "tinyvecdb",
            }
        ],
        "object": "list",
    }


def run_server(host: str = "127.0.0.1", port: int = 53287):
    uvicorn.run(app, host=host, port=port, log_level="info")
