"""
Embedding layer, built on LangChain's `Embeddings` interface
(langchain_core.embeddings.Embeddings).

Only real embedding models are offered now — no offline hashing fallback,
no OpenAI. Pick one of three HuggingFace models via `get_embedder(name=...)`:

  - "bge-m3"               -> BAAI/bge-m3               (strongest, largest, multilingual)
  - "qwen3-embedding-0.6b" -> Qwen/Qwen3-Embedding-0.6B (strong, mid-sized)
  - "bge-small-en-v1.5"    -> BAAI/bge-small-en-v1.5    (fastest, lightest — good default)

All three route through `langchain_huggingface.HuggingFaceEmbeddings`, so
the rest of the pipeline (chunking, Qdrant store, metrics) doesn't change
based on which one is selected — only the model name does. Downloading a
model's weights requires network access the first time it's used; after
that, sentence-transformers caches them locally.

Requires: pip install langchain-huggingface sentence-transformers
"""
from typing import Dict
from langchain_core.embeddings import Embeddings

EMBEDDING_MODELS: Dict[str, str] = {
    "bge-m3": "BAAI/bge-m3",
    "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
}


def get_embedder(name: str = "bge-small-en-v1.5", **kwargs) -> Embeddings:
    """Factory so pipeline configs can select an embedding model by name."""
    if name not in EMBEDDING_MODELS:
        raise ValueError(
            f"Unknown embedder: '{name}'. Choose one of: {', '.join(EMBEDDING_MODELS)}"
        )

    from langchain_huggingface import HuggingFaceEmbeddings  # requires network to download weights (first run only)

    kwargs.setdefault("model_name", EMBEDDING_MODELS[name])
    return HuggingFaceEmbeddings(**kwargs)
