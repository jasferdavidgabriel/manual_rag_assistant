"""Embedding model and persistent ChromaDB vector store."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from .ingestion import Chunk

logger = logging.getLogger(__name__)


def resolve_device(device: str = "auto") -> str:
    """Map "auto" to "cuda" when a GPU is available, else "cpu"."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Embedder:
    """Thin wrapper over SentenceTransformer that returns normalised vectors."""

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self.device = resolve_device(device)
        self.batch_size = batch_size
        logger.info("Loading embedding model %s on %s", model_name, self.device)
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> list[list[float]]:
        """Embed texts into L2 normalised vectors (ready for cosine similarity)."""
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return vectors.tolist()


def get_collection(
    persist_dir: str | Path, name: str, distance: str = "cosine", reset: bool = False
) -> Any:
    """Open (or create) a persistent Chroma collection.

    Args:
        persist_dir: Folder where Chroma stores its files.
        name: Collection name.
        distance: Distance metric for the HNSW index.
        reset: Delete the existing collection first.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    if reset:
        existing = {c.name if hasattr(c, "name") else c for c in client.list_collections()}
        if name in existing:
            client.delete_collection(name)
            logger.info("Deleted existing collection %s", name)
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": distance})


def index_chunks(
    collection: Any, embedder: Embedder, chunks: Sequence[Chunk], batch_size: int = 1000
) -> int:
    """Embed and upsert chunks in batches (big manuals exceed Chroma's per call
    limit). Returns the number of chunks indexed."""
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=embedder.encode([c.embedding_text for c in batch], show_progress=True),
            documents=[c.content for c in batch],
            metadatas=[c.to_metadata() for c in batch],
        )
    logger.info("Indexed %d chunks", len(chunks))
    return len(chunks)


def stored_hash(collection: Any, source: str) -> str | None:
    """File hash stored for a manual, or None if it isn't indexed yet."""
    res = collection.get(where={"source": source}, limit=1, include=["metadatas"])
    metas = res.get("metadatas") or []
    return metas[0].get("doc_hash") if metas else None


def delete_source(collection: Any, source: str) -> None:
    """Remove every chunk of one manual (done before re-indexing it)."""
    collection.delete(where={"source": source})


def list_sources(collection: Any) -> dict[str, int]:
    """Return {manual name: chunk count} for everything in the collection."""
    counts: dict[str, int] = {}
    for meta in collection.get(include=["metadatas"]).get("metadatas") or []:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return dict(sorted(counts.items()))
