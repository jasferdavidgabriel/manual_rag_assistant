"""Semantic retrieval over the indexed manuals."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .indexing import Embedder

logger = logging.getLogger(__name__)


@dataclass
class Hit:
    """One retrieved chunk with its location and similarity score."""

    text: str
    page: int
    section: str
    start_line: int
    end_line: int
    similarity: float  # 1 minus cosine distance, so higher is better
    source: str = ""   # manual the chunk came from


def parse_query_results(results: dict[str, Any]) -> list[Hit]:
    """Convert a raw Chroma query response into Hit objects.

    Chroma returns cosine distance (lower is better). The original notebook
    showed that as a "score", which reads backwards, so we convert it to
    similarity where higher means more relevant.
    """
    if not results.get("ids") or not results["ids"][0]:
        return []
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        if meta is None or doc is None:
            # Happens when another process re-indexed while this one held a
            # stale connection. Skip the entry instead of crashing.
            logger.warning("Skipping a search result with missing data (stale index?)")
            continue
        hits.append(
            Hit(
                text=doc,
                page=int(meta["page"]),
                section=str(meta.get("section", "")),
                start_line=int(meta.get("start_line", 0)),
                end_line=int(meta.get("end_line", 0)),
                similarity=round(1.0 - float(dist), 4),
                source=str(meta.get("source", "")),
            )
        )
    return hits


class Retriever:
    """Embeds a query and fetches the nearest chunks from the collection."""

    def __init__(self, collection: Any, embedder: Embedder) -> None:
        self.collection = collection
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 10, manual: str | None = None) -> list[Hit]:
        """Return the top_k most similar chunks for a query.

        Args:
            query: The user's question.
            top_k: Number of chunks to return.
            manual: Optional manual file name to search within.
        """
        if not query.strip():
            raise ValueError("Query must not be empty.")
        kwargs: dict[str, Any] = {
            "query_embeddings": self.embedder.encode([query]),
            "n_results": top_k,
        }
        if manual:
            kwargs["where"] = {"source": manual}
        return parse_query_results(self.collection.query(**kwargs))
