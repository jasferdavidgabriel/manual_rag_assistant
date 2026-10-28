"""High level orchestration. Phase 4: ingest a library of manuals."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .indexing import (
    Embedder,
    delete_source,
    get_collection,
    index_chunks,
    list_sources,
    stored_hash,
)
from .ingestion import discover_documents, file_hash, load_and_chunk

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    """Summary of one ingest run."""

    indexed: dict[str, int] = field(default_factory=dict)   # manual -> chunks
    skipped: list[str] = field(default_factory=list)        # unchanged manuals
    failed: dict[str, str] = field(default_factory=dict)    # manual -> error
    empty: list[str] = field(default_factory=list)          # no text extracted


def _embedder(cfg: Config) -> Embedder:
    e = cfg.section("embedding")
    return Embedder(e["model_name"], e.get("device", "auto"), e.get("batch_size", 32))


def _collection(cfg: Config, reset: bool = False):
    vs = cfg.section("vector_store")
    return get_collection(
        cfg.path("chroma_dir"), vs["collection_name"], vs.get("distance", "cosine"), reset=reset
    )


def ingest(
    cfg: Config, path: str | Path | None = None, reset: bool = False, force: bool = False
) -> IngestReport:
    """Index one manual or a whole folder of manuals.

    Unchanged manuals (same file hash) are skipped and changed ones are
    replaced cleanly, so this is safe to rerun. One bad file never stops the run.
    """
    docs = discover_documents(path or cfg.path("documents"))
    if not docs:
        raise FileNotFoundError("No supported manuals found in data/raw.")

    collection = _collection(cfg, reset=reset)
    embedder: Embedder | None = None  # only loaded if something needs indexing
    report = IngestReport()

    for doc in docs:
        try:
            if not force and stored_hash(collection, doc.name) == file_hash(doc):
                report.skipped.append(doc.name)
                continue
            chunks = load_and_chunk(doc, cfg.section("chunking"))
            delete_source(collection, doc.name)
            if not chunks:
                report.empty.append(doc.name)
                continue
            embedder = embedder or _embedder(cfg)
            report.indexed[doc.name] = index_chunks(collection, embedder, chunks)
        except Exception as exc:  # keep going, report at the end
            logger.exception("Failed to ingest %s", doc.name)
            report.failed[doc.name] = f"{type(exc).__name__}: {exc}"
    return report


def manuals(cfg: Config) -> dict[str, int]:
    """List indexed manuals with their chunk counts."""
    return list_sources(_collection(cfg))
