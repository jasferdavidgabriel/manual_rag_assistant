"""High level orchestration: ingest, search, answer, and evaluate."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .evaluation import compute_metrics, find_misses, load_eval_set, plot_hit_at_k
from .generation import Answer, LLMClient
from .indexing import (
    Embedder,
    delete_source,
    get_collection,
    index_chunks,
    list_sources,
    stored_hash,
)
from .ingestion import discover_documents, file_hash, load_and_chunk
from .retrieval import Hit, Retriever
from .ui import inspect_document, is_supported, validate_upload

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


def _describer(cfg: Config):
    """Return a function that names a manual's product with the LLM, or None
    if no API key is set (titles then fall back to the file name)."""
    if not cfg.section("chunking").get("contextual_headers", True):
        return None
    try:
        client = LLMClient(cfg.section("llm"))
    except EnvironmentError:
        logger.warning("No LLM key set; manual titles will come from file names.")
        return None
    return client.describe_manual


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
    describer = _describer(cfg)

    for doc in docs:
        try:
            if not force and stored_hash(collection, doc.name) == file_hash(doc):
                report.skipped.append(doc.name)
                continue
            chunks = load_and_chunk(doc, cfg.section("chunking"), describe=describer)
            delete_source(collection, doc.name)
            if not chunks:
                report.empty.append(doc.name)
                continue
            stamp = time.time()
            for chunk in chunks:
                chunk.ingested_at = stamp
            embedder = embedder or _embedder(cfg)
            report.indexed[doc.name] = index_chunks(collection, embedder, chunks)
        except Exception as exc:  # keep going, report at the end
            logger.exception("Failed to ingest %s", doc.name)
            report.failed[doc.name] = f"{type(exc).__name__}: {exc}"
    return report


def manual_title(cfg: Config, source: str) -> str:
    """The product title stored with a manual's chunks ("" if not indexed)."""
    res = _collection(cfg).get(where={"source": source}, limit=1, include=["metadatas"])
    metas = res.get("metadatas") or []
    return str(metas[0].get("manual_title", "")) if metas else ""


def load_file(cfg: Config, path: str | Path) -> str:
    """Ad hoc use: check a manual file, index it if new or changed, return its name.

    Applies the same limits as the web app (page count, mostly-images check).
    Unchanged files are skipped by ingest, so asking again is fast.

    Raises:
        FileNotFoundError, ValueError, RuntimeError with a message meant for users.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No file at {path}")
    if not is_supported(path.name):
        raise ValueError("Unsupported file type. Use PDF, Word (.docx), EPUB, XPS, .txt or .md.")

    limits = cfg.section("upload")
    problem = validate_upload(
        inspect_document(path),
        limits.get("max_pages", 50),
        limits.get("min_chars_per_page", 200),
    )
    if problem:
        raise ValueError(problem)

    report = ingest(cfg, path)
    if path.name in report.failed:
        raise RuntimeError(f"Couldn't read {path.name}: {report.failed[path.name]}")
    if path.name in report.empty:
        raise ValueError("No readable text was found in this manual.")
    return path.name


def latest_from_metadata(metadatas: list[dict]) -> str | None:
    """Name of the most recently indexed manual in a list of chunk metadata."""
    newest: dict[str, float] = {}
    for meta in metadatas:
        source = meta.get("source")
        if source:
            newest[source] = max(newest.get(source, 0.0), float(meta.get("ingested_at") or 0.0))
    if not newest:
        return None
    return max(newest, key=lambda name: (newest[name], name))


def latest_manual(cfg: Config) -> str | None:
    """The manual uploaded most recently: the default for ask, chat and search."""
    return latest_from_metadata(_collection(cfg).get(include=["metadatas"]).get("metadatas") or [])


def manuals(cfg: Config) -> dict[str, int]:
    """List indexed manuals with their chunk counts."""
    return list_sources(_collection(cfg))


def build_retriever(cfg: Config) -> Retriever:
    """Create a retriever over the persisted collection."""
    collection = _collection(cfg)
    if collection.count() == 0:
        raise RuntimeError("Vector store is empty. Run `python main.py ingest` first.")
    return Retriever(collection, _embedder(cfg))


def search(
    cfg: Config, query: str, manual: str | None = None, retriever: Retriever | None = None
) -> list[Hit]:
    """Retrieve the top chunks for a query (no LLM involved)."""
    retriever = retriever or build_retriever(cfg)
    return retriever.retrieve(query, top_k=cfg.section("retrieval").get("top_k", 10), manual=manual)


def ask(
    cfg: Config, question: str, manual: str | None = None, retriever: Retriever | None = None
) -> Answer:
    """Retrieve context for a question and generate a cited answer."""
    hits = search(cfg, question, manual=manual, retriever=retriever)
    if not hits:
        return Answer(answer="No matching content found. Check the manual name.", reasoning=None, hits=[])
    return LLMClient(cfg.section("llm")).generate(question, hits)


def evaluate(cfg: Config, k_max: int = 10) -> dict:
    """Score retrieval on the labelled question set, in two modes.

    within_manual: each question is searched inside its own manual (easy).
    all_manuals:   no filter; a hit needs the right manual AND page (hard).
                   This shows whether retrieval can tell products apart.

    Questions for manuals that aren't indexed are skipped. Saves
    reports/retrieval_metrics.json and a Hit@k chart for each mode.
    """
    retriever = build_retriever(cfg)
    indexed = set(list_sources(retriever.collection))
    records = [r for r in load_eval_set(cfg.path("eval_set")) if r.get("manual", "") in indexed]
    if not records:
        raise RuntimeError("No eval questions match the indexed manuals.")

    ks = [k for k in (1, 3, 5, 10) if k <= k_max]
    within, across = [], []
    for rec in records:
        scoped = retriever.retrieve(rec["question"], top_k=k_max, manual=rec["manual"])
        within.append({**rec, "retrieved_pages": [h.page for h in scoped]})

        unscoped = retriever.retrieve(rec["question"], top_k=k_max)
        across.append({
            **rec,
            "retrieved_pages": [(h.source, h.page) for h in unscoped],
            "expected_pages": [(rec["manual"], p) for p in rec["expected_pages"]],
        })

    def score(rows: list[dict]) -> dict:
        return compute_metrics(
            [r["retrieved_pages"] for r in rows], [r["expected_pages"] for r in rows], ks=ks
        )

    result = {
        "manuals": sorted(indexed),
        "within_manual": {"metrics": score(within), "misses": find_misses(within), "per_question": within},
        "all_manuals": {"metrics": score(across), "misses": find_misses(across), "per_question": across},
    }

    results_dir = cfg.path("results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "retrieval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    figures = cfg.path("figures_dir")
    plot_hit_at_k(result["within_manual"]["metrics"], figures / "retrieval_hit_at_k.png")
    plot_hit_at_k(result["all_manuals"]["metrics"], figures / "retrieval_hit_at_k_all_manuals.png")
    logger.info("Saved results to %s", results_dir)
    return result
