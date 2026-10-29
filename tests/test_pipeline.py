"""Tests for pipeline orchestration, using fakes (no database, model or API)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from manual_rag import pipeline
from manual_rag.config import Config
from manual_rag.ingestion import Chunk
from manual_rag.ui import DocInfo


def test_latest_manual_is_the_newest_upload():
    metas = [
        {"source": "ge_microwave_jvm3160.pdf", "ingested_at": 100.0},
        {"source": "ge_microwave_jvm3160.pdf", "ingested_at": 100.0},
        {"source": "mr_coffee_nl4.pdf", "ingested_at": 250.0},
    ]
    assert pipeline.latest_from_metadata(metas) == "mr_coffee_nl4.pdf"
    metas.append({"source": "router.pdf", "ingested_at": 300.0})
    assert pipeline.latest_from_metadata(metas) == "router.pdf"


def test_latest_manual_handles_old_chunks_and_empty_index():
    assert pipeline.latest_from_metadata([]) is None
    old = [{"source": "a.pdf"}, {"source": "b.pdf", "ingested_at": None}]  # indexed before timestamps
    assert pipeline.latest_from_metadata(old) in {"a.pdf", "b.pdf"}


def test_chunk_metadata_records_upload_time():
    chunk = Chunk(1, "S", 1, 2, "text", "a.pdf", ingested_at=123.0)
    assert chunk.to_metadata()["ingested_at"] == 123.0


def test_ingest_stamps_every_chunk_with_the_same_time(monkeypatch, tmp_path):
    manual = tmp_path / "kettle.txt"
    manual.write_text("x")
    indexed = []

    monkeypatch.setattr(pipeline, "discover_documents", lambda path: [manual])
    monkeypatch.setattr(pipeline, "_collection", lambda cfg, reset=False: object())
    monkeypatch.setattr(pipeline, "stored_hash", lambda col, name: None)
    monkeypatch.setattr(pipeline, "delete_source", lambda col, name: None)
    monkeypatch.setattr(pipeline, "_describer", lambda cfg: None)
    monkeypatch.setattr(pipeline, "_embedder", lambda cfg: SimpleNamespace())
    monkeypatch.setattr(pipeline, "load_and_chunk", lambda doc, cfg, describe=None: [
        Chunk(1, "S", 1, 2, "a", doc.name), Chunk(1, "S", 3, 4, "b", doc.name)])
    monkeypatch.setattr(pipeline, "index_chunks",
                        lambda col, emb, chunks: indexed.extend(chunks) or len(chunks))

    cfg = Config(raw={"paths": {"documents": "."}, "chunking": {}}, root=Path(tmp_path))
    report = pipeline.ingest(cfg)
    assert report.indexed == {"kettle.txt": 2}
    stamps = {c.ingested_at for c in indexed}
    assert len(stamps) == 1 and stamps.pop() > 0


# --- Ad hoc files (--file) --------------------------------------------------- #
def _cfg(tmp_path):
    return Config(raw={"paths": {"documents": "."}, "chunking": {},
                       "upload": {"max_pages": 50, "min_chars_per_page": 200}}, root=Path(tmp_path))


def _report(**kw):
    return SimpleNamespace(**{"indexed": {}, "skipped": [], "empty": [], "failed": {}, **kw})


def test_load_file_indexes_and_returns_the_name(monkeypatch, tmp_path):
    manual = tmp_path / "kettle.pdf"
    manual.write_text("x")
    calls = []
    monkeypatch.setattr(pipeline, "inspect_document", lambda p: DocInfo(pages=8, text_chars=12000))
    monkeypatch.setattr(pipeline, "ingest", lambda cfg, p: calls.append(p) or _report(indexed={"kettle.pdf": 9}))
    assert pipeline.load_file(_cfg(tmp_path), manual) == "kettle.pdf"
    assert calls == [manual]


def test_load_file_unchanged_manual_is_fine(monkeypatch, tmp_path):
    manual = tmp_path / "kettle.pdf"
    manual.write_text("x")
    monkeypatch.setattr(pipeline, "inspect_document", lambda p: DocInfo(pages=8, text_chars=12000))
    monkeypatch.setattr(pipeline, "ingest", lambda cfg, p: _report(skipped=["kettle.pdf"]))
    assert pipeline.load_file(_cfg(tmp_path), manual) == "kettle.pdf"


def test_load_file_rejects_bad_inputs(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(FileNotFoundError):
        pipeline.load_file(cfg, tmp_path / "missing.pdf")

    photo = tmp_path / "photo.jpg"
    photo.write_text("x")
    with pytest.raises(ValueError, match="Unsupported"):
        pipeline.load_file(cfg, photo)

    big = tmp_path / "car_manual.pdf"
    big.write_text("x")
    monkeypatch.setattr(pipeline, "inspect_document", lambda p: DocInfo(pages=400, text_chars=900000))
    with pytest.raises(ValueError, match="400 pages"):
        pipeline.load_file(cfg, big)


def test_load_file_reports_unreadable_manual(monkeypatch, tmp_path):
    manual = tmp_path / "scan.pdf"
    manual.write_text("x")
    monkeypatch.setattr(pipeline, "inspect_document", lambda p: DocInfo(pages=5, text_chars=5000))
    monkeypatch.setattr(pipeline, "ingest", lambda cfg, p: _report(empty=["scan.pdf"]))
    with pytest.raises(ValueError, match="No readable text"):
        pipeline.load_file(cfg := _cfg(tmp_path), manual)
