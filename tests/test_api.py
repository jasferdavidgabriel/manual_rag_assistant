"""Tests for the REST API.

The pipeline is mocked, so no models, ChromaDB or API key are needed.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from manual_rag import api  # noqa: E402
from manual_rag.generation import Answer  # noqa: E402
from manual_rag.retrieval import Hit  # noqa: E402


class FakeCfg:
    """Just enough of Config for the API: sections and paths."""

    def __init__(self, root: Path, upload: dict | None = None):
        self.root = root
        self.upload = {"max_pages": 50, "min_chars_per_page": 200, "max_mb": 25, **(upload or {})}

    def section(self, name):
        return {"upload": self.upload, "retrieval": {"display_k": 2}}.get(name, {})

    def path(self, name):
        return self.root / name


def make_hit(page: int) -> Hit:
    return Hit(text="Hold Cancel/Off for 3 seconds.", page=page, section="Control Lock-Out",
               start_line=10, end_line=12, similarity=0.8123456, source="microwave.pdf",
               manual_title="GE over the range microwave")


GOOD_DOC = SimpleNamespace(pages=12, text_chars=24000, chars_per_page=2000)


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_cfg", FakeCfg(tmp_path))
    monkeypatch.setattr(api, "_retriever", object())
    return TestClient(api.app)


# ---------- /health ----------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------- /ask ----------

def test_ask_answers_with_cited_sources(client, monkeypatch):
    calls = {}

    def fake_ask(cfg, question, manual=None, retriever=None):
        calls.update(question=question, manual=manual)
        return Answer(answer="Hold Cancel/Off for 3 seconds.", reasoning=None,
                      hits=[make_hit(1), make_hit(2), make_hit(3)], model="nemotron")

    monkeypatch.setattr(api.pipeline, "latest_manual", lambda cfg: "microwave.pdf")
    monkeypatch.setattr(api.pipeline, "ask", fake_ask)

    r = client.post("/ask", json={"question": "  How do I lock the controls?  "})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Hold Cancel/Off for 3 seconds."
    assert body["manual"] == "microwave.pdf"
    assert body["model"] == "nemotron"
    assert len(body["sources"]) == 2  # capped at display_k
    assert body["sources"][0]["page"] == 1
    assert body["sources"][0]["similarity"] == 0.8123
    assert calls == {"question": "How do I lock the controls?", "manual": "microwave.pdf"}


def test_ask_without_any_manual_returns_409(client, monkeypatch):
    monkeypatch.setattr(api.pipeline, "latest_manual", lambda cfg: None)
    r = client.post("/ask", json={"question": "How do I lock the controls?"})
    assert r.status_code == 409
    assert "/manuals" in r.json()["detail"]


def test_ask_rejects_blank_questions(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_ask_unknown_manual_returns_404(client, monkeypatch):
    monkeypatch.setattr(api.pipeline, "manuals", lambda cfg: {"microwave.pdf": 48})
    r = client.post("/ask", json={"question": "Warranty?", "manual": "toaster.pdf"})
    assert r.status_code == 404


def test_ask_model_failure_returns_502(client, monkeypatch):
    def boom(*args, **kwargs):
        raise TimeoutError("endpoint overloaded")

    monkeypatch.setattr(api.pipeline, "latest_manual", lambda cfg: "microwave.pdf")
    monkeypatch.setattr(api.pipeline, "ask", boom)
    r = client.post("/ask", json={"question": "Warranty?"})
    assert r.status_code == 502
    assert "TimeoutError" in r.json()["detail"]


# ---------- /manuals/current ----------

def test_current_manual(client, monkeypatch):
    monkeypatch.setattr(api.pipeline, "latest_manual", lambda cfg: "microwave.pdf")
    monkeypatch.setattr(api.pipeline, "manual_title", lambda cfg, name: "GE microwave")
    r = client.get("/manuals/current")
    assert r.status_code == 200
    assert r.json() == {"manual": "microwave.pdf", "title": "GE microwave"}


def test_current_manual_when_none_uploaded(client, monkeypatch):
    monkeypatch.setattr(api.pipeline, "latest_manual", lambda cfg: None)
    assert client.get("/manuals/current").status_code == 404


# ---------- /manuals (upload) ----------

def test_upload_indexes_manual(client, monkeypatch, tmp_path):
    loaded = []
    monkeypatch.setattr(api, "inspect_document", lambda path: GOOD_DOC)
    monkeypatch.setattr(api.pipeline, "load_file", lambda cfg, path: loaded.append(Path(path)))
    monkeypatch.setattr(api.pipeline, "manual_title", lambda cfg, name: "GE microwave")
    monkeypatch.setattr(api.pipeline, "manuals", lambda cfg: {"microwave.pdf": 48})

    r = client.post("/manuals", files={"file": ("microwave.pdf", b"%PDF fake", "application/pdf")})
    assert r.status_code == 201
    assert r.json() == {"manual": "microwave.pdf", "title": "GE microwave",
                        "pages": 12, "sections": 48}
    assert loaded == [tmp_path / "documents" / "microwave.pdf"]
    assert (tmp_path / "documents" / "microwave.pdf").read_bytes() == b"%PDF fake"
    assert api._retriever is None  # rebuilt on the next question


def test_upload_rejects_unsupported_type(client, tmp_path):
    r = client.post("/manuals", files={"file": ("setup.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 415
    assert not (tmp_path / "documents").exists()


def test_upload_rejects_image_heavy_manual(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "inspect_document",
                        lambda path: SimpleNamespace(pages=10, text_chars=50, chars_per_page=5))
    r = client.post("/manuals", files={"file": ("scan.pdf", b"%PDF fake", "application/pdf")})
    assert r.status_code == 422
    assert "mostly images" in r.json()["detail"]
    assert not (tmp_path / "documents" / "scan.pdf").exists()


def test_upload_rejects_files_over_size_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_cfg", FakeCfg(tmp_path, {"max_mb": 1}))
    client = TestClient(api.app)
    big = b"x" * (1024 * 1024 + 1)
    r = client.post("/manuals", files={"file": ("big.pdf", big, "application/pdf")})
    assert r.status_code == 413


def test_upload_cleans_up_when_indexing_fails(client, monkeypatch, tmp_path):
    def fail(cfg, path):
        raise RuntimeError("Couldn't read broken.pdf: bad xref table")

    monkeypatch.setattr(api, "inspect_document", lambda path: GOOD_DOC)
    monkeypatch.setattr(api.pipeline, "load_file", fail)
    r = client.post("/manuals", files={"file": ("broken.pdf", b"%PDF fake", "application/pdf")})
    assert r.status_code == 422
    assert "bad xref" in r.json()["detail"]
    assert not (tmp_path / "documents" / "broken.pdf").exists()
