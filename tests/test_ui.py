"""Tests for the web app's helpers (Gradio is not needed)."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from manual_rag.retrieval import Hit
from manual_rag.ui import (
    DocInfo,
    format_ingest_report,
    format_sources,
    inspect_document,
    is_supported,
    safe_filename,
    validate_upload,
)


def test_safe_filename():
    assert safe_filename("My Router (v2) Manual.PDF") == "my_router_v2_manual.pdf"
    assert safe_filename("mr_coffee_nl4.pdf") == "mr_coffee_nl4.pdf"
    assert safe_filename("!!!.docx") == "manual.docx"


def test_is_supported():
    assert is_supported("a.PDF") and is_supported("b.docx") and is_supported("c.md")
    assert not is_supported("photo.jpg")


# --- Upload limits ---------------------------------------------------------- #
def test_text_manual_within_limit_is_accepted():
    assert validate_upload(DocInfo(pages=10, text_chars=15000)) is None  # Mr. Coffee size
    assert validate_upload(DocInfo(pages=46, text_chars=90000)) is None  # GE size


def test_manual_over_page_limit_is_rejected():
    reason = validate_upload(DocInfo(pages=120, text_chars=300000), max_pages=50)
    assert "120 pages" in reason and "limit is 50" in reason
    assert validate_upload(DocInfo(pages=50, text_chars=100000), max_pages=50) is None  # exactly 50 is ok


def test_mostly_image_manual_is_rejected():
    assert "mostly images" in validate_upload(DocInfo(pages=20, text_chars=0))
    assert "mostly images" in validate_upload(DocInfo(pages=20, text_chars=1500))  # ~75 chars/page


def test_inspect_text_file_estimates_pages():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "kettle.txt"
        f.write_text("Descale monthly with vinegar. " * 400)  # ~12,000 chars
        info = inspect_document(f)
        assert info.pages == 4 and info.text_chars > 11000


def test_inspect_docx():
    docx = pytest.importorskip("docx")
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "kettle.docx"
        doc = docx.Document()
        doc.add_paragraph("Fill with water up to the MAX line before switching on.")
        doc.save(f)
        info = inspect_document(f)
        assert info.pages == 1 and info.text_chars > 40


# --- Formatting --------------------------------------------------------------- #
def test_format_sources_shows_location_and_text():
    hit = Hit("Use equal parts vinegar and hot water.", 8, "CLEANING THE DECANTER", 39, 50, 0.48,
              "mr_coffee_nl4.pdf", "Mr. Coffee NL4 4-cup coffeemaker")
    md = format_sources([hit])
    assert "Mr. Coffee NL4 4-cup coffeemaker" in md
    assert "page 8" in md and "lines 39-50" in md and "0.48" in md
    assert "vinegar" in md
    assert format_sources([]) == "_No sources were retrieved._"


def test_long_source_text_is_trimmed():
    hit = Hit("x" * 2000, 1, "S", 1, 2, 0.5, "m.pdf")
    assert "…" in format_sources([hit], preview_chars=100)


def test_format_ingest_report():
    report = SimpleNamespace(indexed={"a.pdf": 12}, skipped=["b.pdf"], empty=["c.pdf"],
                             failed={"d.pdf": "ValueError: bad"})
    text = format_ingest_report(report)
    assert "Added **a.pdf** (12 sections" in text
    assert "already indexed" in text and "no readable text" in text and "failed" in text
