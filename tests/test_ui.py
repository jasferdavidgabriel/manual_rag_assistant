"""Tests for the web app's pure helpers (Gradio is not needed)."""

from types import SimpleNamespace

from manual_rag.retrieval import Hit
from manual_rag.ui import (
    ALL_MANUALS,
    format_ingest_report,
    format_sources,
    is_supported,
    manual_choices,
    safe_filename,
    to_manual_filter,
)


def test_safe_filename():
    assert safe_filename("My Router (v2) Manual.PDF") == "my_router_v2_manual.pdf"
    assert safe_filename("mr_coffee_nl4.pdf") == "mr_coffee_nl4.pdf"
    assert safe_filename("!!!.docx") == "manual.docx"


def test_is_supported():
    assert is_supported("a.PDF") and is_supported("b.docx") and is_supported("c.md")
    assert not is_supported("photo.jpg")


def test_manual_choices_and_filter():
    choices = manual_choices({"ge.pdf": 80, "coffee.pdf": 19})
    assert choices == [ALL_MANUALS, "ge.pdf", "coffee.pdf"]
    assert to_manual_filter(ALL_MANUALS) is None
    assert to_manual_filter(None) is None
    assert to_manual_filter("coffee.pdf") == "coffee.pdf"


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
