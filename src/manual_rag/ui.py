"""Helpers for the web app (no Gradio import, so they are easy to test)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .ingestion import SUPPORTED_EXTENSIONS
from .retrieval import Hit

CHARS_PER_PAGE_ESTIMATE = 3000  # for files without real pages (Word, text)


def safe_filename(name: str) -> str:
    """Make an uploaded file name safe and readable as a manual identifier.

    "My Router (v2) Manual.PDF" -> "my_router_v2_manual.pdf"
    """
    path = Path(name)
    stem = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_") or "manual"
    return f"{stem}{path.suffix.lower()}"


def is_supported(name: str) -> bool:
    """True if the file extension is one the pipeline can read."""
    return Path(name).suffix.lower() in SUPPORTED_EXTENSIONS


@dataclass
class DocInfo:
    """Quick facts about an upload, read without OCR so the check is fast."""

    pages: int
    text_chars: int  # characters of real (selectable) text

    @property
    def chars_per_page(self) -> float:
        return self.text_chars / max(self.pages, 1)


def inspect_document(path: str | Path) -> DocInfo:
    """Count pages and real text. PDF/EPUB/XPS use PyMuPDF; Word and text
    files have no fixed pages, so pages are estimated from their length."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in {".pdf", ".epub", ".xps"}:
        try:
            import pymupdf
        except ImportError:  # older PyMuPDF versions
            import fitz as pymupdf
        with pymupdf.open(str(path)) as doc:
            chars = sum(len(page.get_text().strip()) for page in doc)
            return DocInfo(pages=doc.page_count, text_chars=chars)
    if ext == ".docx":
        import docx

        chars = sum(len(p.text.strip()) for p in docx.Document(str(path)).paragraphs)
    else:
        chars = len(path.read_text(encoding="utf-8", errors="replace").strip())
    return DocInfo(pages=max(1, -(-chars // CHARS_PER_PAGE_ESTIMATE)), text_chars=chars)


def validate_upload(info: DocInfo, max_pages: int = 50, min_chars_per_page: int = 200) -> str | None:
    """Return a friendly reason to reject the upload, or None if it's fine."""
    if info.pages > max_pages:
        return (f"This manual has {info.pages} pages. The limit is {max_pages} pages, "
                "so please upload a shorter manual.")
    if info.text_chars == 0 or info.chars_per_page < min_chars_per_page:
        return ("This file is mostly images (very little readable text). Please upload a "
                "manual with written instructions, not a picture-only scan or diagram booklet.")
    return None


def format_sources(hits: Sequence[Hit], limit: int = 5, preview_chars: int = 600) -> str:
    """Markdown list of the chunks the answer was based on, each expandable."""
    if not hits:
        return "_No sources were retrieved._"
    parts = []
    for i, h in enumerate(hits[:limit], start=1):
        name = h.manual_title or h.source
        text = h.text if len(h.text) <= preview_chars else h.text[:preview_chars].rstrip() + " …"
        parts.append(
            f"<details><summary><b>{i}. {name}</b> · page {h.page} · {h.section} "
            f"· lines {h.start_line}-{h.end_line} · similarity {h.similarity:.2f}</summary>\n\n"
            f"{text}\n\n</details>"
        )
    return "\n\n".join(parts)


def format_ingest_report(report) -> str:
    """Human friendly summary of an IngestReport."""
    lines = []
    lines += [f"✅ Added **{name}** ({n} sections indexed)" for name, n in report.indexed.items()]
    lines += [f"ℹ️ **{name}** was already indexed and hasn't changed" for name in report.skipped]
    lines += [f"⚠️ **{name}**: no readable text found. If it's a scan, OCR (Tesseract) is needed."
              for name in report.empty]
    lines += [f"❌ **{name}** failed: {err}" for name, err in report.failed.items()]
    return "\n\n".join(lines) or "Nothing to add."
