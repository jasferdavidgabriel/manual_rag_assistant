"""Pure helpers for the web app (no Gradio import, so they are easy to test)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from .ingestion import SUPPORTED_EXTENSIONS
from .retrieval import Hit

ALL_MANUALS = "All manuals"


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


def manual_choices(manuals: dict[str, int]) -> list[str]:
    """Dropdown options: "All manuals" first, then each indexed manual."""
    return [ALL_MANUALS, *manuals.keys()]


def to_manual_filter(choice: str | None) -> str | None:
    """Convert the dropdown value into the pipeline's manual filter."""
    return None if not choice or choice == ALL_MANUALS else choice


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
