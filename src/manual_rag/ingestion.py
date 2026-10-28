"""Document extraction for instruction manuals.

Every document becomes a list of {"page": int, "text": str} Markdown pages.
Phase 2 supports PDF; more formats are added later via extract_pages().
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PYMUPDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = set(PYMUPDF_EXTENSIONS)


def file_hash(path: str | Path) -> str:
    """SHA256 of a file's bytes (first 16 hex chars). Used later to skip
    re-indexing manuals that haven't changed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _extract_pymupdf(path: Path) -> list[dict[str, Any]]:
    """Convert each PDF page to Markdown, keeping headings, lists and tables."""
    import pymupdf4llm  # heavy import, kept local so tests stay fast

    md_pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, write_images=False)
    return [{"page": p["metadata"]["page_number"], "text": p["text"]} for p in md_pages]


def extract_pages(path: str | Path) -> list[dict[str, Any]]:
    """Convert a manual into per-page Markdown, dispatching on file type.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = path.suffix.lower()
    logger.info("Extracting %s", path.name)

    if ext in PYMUPDF_EXTENSIONS:
        pages = _extract_pymupdf(path)
    else:
        raise ValueError(f"Unsupported file type {ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    if not any(p["text"].strip() for p in pages):
        logger.warning("%s produced no text. If it is scanned, install Tesseract for OCR.", path.name)
    return pages


def discover_documents(path: str | Path) -> list[Path]:
    """Return supported manuals at path (a single file, or a folder searched recursively)."""
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"No file or folder at {path}")
    return sorted(
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith(".")
    )
