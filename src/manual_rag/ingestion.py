"""Document extraction and structure aware chunking for instruction manuals.

Every document becomes a list of {"page": int, "text": str} Markdown pages,
and chunk_pages() turns those into chunks that carry the manual name, page,
section and line range so answers can always be cited.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

PYMUPDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = set(PYMUPDF_EXTENSIONS)

# "1.2 Safety" or "3.1.4 Cleaning the Filter": at least two numeric levels and a
# capitalised title, so instruction steps like "1. Press Start." are NOT headings.
_NUMBERED_SECTION = re.compile(r"^\d+(\.\d+)+\.?\s+[A-Z]")
# A short line that is entirely bold, e.g. "**Care and Cleaning**".
_BOLD_LINE = re.compile(r"^\*\*[^*]{2,70}\*\*$")
# Markup that pymupdf4llm adds: picture text comments, <br>, and simple style tags.
_HTML_COMMENT = re.compile(r"<!--.*?-->")
_HTML_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG = re.compile(r"</?(?:u|b|i|em|strong|sup|sub|span)\b[^>]*>", re.IGNORECASE)

_EN_STOPWORDS = {
    "the", "and", "to", "of", "a", "is", "in", "for", "with", "on", "your",
    "you", "be", "or", "not", "if", "this", "when", "use", "it", "are", "from",
}
_ES_STOPWORDS = {
    "el", "la", "los", "las", "de", "del", "que", "y", "en", "un", "una", "por",
    "para", "con", "no", "se", "su", "es", "al", "lo", "como", "más", "o",
}
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_MIN_WORDS_FOR_LANG_DETECT = 8


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    """A retrievable unit of manual text plus its location metadata."""

    page: int
    section: str
    start_line: int
    end_line: int
    content: str
    source: str           # manual identifier (file name)
    doc_hash: str = ""    # hash of the source file, used to skip unchanged manuals

    @property
    def chunk_id(self) -> str:
        """Deterministic id, unique across documents (safe to re-index)."""
        key = f"{self.source}|{self.page}|{self.start_line}|{self.end_line}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedder. Prefixing the section title gives short
        chunks (like a lone step list) context about what they belong to."""
        if self.section and self.section != "General":
            return f"{self.section}\n{self.content}"
        return self.content

    def to_metadata(self) -> dict[str, Any]:
        """Metadata stored alongside the vector in the vector database."""
        meta = asdict(self)
        meta.pop("content")
        return meta


# --------------------------------------------------------------------------- #
# Line cleaning, headings, language
# --------------------------------------------------------------------------- #
def clean_line(line: str) -> str:
    """Remove extraction markup but keep the words (e.g. OCR'd picture labels)."""
    line = _HTML_COMMENT.sub(" ", line)
    line = _HTML_BR.sub(" ", line)
    line = _HTML_TAG.sub("", line)
    return re.sub(r"[ \t]{2,}", " ", line).strip()


def is_header(line: str) -> bool:
    """Return True if a cleaned Markdown line looks like a section heading."""
    if line.startswith("#"):
        return True
    if _BOLD_LINE.match(line) and not line.rstrip("*").rstrip().endswith((".", ":", ",")):
        return True
    return len(line) < 80 and bool(_NUMBERED_SECTION.match(line))


def clean_section_title(line: str) -> str:
    """Strip Markdown heading and emphasis markers from a heading line."""
    return line.lstrip("#").strip().strip("*_").strip() or "General"


def _heuristic_language(text: str) -> str | None:
    """Fallback detector (English vs Spanish only) when langdetect is missing."""
    words = _WORD.findall(text.lower())
    en = sum(w in _EN_STOPWORDS for w in words)
    es = sum(w in _ES_STOPWORDS for w in words)
    if es >= 3 and es > en:
        return "es"
    if en >= 3 and en >= es:
        return "en"
    return None


def detect_language(text: str) -> str | None:
    """ISO 639-1 code, or None when text is too short to classify
    (button labels, part numbers, tables)."""
    if len(_WORD.findall(text)) < _MIN_WORDS_FOR_LANG_DETECT:
        return None
    try:
        from langdetect import DetectorFactory, LangDetectException, detect

        DetectorFactory.seed = 0  # deterministic results
        try:
            return detect(text)
        except LangDetectException:
            return None
    except ImportError:
        return _heuristic_language(text)


def keep_chunk_language(text: str, keep_languages: Sequence[str] | None) -> bool:
    """True if a chunk passes the language filter. Unclassifiable text is kept."""
    if not keep_languages:
        return True
    lang = detect_language(text)
    return lang is None or lang in set(keep_languages)


# --------------------------------------------------------------------------- #
# Chunking (pure function, easy to unit test)
# --------------------------------------------------------------------------- #
def chunk_pages(
    pages: Iterable[dict[str, Any]],
    source: str,
    max_lines_per_chunk: int = 30,
    min_chunk_chars: int = 20,
    keep_languages: Sequence[str] | None = None,
    doc_hash: str = "",
) -> list[Chunk]:
    """Split Markdown pages into section aware chunks.

    A new chunk starts at every heading and whenever the current chunk reaches
    max_lines_per_chunk non-empty lines. Line numbers refer to the page's
    Markdown lines. Sections carry across page breaks.
    """
    chunks: list[Chunk] = []
    current_section = "General"

    for page in pages:
        page_num = int(page["page"])
        text = page.get("text") or ""
        if not text.strip():
            continue

        buffer: list[str] = []
        start_line = 1
        last_line = 0
        line_number = 0

        def flush() -> None:
            content = "\n".join(buffer).strip()
            if len(content) >= min_chunk_chars:
                chunks.append(
                    Chunk(page_num, current_section, start_line, last_line, content, source, doc_hash)
                )
            buffer.clear()

        for raw_line in text.split("\n"):
            line_number += 1
            line = clean_line(raw_line)
            if not line:
                continue

            if is_header(line):
                if buffer:
                    flush()
                current_section = clean_section_title(line)
                continue

            if not buffer:
                start_line = line_number
            buffer.append(line)
            last_line = line_number

            if len(buffer) >= max_lines_per_chunk:
                flush()

        if buffer:
            flush()

    if keep_languages:
        before = len(chunks)
        chunks = [c for c in chunks if keep_chunk_language(c.embedding_text, keep_languages)]
        logger.info("%s: language filter kept %d of %d chunks", source, len(chunks), before)

    return chunks


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def file_hash(path: str | Path) -> str:
    """SHA256 of a file's bytes (first 16 hex chars)."""
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
    """Convert a manual into per-page Markdown, dispatching on file type."""
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


def load_and_chunk(path: str | Path, chunk_cfg: dict[str, Any]) -> list[Chunk]:
    """Extract one manual and chunk it using values from the config."""
    path = Path(path)
    chunks = chunk_pages(
        extract_pages(path),
        source=path.name,
        max_lines_per_chunk=chunk_cfg.get("max_lines_per_chunk", 30),
        min_chunk_chars=chunk_cfg.get("min_chunk_chars", 20),
        keep_languages=chunk_cfg.get("keep_languages"),
        doc_hash=file_hash(path),
    )
    logger.info("%s: %d chunks", path.name, len(chunks))
    return chunks
