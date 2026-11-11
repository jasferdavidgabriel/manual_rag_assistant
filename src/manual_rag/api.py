"""REST API for the Manual RAG Assistant.

A thin layer over :mod:`manual_rag.pipeline`, the same functions the Gradio
app and the CLI use, so every interface shares one code path.

Run it from the repo root:

    uvicorn manual_rag.api:app --app-dir src --reload

Then open http://127.0.0.1:8000/docs to try every endpoint in the browser.
"""

import logging
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import pipeline
from .config import Config, load_config
from .ui import inspect_document, is_supported, safe_filename, validate_upload

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Manual RAG Assistant API",
    version="1.0.0",
    description=(
        "Upload an instruction manual, then ask questions about it in plain English. "
        "Every answer cites the manual, page, section and lines it came from."
    ),
)

_cfg: Optional[Config] = None
_retriever = None
_lock = threading.Lock()  # uploads change the shared index, so one at a time

UNSUPPORTED = "Unsupported file type. Use PDF, Word (.docx), EPUB, XPS, .txt or .md."
NO_MANUAL = "No manual uploaded yet. POST one to /manuals first."


def get_cfg() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def get_retriever():
    """Shared search engine, built on first use and rebuilt after each upload."""
    global _retriever
    with _lock:
        if _retriever is None:
            _retriever = pipeline.build_retriever(get_cfg())
        return _retriever


# ---------- request and response shapes (these also drive the /docs page) ----------

class Source(BaseModel):
    manual: str
    manual_title: str
    page: int
    section: str
    start_line: int
    end_line: int
    similarity: float
    text: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000,
                          examples=["How do I lock the controls?"])
    manual: Optional[str] = Field(
        None, description="Manual to search. Defaults to the most recently uploaded one.")


class AskResponse(BaseModel):
    answer: str
    manual: str
    model: str
    sources: list[Source]


class ManualInfo(BaseModel):
    manual: str
    title: str
    pages: int
    sections: int


class CurrentManual(BaseModel):
    manual: str
    title: str


# ---------- endpoints ----------

@app.get("/health")
def health() -> dict:
    """Liveness check for Docker, CI and AWS. Touches no models."""
    return {"status": "ok"}


@app.post("/manuals", response_model=ManualInfo, status_code=201)
def upload_manual(file: UploadFile = File(...)) -> ManualInfo:
    """Upload a manual, check it, index it, and make it the current manual."""
    global _retriever
    cfg = get_cfg()
    limits = cfg.section("upload")

    name = safe_filename(Path(file.filename or "").name)
    if not name or not is_supported(name):
        raise HTTPException(415, UNSUPPORTED)

    # Check the upload in a temporary folder first, so a bad file never
    # replaces a good manual that's already in the documents folder.
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / name
        _save_upload(file, staged, int(limits.get("max_mb", 25)) * 1024 * 1024)
        try:
            info = inspect_document(staged)
        except Exception as exc:
            logger.info("Could not open %s: %s", name, exc)
            raise HTTPException(422, "Couldn't open that file. Is it a valid document?") from exc
        problem = validate_upload(info, limits.get("max_pages", 50),
                                  limits.get("min_chars_per_page", 200))
        if problem:
            raise HTTPException(422, problem)

        dest = cfg.path("documents") / name
        existed = dest.exists()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(staged, dest)

    with _lock:
        try:
            pipeline.load_file(cfg, dest)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            if not existed:
                dest.unlink(missing_ok=True)
            raise HTTPException(422, str(exc)) from exc
        _retriever = None  # the index changed

    logger.info("Indexed %s (%d pages)", name, info.pages)
    return ManualInfo(
        manual=name,
        title=pipeline.manual_title(cfg, name) or name,
        pages=info.pages,
        sections=pipeline.manuals(cfg).get(name, 0),
    )


@app.get("/manuals/current", response_model=CurrentManual)
def current_manual() -> CurrentManual:
    """The manual that questions go to by default."""
    cfg = get_cfg()
    name = pipeline.latest_manual(cfg)
    if not name:
        raise HTTPException(404, NO_MANUAL)
    return CurrentManual(manual=name, title=pipeline.manual_title(cfg, name) or name)


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest) -> AskResponse:
    """Answer a question from the current manual (or the one named), with citations."""
    question = req.question.strip()
    if not question:
        raise HTTPException(422, "Type a question first.")

    cfg = get_cfg()
    if req.manual:
        if req.manual not in pipeline.manuals(cfg):
            raise HTTPException(404, f"No manual named {req.manual} has been uploaded.")
        manual = req.manual
    else:
        manual = pipeline.latest_manual(cfg)
        if not manual:
            raise HTTPException(409, NO_MANUAL)

    try:
        answer = pipeline.ask(cfg, question, manual=manual, retriever=get_retriever())
    except Exception as exc:
        logger.exception("Question failed")
        raise HTTPException(
            502, f"The language model couldn't answer right now ({type(exc).__name__}). "
                 "Please try again.") from exc

    display_k = cfg.section("retrieval").get("display_k", 5)
    return AskResponse(
        answer=answer.answer,
        manual=manual,
        model=answer.model or "",
        sources=[
            Source(manual=h.source, manual_title=h.manual_title, page=h.page, section=h.section,
                   start_line=h.start_line, end_line=h.end_line,
                   similarity=round(h.similarity, 4), text=h.text)
            for h in answer.hits[:display_k]
        ],
    )


def _save_upload(upload: UploadFile, target: Path, max_bytes: int) -> None:
    """Stream the upload to disk, stopping early if it's over the size limit."""
    size = 0
    with target.open("wb") as out:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(
                    413, f"File is larger than the {max_bytes // (1024 * 1024)} MB limit.")
            out.write(chunk)
    if size == 0:
        raise HTTPException(422, "The uploaded file is empty.")
