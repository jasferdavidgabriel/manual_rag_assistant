"""Web app: upload one manual, ask questions in plain English, get cited answers.

Each visitor works with the manual they uploaded in their own session.
Manuals over the page limit, or made mostly of images, are rejected.

Usage:
    python app.py            # local: http://127.0.0.1:7860
    python app.py --share    # also prints a public link (use this in Colab)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr  # noqa: E402

from manual_rag import pipeline  # noqa: E402
from manual_rag.config import load_config  # noqa: E402
from manual_rag.ui import (  # noqa: E402
    format_sources,
    inspect_document,
    is_supported,
    safe_filename,
    validate_upload,
)

logger = logging.getLogger("app")
cfg = load_config()
UPLOAD = cfg.section("upload")
MAX_PAGES = UPLOAD.get("max_pages", 50)
MIN_CHARS_PER_PAGE = UPLOAD.get("min_chars_per_page", 200)
DISPLAY_K = cfg.section("retrieval").get("display_k", 5)
_retriever = None  # shared search engine, rebuilt after each upload


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = pipeline.build_retriever(cfg)
    return _retriever


def load_manual(file_path: str | None):
    """Check the upload, index it, and make it this session's manual.

    Returns: (status message, session manual, Ask button update)
    """
    global _retriever
    locked = gr.Button(interactive=False)
    if not file_path:
        return "Choose a file first.", None, locked

    name = safe_filename(Path(file_path).name)
    if not is_supported(name):
        return ("❌ That file type isn't supported. Please upload a PDF, Word (.docx), "
                "EPUB, XPS, .txt or .md manual."), None, locked

    try:
        info = inspect_document(file_path)
    except Exception as exc:
        logger.exception("Could not open upload")
        return f"❌ Couldn't open that file: `{exc}`", None, locked
    problem = validate_upload(info, MAX_PAGES, MIN_CHARS_PER_PAGE)
    if problem:
        logger.info("Rejected %s: %s", name, problem)
        return f"❌ {problem}", None, locked

    dest = cfg.path("documents") / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(file_path, dest)
    logger.info("Ingesting %s (%d pages)", name, info.pages)
    report = pipeline.ingest(cfg, dest)
    _retriever = None  # the index changed

    if name in report.failed:
        return f"❌ Couldn't read **{name}**: {report.failed[name]}", None, locked
    if name in report.empty:
        return "❌ No readable text was found in this manual.", None, locked

    title = pipeline.manual_title(cfg, name) or name
    sections = report.indexed.get(name) or pipeline.manuals(cfg).get(name, 0)
    status = f"✅ Ready: **{title}**\n\n{info.pages} pages · {sections} sections. Ask away!"
    return status, name, gr.Button(interactive=True)


def answer_question(question: str, manual: str | None):
    """Answer a question about this session's manual.

    A generator, so the page shows a "Reading the manual..." message straight
    away instead of looking frozen while the model works.
    """
    if not manual:
        yield "Upload a manual first (left side).", ""
        return
    if not question or not question.strip():
        yield "Type a question first.", ""
        return

    logger.info("Question about %s: %s", manual, question.strip())
    yield "⏳ Reading the manual… this usually takes about 5 seconds.", ""
    try:
        answer = pipeline.ask(cfg, question.strip(), manual=manual, retriever=get_retriever())
    except Exception as exc:  # show the problem on the page, not only in the logs
        logger.exception("Question failed")
        yield f"❌ Something went wrong: `{type(exc).__name__}: {exc}`", ""
        return
    logger.info("Answered by %s", answer.model)
    footer = f"\n\n<sub>Answered by {answer.model}</sub>" if answer.model else ""
    yield answer.answer + footer, format_sources(answer.hits, DISPLAY_K)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Manual Assistant") as demo:
        current_manual = gr.State(None)  # per visitor: the manual they uploaded

        gr.Markdown(
            "# 📘 Manual Assistant\n"
            "Upload an instruction manual and ask questions in plain English. "
            "Answers come only from your manual and show the page they came from.\n\n"
            f"<sub>Text based manuals up to {MAX_PAGES} pages: PDF, Word, EPUB, XPS or text.</sub>"
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Upload your manual")
                upload = gr.File(
                    label="Manual file",
                    file_types=[".pdf", ".docx", ".epub", ".xps", ".txt", ".md"],
                    type="filepath",
                )
                load_btn = gr.Button("Load manual", variant="secondary")
                status = gr.Markdown()
            with gr.Column(scale=2):
                gr.Markdown("### 2. Ask a question")
                question = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. How do I clean it?",
                    lines=2,
                )
                ask_btn = gr.Button("Ask", variant="primary", interactive=False)
                answer = gr.Markdown()
                with gr.Accordion("Sources used for this answer", open=False):
                    sources = gr.Markdown()

        load_btn.click(load_manual, inputs=upload, outputs=[status, current_manual, ask_btn])
        ask_btn.click(answer_question, inputs=[question, current_manual], outputs=[answer, sources])
        question.submit(answer_question, inputs=[question, current_manual], outputs=[answer, sources])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Assistant web app")
    parser.add_argument("--share", action="store_true", help="Create a public link (for Colab).")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    build_app().queue().launch(share=args.share, server_port=args.port)
