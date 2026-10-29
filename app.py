"""Web app: upload a manual, ask questions in plain English, get cited answers.

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
    ALL_MANUALS,
    format_ingest_report,
    format_sources,
    is_supported,
    manual_choices,
    safe_filename,
    to_manual_filter,
)

logger = logging.getLogger("app")
cfg = load_config()
DISPLAY_K = cfg.section("retrieval").get("display_k", 5)
_retriever = None  # built lazily and rebuilt after each upload


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = pipeline.build_retriever(cfg)
    return _retriever


def list_manuals() -> dict[str, int]:
    try:
        return pipeline.manuals(cfg)
    except Exception:
        return {}


def add_manual(file_path: str | None):
    """Copy the upload into the documents folder, index it, select it."""
    global _retriever
    manuals = list_manuals()
    if not file_path:
        return "Choose a file first.", gr.Dropdown(choices=manual_choices(manuals))
    name = safe_filename(Path(file_path).name)
    if not is_supported(name):
        return (f"❌ `{Path(name).suffix}` files aren't supported. "
                "Use PDF, Word (.docx), EPUB, XPS, .txt or .md."), gr.Dropdown(choices=manual_choices(manuals))

    dest = cfg.path("documents") / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(file_path, dest)
    report = pipeline.ingest(cfg, dest)
    _retriever = None  # the index changed

    manuals = list_manuals()
    selected = name if name in manuals else ALL_MANUALS
    return format_ingest_report(report), gr.Dropdown(choices=manual_choices(manuals), value=selected)


def answer_question(question: str, manual_choice: str):
    """Retrieve, generate and format an answer for the UI."""
    if not question or not question.strip():
        return "Type a question first.", ""
    if not list_manuals():
        return "Add a manual first (left side).", ""
    try:
        answer = pipeline.ask(cfg, question.strip(), manual=to_manual_filter(manual_choice),
                              retriever=get_retriever())
    except Exception as exc:  # show the problem instead of a stack trace
        logger.exception("Question failed")
        return f"❌ Something went wrong: `{type(exc).__name__}: {exc}`", ""
    footer = f"\n\n<sub>Answered by {answer.model}</sub>" if answer.model else ""
    return answer.answer + footer, format_sources(answer.hits, DISPLAY_K)


def build_app() -> gr.Blocks:
    manuals = list_manuals()
    with gr.Blocks(title="Manual Assistant") as demo:
        gr.Markdown(
            "# 📘 Manual Assistant\n"
            "Upload any instruction manual and ask questions in plain English. "
            "Answers come only from the manual and cite the page they came from."
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Add a manual")
                upload = gr.File(
                    label="PDF, Word, EPUB, XPS or text",
                    file_types=[".pdf", ".docx", ".epub", ".xps", ".txt", ".md"],
                    type="filepath",
                )
                add_btn = gr.Button("Add manual", variant="secondary")
                status = gr.Markdown()
            with gr.Column(scale=2):
                gr.Markdown("### 2. Ask a question")
                manual = gr.Dropdown(
                    choices=manual_choices(manuals),
                    value=next(iter(manuals), ALL_MANUALS) if len(manuals) == 1 else ALL_MANUALS,
                    label="Which manual?",
                )
                question = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. How do I clean it?",
                    lines=2,
                )
                ask_btn = gr.Button("Ask", variant="primary")
                answer = gr.Markdown()
                with gr.Accordion("Sources used for this answer", open=False):
                    sources = gr.Markdown()

        add_btn.click(add_manual, inputs=upload, outputs=[status, manual])
        ask_btn.click(answer_question, inputs=[question, manual], outputs=[answer, sources])
        question.submit(answer_question, inputs=[question, manual], outputs=[answer, sources])
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
