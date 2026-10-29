"""Command line entry point.

Usage:
    python main.py ask "How do I clean it?" --file manual.pdf   # ad hoc: this file only
    python main.py ingest                        # index every manual in data/raw
    python main.py ingest --path my_manual.pdf   # index one file (becomes the default manual)
    python main.py list                          # show indexed manuals
    python main.py search "light not working"    # retrieval only, no LLM
    python main.py ask "How do I replace the cooktop light?"
    python main.py chat                          # interactive question loop
    python main.py evaluate                      # retrieval Hit@k and MRR
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from manual_rag import pipeline  # noqa: E402
from manual_rag.config import load_config  # noqa: E402
from manual_rag.generation import Answer  # noqa: E402
from manual_rag.retrieval import Hit  # noqa: E402


def print_hits(hits: list[Hit], limit: int) -> None:
    """Print retrieved chunks as a compact table."""
    print(f"{'#':<3}{'Manual':<24}{'Page':<6}{'Lines':<10}{'Sim':<7}Section")
    for i, h in enumerate(hits[:limit], start=1):
        lines = f"{h.start_line}-{h.end_line}"
        print(f"{i:<3}{h.source[:22]:<24}{h.page:<6}{lines:<10}{h.similarity:<7.3f}{h.section[:40]}")


def print_answer(answer: Answer, limit: int, show_reasoning: bool) -> None:
    """Print the answer, optionally the reasoning trace, and the sources used."""
    if show_reasoning and answer.reasoning:
        print("\n=== Reasoning ===\n" + answer.reasoning)
    model = f"  ({answer.model})" if answer.model else ""
    print(f"\n=== Answer{model} ===\n" + answer.answer)
    print("\n=== Top retrieved chunks ===")
    print_hits(answer.hits, limit)


def _add_manual_options(parser) -> None:
    """--file, --manual and --all are alternatives, so only one can be given."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", default=None,
                       help="Ad hoc: answer from this manual file (indexed first if new or changed).")
    group.add_argument("--manual", default=None,
                       help="An already indexed manual (file name). Default: the most recent upload.")
    group.add_argument("--all", action="store_true", help="Search every indexed manual.")


def choose_manual(cfg, args) -> str | None:
    """--file (ad hoc) or --manual pick one manual, --all searches everything,
    otherwise the most recently uploaded manual is used."""
    if args.file:
        name = pipeline.load_file(cfg, args.file)
        title = pipeline.manual_title(cfg, name)
        print(f"Using: {name}" + (f" ({title})" if title else ""))
        return name
    if args.manual:
        return args.manual
    if args.all:
        print("Searching: all manuals")
        return None
    latest = pipeline.latest_manual(cfg)
    if latest:
        title = pipeline.manual_title(cfg, latest)
        print(f"Searching: {latest}" + (f" ({title})" if title else "")
              + "   [use --manual NAME or --all to change]")
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description="Cited Q&A over instruction manuals.")
    parser.add_argument("--config", default=None, help="Path to a YAML config file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index a manual or a folder of manuals.")
    p_ingest.add_argument("--path", default=None, help="File or folder (default from config).")
    p_ingest.add_argument("--reset", action="store_true", help="Delete the whole index first.")
    p_ingest.add_argument("--force", action="store_true", help="Re-index unchanged manuals.")

    sub.add_parser("list", help="List indexed manuals.")

    p_search = sub.add_parser("search", help="Show the top chunks for a query (no LLM).")
    p_search.add_argument("query")
    _add_manual_options(p_search)
    p_search.add_argument("--show-text", action="store_true", help="Print each chunk's text.")

    p_ask = sub.add_parser("ask", help="Ask one question and get a cited answer.")
    p_ask.add_argument("question")
    _add_manual_options(p_ask)
    p_ask.add_argument("--show-reasoning", action="store_true", help="Print the model's reasoning.")

    p_chat = sub.add_parser("chat", help="Interactive question loop.")
    _add_manual_options(p_chat)
    p_chat.add_argument("--show-reasoning", action="store_true", help="Print the model's reasoning.")

    sub.add_parser("evaluate", help="Score retrieval on the labelled question set.")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:  # load NVIDIA_API_KEY from a local .env file if python-dotenv is installed
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    cfg = load_config(args.config)
    try:
        run(cfg, args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"❌ {exc}")
        sys.exit(1)


def run(cfg, args) -> None:
    """Execute one CLI command."""
    display_k = cfg.section("retrieval").get("display_k", 5)

    if args.command == "ingest":
        report = pipeline.ingest(cfg, args.path, reset=args.reset, force=args.force)
        for name, n in report.indexed.items():
            print(f"indexed  {name}  ({n} chunks)")
        for name in report.skipped:
            print(f"skipped  {name}  (unchanged)")
        for name in report.empty:
            print(f"empty    {name}  (no text found; scanned PDF without Tesseract?)")
        for name, err in report.failed.items():
            print(f"FAILED   {name}  {err}")
        if report.failed:
            sys.exit(1)
    elif args.command == "list":
        for name, n in pipeline.manuals(cfg).items():
            print(f"{n:>6} chunks  {name}")
    elif args.command == "search":
        hits = pipeline.search(cfg, args.query, manual=choose_manual(cfg, args))
        print_hits(hits, display_k)
        if args.show_text:
            for i, h in enumerate(hits[:3], start=1):
                print(f"\n--- #{i}  page {h.page}  {h.section} ---\n{h.text}")
    elif args.command == "ask":
        answer = pipeline.ask(cfg, args.question, manual=choose_manual(cfg, args))
        print_answer(answer, display_k, args.show_reasoning)
    elif args.command == "chat":
        manual = choose_manual(cfg, args)  # first, so an ad hoc file is indexed before searching
        retriever = pipeline.build_retriever(cfg)
        print("Ask about your manual (empty line or Ctrl+C to quit).")
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            answer = pipeline.ask(cfg, question, manual=manual, retriever=retriever)
            print_answer(answer, display_k, args.show_reasoning)
    elif args.command == "evaluate":
        result = pipeline.evaluate(cfg)
        print(f"Manuals indexed: {', '.join(result['manuals'])}")
        for mode, title in (("within_manual", "Search within each question's manual"),
                            ("all_manuals", "Search across ALL manuals (right manual + page)")):
            block = result[mode]
            print(f"\n=== {title} ===")
            for name, value in block["metrics"].items():
                print(f"{name:>8}: {value}")
            print(f"  Not in top 3 ({len(block['misses'])}):")
            for m in block["misses"]:
                rank = m["rank"] if m["rank"] is not None else "not in top 10"
                print(f"    [{rank}] {m['question']}")
                print(f"          expected {m['expected_pages']}, got {m['retrieved_pages'][:4]}")
        print("\nSaved reports/retrieval_metrics.json and two charts in reports/figures/")

if __name__ == "__main__":
    main()
