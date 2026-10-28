"""Command line entry point.

Usage:
    python main.py ingest                        # index every manual in data/raw
    python main.py ingest --path my_manual.pdf   # index one file
    python main.py list                          # show indexed manuals
    python main.py search "light not working"    # retrieval only, no LLM
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from manual_rag import pipeline  # noqa: E402
from manual_rag.config import load_config  # noqa: E402
from manual_rag.retrieval import Hit  # noqa: E402


def print_hits(hits: list[Hit], limit: int) -> None:
    """Print retrieved chunks as a compact table."""
    print(f"{'#':<3}{'Manual':<24}{'Page':<6}{'Lines':<10}{'Sim':<7}Section")
    for i, h in enumerate(hits[:limit], start=1):
        lines = f"{h.start_line}-{h.end_line}"
        print(f"{i:<3}{h.source[:22]:<24}{h.page:<6}{lines:<10}{h.similarity:<7.3f}{h.section[:40]}")


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
    p_search.add_argument("--manual", default=None, help="Search within one manual (file name).")
    p_search.add_argument("--show-text", action="store_true", help="Print each chunk's text.")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cfg = load_config(args.config)

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
        hits = pipeline.search(cfg, args.query, manual=args.manual)
        print_hits(hits, cfg.section("retrieval").get("display_k", 5))
        if args.show_text:
            for i, h in enumerate(hits[:3], start=1):
                print(f"\n--- #{i}  page {h.page}  {h.section} ---\n{h.text}")


if __name__ == "__main__":
    main()
