"""Retrieval evaluation against a small hand labelled question set.

Each line of ``data/eval/questions.jsonl`` holds a question and the PDF
page numbers that contain the answer. A retrieval is counted as correct when
any of the top ``k`` chunks comes from one of those pages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def load_eval_set(path: str | Path) -> list[dict[str, Any]]:
    """Load question records from a JSONL file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def first_relevant_rank(retrieved_pages: Sequence[int], expected_pages: Sequence[int]) -> int | None:
    """1 based rank of the first retrieved chunk on an expected page, or None."""
    expected = set(expected_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected:
            return rank
    return None


def compute_metrics(
    retrieved: Sequence[Sequence[int]],
    expected: Sequence[Sequence[int]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Compute Hit@k and Mean Reciprocal Rank.

    Args:
        retrieved: For each question, retrieved page numbers in rank order.
        expected: For each question, the list of acceptable answer pages.
        ks: Cutoffs for Hit@k.

    Returns:
        Dict like ``{"hit@1": 0.6, ..., "mrr": 0.71, "n": 18}``.
    """
    if len(retrieved) != len(expected):
        raise ValueError("retrieved and expected must have the same length")
    n = len(retrieved)
    if n == 0:
        return {"n": 0}

    ranks = [first_relevant_rank(r, e) for r, e in zip(retrieved, expected)]
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"hit@{k}"] = round(sum(1 for r in ranks if r is not None and r <= k) / n, 4)
    metrics["mrr"] = round(sum(1.0 / r for r in ranks if r is not None) / n, 4)
    metrics["n"] = n
    return metrics


def find_misses(per_question: Sequence[dict[str, Any]], k: int = 3) -> list[dict[str, Any]]:
    """Questions whose answer page was not in the top k results.
    These are the ones worth reading by hand to understand failures."""
    misses = []
    for rec in per_question:
        rank = first_relevant_rank(rec["retrieved_pages"], rec["expected_pages"])
        if rank is None or rank > k:
            misses.append({**rec, "rank": rank})
    return misses


def plot_hit_at_k(metrics: dict[str, float], out_path: str | Path) -> Path:
    """Save a bar chart of Hit@k values. Returns the output path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in metrics if k.startswith("hit@")]
    values = [metrics[k] for k in keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(keys, values, color="#4C72B0")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of questions")
    ax.set_title(f"Retrieval Hit@k (n={int(metrics['n'])}, MRR={metrics['mrr']:.2f})")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
