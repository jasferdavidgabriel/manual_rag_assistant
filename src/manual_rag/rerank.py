"""Learned reranker for retrieved chunks, checked with grouped cross validation.

Retrieval ranks chunks by embedding similarity alone. This module turns each
question's top k chunks into feature rows (similarity, rank, word overlap with
the section title and text, ...) labelled by whether the chunk is on a correct
page, then trains a small scikit-learn model to reorder them.

Why GroupKFold: the k rows from one question are closely related. A random
split would put some of a question's chunks in training and the rest in
testing, so the model would be scored on questions it had partly seen and the
result would look better than it is. GroupKFold keeps every question entirely
inside one fold, so each fold is scored only on unseen questions.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Iterable, Sequence

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "how", "i", "if", "in", "is", "it", "my", "of", "on", "or", "should", "that", "the",
    "this", "to", "what", "when", "where", "which", "who", "why", "with", "you", "your",
}

FEATURES = [
    "similarity",      # embedding cosine similarity (what retrieval ranks by today)
    "rank",            # position in the similarity ranking (1 = top)
    "gap_to_top",      # how far below the best chunk's similarity this one is
    "title_overlap",   # share of question words found in the section title
    "text_overlap",    # share of question words found in the chunk text
    "title_words",     # length of the section title, in words
    "text_len_log",    # log of the chunk length in characters
]


def tokens(text: str) -> set[str]:
    """Lowercase content words: "How do I lock the Controls?" -> {"lock", "controls"}."""
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if w not in STOPWORDS and len(w) > 1}


def hit_features(question: str, hits: Sequence) -> list[dict[str, float]]:
    """One feature dict per retrieved chunk, in retrieval order."""
    q = tokens(question)
    top = hits[0].similarity if hits else 0.0
    rows = []
    for rank, h in enumerate(hits, start=1):
        title, body = tokens(h.section), tokens(h.text)
        rows.append({
            "similarity": float(h.similarity),
            "rank": float(rank),
            "gap_to_top": float(top - h.similarity),
            "title_overlap": len(q & title) / len(q) if q else 0.0,
            "text_overlap": len(q & body) / len(q) if q else 0.0,
            "title_words": float(len(title)),
            "text_len_log": math.log1p(len(h.text or "")),
        })
    return rows


def build_dataset(retriever, records: Iterable[dict], k: int = 10,
                  across: bool = False, scorer=None) -> list[dict[str, Any]]:
    """Feature rows for every (question, retrieved chunk) pair.

    across=False searches inside each question's own manual; across=True
    searches all manuals, and a chunk only counts if it's the right manual
    AND page. "qid" groups the rows of one question for GroupKFold.
    scorer(question, hits) -> scores, if given, adds a "ce_score" to each row
    (used for the cross encoder comparison).
    """
    rows = []
    for qid, rec in enumerate(records):
        hits = retriever.retrieve(rec["question"], top_k=k,
                                  manual=None if across else rec["manual"])
        expected = set(rec["expected_pages"])
        ce_scores = scorer(rec["question"], hits) if scorer else [None] * len(hits)
        for feats, h, ce in zip(hit_features(rec["question"], hits), hits, ce_scores):
            correct = h.page in expected and (h.source or rec["manual"]) == rec["manual"]
            rows.append({"qid": qid, "question": rec["question"], "manual": rec["manual"],
                         "source": h.source, "page": h.page, "section": h.section,
                         "label": int(correct), **feats})
            if ce is not None:
                rows[-1]["ce_score"] = float(ce)
    return rows


def rank_metrics(rows: Sequence[dict], scores: Sequence[float]) -> dict[str, float]:
    """Hit@1, Hit@3 and MRR when each question's chunks are ordered by score.

    Ties keep the original retrieval order. A question with no correct chunk
    in its top k scores 0.
    """
    by_q: dict[int, list] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_q[row["qid"]].append((float(score), row["rank"], row["label"]))

    hit1, hit3, rr = [], [], []
    for items in by_q.values():
        items.sort(key=lambda t: (-t[0], t[1]))
        pos = next((i for i, (_, _, y) in enumerate(items, start=1) if y), None)
        hit1.append(pos == 1)
        hit3.append(pos is not None and pos <= 3)
        rr.append(1.0 / pos if pos else 0.0)
    n = len(by_q)
    return {"hit@1": round(sum(hit1) / n, 4), "hit@3": round(sum(hit3) / n, 4),
            "mrr": round(sum(rr) / n, 4), "n": n} if n else {"hit@1": 0, "hit@3": 0, "mrr": 0, "n": 0}


def fold_splits(groups: Sequence[int], n_splits: int):
    """GroupKFold train/test index splits; a question never spans two folds."""
    import numpy as np
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    n_splits = max(2, min(n_splits, len(set(groups.tolist()))))
    dummy = np.zeros(len(groups))
    return list(GroupKFold(n_splits=n_splits).split(dummy, dummy, groups))


def make_model():
    """Scaled logistic regression: small, fast, and its weights are readable."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000))


def cross_validate(rows: Sequence[dict], n_splits: int = 5) -> dict[str, Any]:
    """Out-of-fold reranking scores compared with plain similarity ranking.

    Each row's reranker score comes from a model that never saw its question,
    so the reranked metrics are an honest estimate on unseen questions.
    """
    import numpy as np

    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows])
    groups = [r["qid"] for r in rows]
    if len(set(y.tolist())) < 2:
        raise ValueError("Need both correct and incorrect chunks to train a reranker.")

    splits = fold_splits(groups, n_splits)
    oof = np.zeros(len(rows))
    folds = []
    for fold, (train, test) in enumerate(splits, start=1):
        if len(set(y[train].tolist())) < 2:  # can't train on one class: keep similarity order
            oof[test] = X[test, FEATURES.index("similarity")]
        else:
            oof[test] = make_model().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        test_rows = [rows[i] for i in test]
        folds.append({
            "fold": fold,
            "questions": len({r["qid"] for r in test_rows}),
            "baseline": rank_metrics(test_rows, [r["similarity"] for r in test_rows]),
            "reranked": rank_metrics(test_rows, oof[test]),
        })

    full = make_model().fit(X, y)  # trained on everything, only to read the weights
    weights = dict(zip(FEATURES, (round(float(w), 3) for w in full[-1].coef_[0])))
    extra = {}
    if all("ce_score" in r for r in rows):  # pretrained, no training, so no folds needed
        extra["cross_encoder"] = rank_metrics(rows, [r["ce_score"] for r in rows])
    return {
        "n_splits": len(splits),
        "rows": len(rows),
        "questions": len(set(groups)),
        "baseline": rank_metrics(rows, [r["similarity"] for r in rows]),
        "reranked": rank_metrics(rows, oof),
        **extra,
        "folds": folds,
        "weights": weights,
    }


CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L6-v2"


def cross_encoder_scorer(model_name: str = CROSS_ENCODER, device: str | None = None):
    """A free pretrained reranker that reads the question and chunk together.

    Unlike the embedding model, which encodes question and chunk separately,
    a cross encoder looks at both at once, which is slower but usually more
    accurate. Returns scorer(question, hits) -> list of relevance scores.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device=device)

    def score(question: str, hits: Sequence) -> list[float]:
        pairs = [(question, f"{h.section}\n{h.text}") for h in hits]
        return [float(x) for x in model.predict(pairs)] if pairs else []

    return score


def run(cfg, k: int = 10, n_splits: int = 5, across: bool = False,
        cross_encoder: bool = True) -> dict[str, Any]:
    """Run the comparison on the indexed manuals and the labelled questions.

    Saves reports/rerank_cv.json and returns the result, including the
    settings used, so the run can be logged to MLflow.
    """
    import json

    from . import pipeline

    retriever = pipeline.build_retriever(cfg)
    indexed = set(pipeline.list_sources(retriever.collection))
    records = [r for r in pipeline.load_eval_set(cfg.path("eval_set"))
               if r.get("manual", "") in indexed]
    if not records:
        raise RuntimeError("No eval questions match the indexed manuals.")

    scorer = cross_encoder_scorer() if cross_encoder else None
    rows = build_dataset(retriever, records, k=k, across=across, scorer=scorer)
    result = cross_validate(rows, n_splits)
    result["settings"] = {
        "k": k,
        "folds": result["n_splits"],
        "mode": "all_manuals" if across else "within_manual",
        "model": "logistic_regression",
        "cross_encoder": CROSS_ENCODER if cross_encoder else "none",
    }

    out = cfg.path("results_dir") / "rerank_cv.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["saved_to"] = str(out)
    return result
