"""Tests for the learned reranker and its grouped cross validation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("sklearn")
from manual_rag import rerank  # noqa: E402
from manual_rag.retrieval import Hit  # noqa: E402


def hit(page, section="Other", text="some text", sim=0.5, source="m.pdf"):
    return Hit(text=text, page=page, section=section, start_line=1, end_line=2,
               similarity=sim, source=source, manual_title="")


class FakeRetriever:
    def __init__(self, hits_by_question):
        self.hits_by_question = hits_by_question
        self.calls = []

    def retrieve(self, query, top_k=10, manual=None):
        self.calls.append((query, top_k, manual))
        return self.hits_by_question[query][:top_k]


def test_tokens_drop_stopwords_and_punctuation():
    assert rerank.tokens("How do I lock the Controls?") == {"lock", "controls"}


def test_hit_features():
    hits = [hit(13, section="Control Lock-Out", text="Hold Cancel to lock", sim=0.6),
            hit(7, section="Cleaning", text="Wipe the panel", sim=0.4)]
    rows = rerank.hit_features("How do I lock the controls?", hits)
    assert rows[0]["rank"] == 1 and rows[1]["rank"] == 2
    assert rows[0]["title_overlap"] == 0.5          # "controls" isn't "control"; "lock" matches
    assert rows[0]["text_overlap"] == 0.5
    assert rows[1]["title_overlap"] == 0.0
    assert rows[1]["gap_to_top"] == pytest.approx(0.2)
    assert set(rows[0]) == set(rerank.FEATURES)


def test_build_dataset_labels_right_page_and_manual():
    retriever = FakeRetriever({"q": [hit(5, source="a.pdf"), hit(9, source="b.pdf"),
                                     hit(9, source="a.pdf")]})
    records = [{"question": "q", "manual": "a.pdf", "expected_pages": [9]}]
    rows = rerank.build_dataset(retriever, records, k=3, across=True)
    assert [r["label"] for r in rows] == [0, 0, 1]  # page 9 of the wrong manual doesn't count
    assert retriever.calls == [("q", 3, None)]

    rerank.build_dataset(retriever, records, k=3)    # within-manual mode filters by manual
    assert retriever.calls[-1] == ("q", 3, "a.pdf")


def test_rank_metrics():
    rows = [{"qid": 0, "rank": 1, "label": 0}, {"qid": 0, "rank": 2, "label": 1},
            {"qid": 1, "rank": 1, "label": 1}, {"qid": 1, "rank": 2, "label": 0}]
    by_similarity = rerank.rank_metrics(rows, [0.9, 0.8, 0.9, 0.8])
    assert by_similarity == {"hit@1": 0.5, "hit@3": 1.0, "mrr": 0.75, "n": 2}
    flipped = rerank.rank_metrics(rows, [0.1, 0.8, 0.9, 0.8])  # tie keeps original order
    assert flipped["hit@1"] == 1.0


def test_folds_never_split_a_question():
    groups = [q for q in range(12) for _ in range(5)]
    for train, test in rerank.fold_splits(groups, 4):
        assert not {groups[i] for i in train} & {groups[i] for i in test}


def test_cross_validation_learns_a_useful_signal():
    # The correct chunk always has a matching section title but is ranked 2nd by
    # similarity, so a reranker that learns title overlap beats similarity order.
    hits, records = {}, []
    for i in range(20):
        q = f"how to descale unit {i}"
        hits[q] = [hit(1, section="Warranty", sim=0.62),
                   hit(2, section="Descale", sim=0.60),
                   hit(3, section="Cleaning", sim=0.55),
                   hit(4, section="Parts", sim=0.50)]
        records.append({"question": q, "manual": "m.pdf", "expected_pages": [2]})
    rows = rerank.build_dataset(FakeRetriever(hits), records, k=4)

    result = rerank.cross_validate(rows, n_splits=5)
    assert result["baseline"]["hit@1"] == 0.0
    assert result["reranked"]["hit@1"] == 1.0
    assert result["questions"] == 20 and result["n_splits"] == 5
    assert len(result["folds"]) == 5
    assert result["weights"]["title_overlap"] > 0


def test_cross_validation_needs_both_classes():
    rows = [{"qid": q, "label": 0, **{f: 0.0 for f in rerank.FEATURES}} for q in range(6)]
    with pytest.raises(ValueError):
        rerank.cross_validate(rows)


def test_cross_encoder_scores_are_attached_and_compared():
    hits, records = {}, []
    for i in range(10):
        q = f"descale question {i}"
        hits[q] = [hit(1, section="Warranty", sim=0.62), hit(2, section="Descale", sim=0.60),
                   hit(3, section="Cleaning", sim=0.55)]
        records.append({"question": q, "manual": "m.pdf", "expected_pages": [2]})

    def fake_scorer(question, chunk_hits):  # pretends to know the right chunk
        return [5.0 if h.section == "Descale" else 0.0 for h in chunk_hits]

    rows = rerank.build_dataset(FakeRetriever(hits), records, k=3, scorer=fake_scorer)
    assert [r["ce_score"] for r in rows[:3]] == [0.0, 5.0, 0.0]

    result = rerank.cross_validate(rows, n_splits=5)
    assert result["baseline"]["hit@1"] == 0.0
    assert result["cross_encoder"]["hit@1"] == 1.0


def test_no_cross_encoder_result_without_scores():
    hits = {f"q{i}": [hit(1, sim=0.6), hit(2, sim=0.5)] for i in range(6)}
    records = [{"question": f"q{i}", "manual": "m.pdf", "expected_pages": [1 + i % 2]}
               for i in range(6)]
    result = rerank.cross_validate(rerank.build_dataset(FakeRetriever(hits), records, k=2), 3)
    assert "cross_encoder" not in result


class FakeCfg:
    def __init__(self, root):
        self.root = root

    def path(self, name):
        return self.root / name


def test_run_end_to_end(monkeypatch, tmp_path):
    from manual_rag import pipeline

    hits = {f"q{i}": [hit(1, section="Warranty", sim=0.62), hit(2, section="Descale", sim=0.6)]
            for i in range(10)}
    retriever = FakeRetriever(hits)
    retriever.collection = object()
    records = [{"question": f"q{i}", "manual": "m.pdf", "expected_pages": [2]} for i in range(10)]
    records.append({"question": "ignored", "manual": "not_indexed.pdf", "expected_pages": [1]})
    monkeypatch.setattr(pipeline, "build_retriever", lambda cfg: retriever)
    monkeypatch.setattr(pipeline, "list_sources", lambda collection: ["m.pdf"])
    monkeypatch.setattr(pipeline, "load_eval_set", lambda path: records)

    result = rerank.run(FakeCfg(tmp_path), k=2, n_splits=5, cross_encoder=False)
    assert result["questions"] == 10                   # the unindexed manual's question is skipped
    assert result["settings"]["mode"] == "within_manual"
    assert result["settings"]["cross_encoder"] == "none"
    assert (tmp_path / "results_dir" / "rerank_cv.json").exists()
