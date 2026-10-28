import pytest

from manual_rag.evaluation import compute_metrics, find_misses, first_relevant_rank


def test_first_relevant_rank():
    assert first_relevant_rank([4, 17, 19], [17]) == 2
    assert first_relevant_rank([4, 5], [17]) is None


def test_compute_metrics():
    retrieved = [[17, 4, 19], [4, 13, 19], [1, 2, 3]]
    expected = [[17], [13, 19], [21]]
    m = compute_metrics(retrieved, expected, ks=(1, 3))
    assert m["hit@1"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["hit@3"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0) / 3, abs=1e-4)
    assert m["n"] == 3


def test_compute_metrics_length_mismatch():
    with pytest.raises(ValueError):
        compute_metrics([[1]], [[1], [2]])


def test_find_misses():
    per_question = [
        {"question": "a", "expected_pages": [17], "retrieved_pages": [17, 4]},
        {"question": "b", "expected_pages": [13], "retrieved_pages": [4, 5, 6, 13]},
        {"question": "c", "expected_pages": [21], "retrieved_pages": [1, 2]},
    ]
    misses = find_misses(per_question, k=3)
    assert [(m["question"], m["rank"]) for m in misses] == [("b", 4), ("c", None)]


def test_cross_manual_hits_need_the_right_manual():
    # Searching all manuals: page 9 of the wrong manual must not count.
    retrieved = [[("ge.pdf", 9), ("coffee.pdf", 9)]]
    expected = [[("coffee.pdf", 9)]]
    assert compute_metrics(retrieved, expected, ks=(1, 3)) == {"hit@1": 0.0, "hit@3": 1.0, "mrr": 0.5, "n": 1}
