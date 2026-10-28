"""Unit tests for retrieval helpers (no database or model needed)."""

import pytest

from manual_rag.retrieval import Retriever, parse_query_results


def _raw(dist=0.401):
    return {
        "ids": [["a"]],
        "documents": [["Remove the screw."]],
        "metadatas": [[{"page": 17, "section": "Cooktop Light", "start_line": 5,
                        "end_line": 13, "source": "734301.pdf"}]],
        "distances": [[dist]],
    }


def test_distance_converted_to_similarity():
    hit = parse_query_results(_raw(0.401))[0]
    assert hit.similarity == 0.599
    assert (hit.page, hit.source, hit.section) == (17, "734301.pdf", "Cooktop Light")


def test_empty_results():
    assert parse_query_results({"ids": [[]]}) == []


class _FakeEmbedder:
    def encode(self, texts):
        return [[0.0, 1.0]]


class _FakeCollection:
    def __init__(self):
        self.last_kwargs = None

    def query(self, **kwargs):
        self.last_kwargs = kwargs
        return _raw()


def test_manual_filter_is_passed_to_chroma():
    col = _FakeCollection()
    Retriever(col, _FakeEmbedder()).retrieve("light broken", top_k=5, manual="734301.pdf")
    assert col.last_kwargs["where"] == {"source": "734301.pdf"}
    assert col.last_kwargs["n_results"] == 5


def test_no_filter_searches_everything():
    col = _FakeCollection()
    Retriever(col, _FakeEmbedder()).retrieve("light broken")
    assert "where" not in col.last_kwargs


def test_empty_query_rejected():
    with pytest.raises(ValueError):
        Retriever(_FakeCollection(), _FakeEmbedder()).retrieve("   ")
