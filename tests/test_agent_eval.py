"""Tests for the agent evaluation. A fake agent and retriever stand in for
the real ones, so no model, index or API key is needed."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from manual_rag import agent_eval  # noqa: E402
from manual_rag.agent import AgentResult, Step  # noqa: E402
from manual_rag.retrieval import Hit  # noqa: E402


def hit(page, source="c.pdf"):
    return Hit(text="t", page=page, section="S", start_line=1, end_line=2,
               similarity=0.5, source=source, manual_title="")


RECORDS = [
    {"question": "warranty?", "manual": "c.pdf", "expected_pages": [9]},
    {"question": "descale?", "manual": "c.pdf", "expected_pages": [7]},
    {"question": "clock?", "manual": "m.pdf", "expected_pages": [13]},
]


class FakeRetriever:
    def retrieve(self, query, top_k=5, manual=None):
        return {"warranty?": [hit(9)], "descale?": [hit(3)], "clock?": [hit(13, "m.pdf")]}[query]


class FakeAgent:
    top_k = 5
    max_steps = 6

    def __init__(self, cfg=None, retriever=None, fail_times=0):
        self.fail_times = fail_times

    def run(self, question):
        if self.fail_times:
            self.fail_times -= 1
            raise TimeoutError("rate limited")
        if question == "warranty?":
            return AgentResult("One year (c.pdf, page 9, Warranty).", "m1",
                               [Step("search_manual", {"query": "warranty"}, "")], [hit(9)], 2)
        if question == "descale?":
            return AgentResult("Use vinegar (c.pdf, page 7).", "m1",
                               [Step("search_manual", {"query": "slow"}, ""),
                                Step("search_manual", {"query": "descale"}, "")],
                               [hit(3), hit(7)], 3)
        return AgentResult("Not covered.", "m1",
                           [Step("list_manuals", {}, ""), Step("search_manual", {"query": "clock"}, "")],
                           [hit(13, "c.pdf")], 3)  # page 13 but the wrong manual


def clock_from(times):
    it = iter(times)
    return lambda: next(it)


def test_cited_pages():
    assert agent_eval.cited_pages("See page 9 and pages 7, also Page 12.") == {9, 7, 12}
    assert agent_eval.cited_pages("") == set()


def test_scores_each_question():
    result = agent_eval.evaluate_agent(FakeAgent(), FakeRetriever(), RECORDS,
                                       clock=clock_from([0, 2, 0, 4, 0, 3]))
    rows = result["per_question"]
    assert [r["found"] for r in rows] == [True, True, False]      # last: right page, wrong manual
    assert [r["cited"] for r in rows] == [True, True, False]
    assert [r["baseline_found"] for r in rows] == [True, False, True]
    assert rows[1]["searches"] == 2 and rows[2]["searches"] == 1

    s = result["summary"]
    assert s["questions"] == 3 and s["errors"] == 0
    assert s["found_rate"] == 0.6667 and s["baseline_found_rate"] == 0.6667
    assert s["avg_searches"] == 1.33 and s["avg_model_calls"] == 2.67
    assert s["avg_seconds"] == 3.0


def test_failed_question_is_retried_once():
    waits = []
    agent = FakeAgent(fail_times=1)
    result = agent_eval.evaluate_agent(agent, FakeRetriever(), RECORDS[:1], sleep=waits.append,
                                       clock=clock_from([0, 1, 0, 5]))
    assert waits == [30.0]
    assert result["per_question"][0]["found"] and result["summary"]["errors"] == 0


def test_question_failing_twice_is_recorded_as_an_error():
    agent = FakeAgent(fail_times=2)
    result = agent_eval.evaluate_agent(agent, FakeRetriever(), RECORDS[:1], sleep=lambda s: None,
                                       clock=clock_from([0, 1, 0, 1]))
    row = result["per_question"][0]
    assert row["error"].startswith("TimeoutError") and not row["found"]
    assert result["summary"]["errors"] == 1 and result["summary"]["avg_seconds"] == 0.0


def test_run_end_to_end(monkeypatch, tmp_path):
    from manual_rag import pipeline

    retriever = FakeRetriever()
    retriever.collection = object()
    monkeypatch.setattr(pipeline, "build_retriever", lambda cfg: retriever)
    monkeypatch.setattr(pipeline, "list_sources", lambda c: ["c.pdf", "m.pdf"])
    monkeypatch.setattr(pipeline, "load_eval_set", lambda p: RECORDS)
    cfg = SimpleNamespace(path=lambda name: tmp_path / name,
                          section=lambda name: {"model": "m1"} if name == "llm" else {})

    result = agent_eval.run(cfg, limit=2, agent_factory=FakeAgent)
    assert result["summary"]["questions"] == 2
    assert result["settings"]["model"] == "m1" and result["settings"]["manual_hint"] is False
    assert (tmp_path / "results_dir" / "agent_eval.json").exists()


# ---------- questions with several manuals ----------

COMPARISON = {"id": 101, "type": "comparison", "match": "all", "question": "compare?",
              "expected": [{"manual": "c.pdf", "pages": [9]}, {"manual": "m.pdf", "pages": [21]}]}
AMBIGUOUS = {"id": 108, "type": "ambiguous", "match": "any", "question": "how long?",
             "expected": [{"manual": "c.pdf", "pages": [9]}, {"manual": "m.pdf", "pages": [21]}]}


def test_targets_reads_both_formats():
    assert agent_eval.targets(RECORDS[0]) == ([("c.pdf", {9})], "all")
    goals, rule = agent_eval.targets(AMBIGUOUS)
    assert goals == [("c.pdf", {9}), ("m.pdf", {21})] and rule == "any"


def test_score_hits_all_and_any():
    one_manual = [hit(9)]
    both = [hit(9), hit(21, "m.pdf")]
    assert agent_eval.score_hits(one_manual, COMPARISON) == (False, False)
    assert agent_eval.score_hits(both, COMPARISON) == (True, True)
    assert agent_eval.score_hits(one_manual, AMBIGUOUS) == (True, False)
    assert agent_eval.score_hits([hit(21)], AMBIGUOUS) == (False, False)  # page 21 of the wrong manual


def test_score_citations_all_and_any():
    assert agent_eval.score_citations("See page 9 and page 21.", COMPARISON)
    assert not agent_eval.score_citations("See page 9.", COMPARISON)
    assert agent_eval.score_citations("See page 9.", AMBIGUOUS)


class TwoManualRetriever:
    def retrieve(self, query, top_k=5, manual=None):
        return [hit(9)]  # a single search only finds the coffee maker page


class ComparingAgent:
    top_k = 5
    max_steps = 6

    def run(self, question):
        return AgentResult("Both one year (c.pdf, page 9; m.pdf, page 21).", "m1",
                           [Step("search_manual", {"query": "w", "manual": "c.pdf"}, ""),
                            Step("search_manual", {"query": "w", "manual": "m.pdf"}, "")],
                           [hit(9), hit(21, "m.pdf")], 3)


def test_comparison_and_ambiguous_scored_per_type():
    result = agent_eval.evaluate_agent(ComparingAgent(), TwoManualRetriever(), [COMPARISON, AMBIGUOUS],
                                       clock=clock_from([0, 1, 0, 1]))
    rows = result["per_question"]
    assert [r["found"] for r in rows] == [True, True]
    assert [r["baseline_found"] for r in rows] == [False, True]   # one search misses the 2nd manual
    assert [r["covered_all"] for r in rows] == [True, True]
    assert rows[0]["cited"] and rows[0]["type"] == "comparison"
    assert result["by_type"]["comparison"] == {"questions": 1, "found_rate": 1.0,
                                               "baseline_found_rate": 0.0, "covered_all_rate": 1.0,
                                               "baseline_covered_all_rate": 0.0}
    assert result["summary"]["baseline_covered_all_rate"] == 0.0


def test_run_with_a_questions_file(monkeypatch, tmp_path):
    import json
    from manual_rag import pipeline

    path = tmp_path / "hard.jsonl"
    other = {**AMBIGUOUS, "id": 109, "expected": [{"manual": "x.pdf", "pages": [1]}]}
    path.write_text("\n".join(json.dumps(r) for r in [COMPARISON, other]) + "\n", encoding="utf-8")
    retriever = TwoManualRetriever()
    retriever.collection = object()
    monkeypatch.setattr(pipeline, "build_retriever", lambda cfg: retriever)
    monkeypatch.setattr(pipeline, "list_sources", lambda c: ["c.pdf", "m.pdf"])
    cfg = SimpleNamespace(path=lambda name: tmp_path / name, section=lambda name: {})

    result = agent_eval.run(cfg, questions_path=path, agent_factory=lambda cfg, retriever: ComparingAgent())
    assert result["summary"]["questions"] == 1          # x.pdf isn't indexed, so 109 is skipped
    assert result["settings"]["questions_file"] == str(path)
