"""Evaluate the tool calling agent on labelled questions.

For every question the agent gets no manual hint, so it has to decide where
and how to search. Each question is scored on:

    found  any section the agent looked at is on a correct page of the right manual
    cited  the answer mentions a correct page number
    cost   searches, model calls and seconds

As a reference, the same question is also run as one plain search over all
manuals (top k, no language model), which is what the regular pipeline does.

Two question formats are supported:

    {"manual": "a.pdf", "expected_pages": [9], ...}             one manual
    {"expected": [{"manual": "a.pdf", "pages": [9]},
                  {"manual": "b.pdf", "pages": [21]}],
     "match": "all" or "any", "type": "comparison", ...}        several manuals

With match "all" (comparison questions) every manual's page must be found;
with "any" (ambiguous questions) one is enough. "covered_all" also records
whether every listed manual was found, whatever the rule.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable

PAGE_RE = re.compile(r"\bpages?\s*(\d+)", re.IGNORECASE)


def targets(record: dict) -> tuple[list[tuple[str, set[int]]], str]:
    """The (manual, pages) a question expects, and its rule: "all" or "any"."""
    if "expected" in record:
        return ([(e["manual"], set(e["pages"])) for e in record["expected"]],
                record.get("match", "all"))
    return [(record["manual"], set(record["expected_pages"]))], "all"


def score_hits(hits, record: dict) -> tuple[bool, bool]:
    """(found under the question's rule, every expected manual found)."""
    goals, rule = targets(record)
    per_goal = [any(h.source == manual and h.page in pages for h in hits) for manual, pages in goals]
    found = all(per_goal) if rule == "all" else any(per_goal)
    return found, all(per_goal)


def cited_pages(answer: str) -> set[int]:
    """Page numbers mentioned in an answer: "page 9" or "pages 7" -> {9, 7}."""
    return {int(n) for n in PAGE_RE.findall(answer or "")}


def score_citations(answer: str, record: dict) -> bool:
    """Does the answer mention a correct page, for each manual the rule needs?"""
    goals, rule = targets(record)
    mentioned = cited_pages(answer)
    per_goal = [bool(mentioned & pages) for _, pages in goals]
    return all(per_goal) if rule == "all" else any(per_goal)


def load_questions(path: str | Path) -> list[dict]:
    """Questions from a .jsonl file, one JSON object per line."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_agent(agent, retriever, records: list[dict], retry_wait: float = 30.0,
                   sleep: Callable[[float], None] = time.sleep,
                   clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """Run the agent on every record; a failed question is retried once after a pause."""
    rows = []
    for rec in records:
        goals, rule = targets(rec)
        baseline = retriever.retrieve(rec["question"], top_k=agent.top_k)
        base_found, base_all = score_hits(baseline, rec)
        row = {"id": rec.get("id"), "question": rec["question"], "type": rec.get("type", "single"),
               "match": rule, "expected": [{"manual": m, "pages": sorted(p)} for m, p in goals],
               "baseline_found": base_found, "baseline_covered_all": base_all}

        result, error, seconds = None, None, 0.0
        for attempt in range(2):
            start = clock()
            try:
                result = agent.run(rec["question"])
                seconds, error = clock() - start, None
                break
            except Exception as exc:  # rate limits, timeouts: pause and try once more
                error = f"{type(exc).__name__}: {exc}"
                if attempt == 0:
                    sleep(retry_wait)

        if result is None:
            row.update(found=False, covered_all=False, cited=False, searches=0, model_calls=0,
                       seconds=None, steps=[], answer="", error=error)
        else:
            found, covered_all = score_hits(result.hits, rec)
            row.update(
                found=found,
                covered_all=covered_all,
                cited=score_citations(result.answer, rec),
                searches=sum(1 for s in result.steps if s.tool == "search_manual"),
                model_calls=result.model_calls,
                seconds=round(seconds, 2),
                steps=[{"tool": s.tool, "arguments": s.arguments} for s in result.steps],
                answer=result.answer[:500],
                error=None,
            )
        rows.append(row)

    return {"summary": summarize(rows), "by_type": by_type(rows), "per_question": rows}


def _share(rows, key):
    return round(sum(bool(r[key]) for r in rows) / len(rows), 4) if rows else 0.0


def summarize(rows: list[dict]) -> dict[str, float]:
    ok = [r for r in rows if not r["error"]]

    def average(key):
        return round(mean(r[key] for r in ok), 2) if ok else 0.0

    return {
        "questions": len(rows),
        "found_rate": _share(rows, "found"),
        "cited_rate": _share(rows, "cited"),
        "covered_all_rate": _share(rows, "covered_all"),
        "baseline_found_rate": _share(rows, "baseline_found"),
        "baseline_covered_all_rate": _share(rows, "baseline_covered_all"),
        "avg_searches": average("searches"),
        "avg_model_calls": average("model_calls"),
        "avg_seconds": average("seconds"),
        "errors": len(rows) - len(ok),
    }


def by_type(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Found rates per question type (single, comparison, vague, ambiguous)."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["type"], []).append(r)
    return {t: {"questions": len(g), "found_rate": _share(g, "found"),
                "baseline_found_rate": _share(g, "baseline_found"),
                "covered_all_rate": _share(g, "covered_all"),
                "baseline_covered_all_rate": _share(g, "baseline_covered_all")}
            for t, g in groups.items()}


def run(cfg, limit: int | None = None, agent_factory=None,
        questions_path: str | Path | None = None) -> dict[str, Any]:
    """Evaluate on questions whose manuals are all indexed; save reports/agent_eval.json.

    questions_path picks the question file; by default it's the eval set from the config.
    """
    from . import pipeline
    from .agent import Agent

    retriever = pipeline.build_retriever(cfg)
    indexed = set(pipeline.list_sources(retriever.collection))
    all_records = (load_questions(questions_path) if questions_path
                   else pipeline.load_eval_set(cfg.path("eval_set")))
    records = [r for r in all_records if all(m in indexed for m, _ in targets(r)[0])]
    if not records:
        raise RuntimeError("No eval questions match the indexed manuals.")
    if limit:
        records = records[:limit]

    agent = (agent_factory or Agent)(cfg, retriever=retriever)
    result = evaluate_agent(agent, retriever, records)
    llm = cfg.section("llm") or {}
    result["settings"] = {"model": llm.get("model", ""), "max_steps": agent.max_steps,
                          "top_k": agent.top_k, "manual_hint": False,
                          "questions_file": str(questions_path or cfg.path("eval_set"))}

    out = cfg.path("results_dir") / "agent_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["saved_to"] = str(out)
    return result


def log_to_mlflow(cfg, result: dict, run_name: str | None = None,
                  tags: dict[str, str] | None = None) -> str:
    """Log an agent evaluation to MLflow and return the run id."""
    from .tracking import run_params

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow isn't installed. Run: pip install mlflow") from exc

    tracking = cfg.section("tracking") or {}
    mlflow.set_tracking_uri(tracking.get("tracking_uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment(tracking.get("experiment_name", "manual-rag-retrieval"))
    summary = result["summary"]
    metrics = {f"agent.{k}": float(v) for k, v in summary.items() if k != "questions"}
    for qtype, scores in (result.get("by_type") or {}).items():
        metrics.update({f"agent.{qtype}.{k}": float(v) for k, v in scores.items()})
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(run_params(cfg))
        mlflow.log_params({f"agent.{k}": v for k, v in (result.get("settings") or {}).items()})
        mlflow.log_params({"eval.questions": summary["questions"]})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"stage": "agent-eval", **(tags or {})})
        path = cfg.path("results_dir") / "agent_eval.json"
        if path.exists():
            mlflow.log_artifact(str(path))
        return run.info.run_id
