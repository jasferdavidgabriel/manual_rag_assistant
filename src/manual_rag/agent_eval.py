"""Evaluate the tool calling agent on the labelled questions.

For every question the agent gets no manual hint, so it has to decide where
and how to search. Each question is scored on:

    found  any section the agent looked at is on a correct page of the right manual
    cited  the answer mentions a correct page number
    cost   searches, model calls and seconds

As a reference, the same question is also run as one plain search over all
manuals (top k, no language model), which is what the regular pipeline does.
"""

from __future__ import annotations

import json
import re
import time
from statistics import mean
from typing import Any, Callable

PAGE_RE = re.compile(r"\bpages?\s*(\d+)", re.IGNORECASE)


def is_correct(hit, record: dict) -> bool:
    return hit.source == record["manual"] and hit.page in set(record["expected_pages"])


def cited_pages(answer: str) -> set[int]:
    """Page numbers mentioned in an answer: "page 9" or "pages 7" -> {9, 7}."""
    return {int(n) for n in PAGE_RE.findall(answer or "")}


def evaluate_agent(agent, retriever, records: list[dict], retry_wait: float = 30.0,
                   sleep: Callable[[float], None] = time.sleep,
                   clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """Run the agent on every record; a failed question is retried once after a pause."""
    rows = []
    for rec in records:
        expected = set(rec["expected_pages"])
        baseline = retriever.retrieve(rec["question"], top_k=agent.top_k)
        row = {"question": rec["question"], "manual": rec["manual"],
               "expected_pages": sorted(expected),
               "baseline_found": any(is_correct(h, rec) for h in baseline)}

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
            row.update(found=False, cited=False, searches=0, model_calls=0,
                       seconds=None, steps=[], answer="", error=error)
        else:
            row.update(
                found=any(is_correct(h, rec) for h in result.hits),
                cited=bool(cited_pages(result.answer) & expected),
                searches=sum(1 for s in result.steps if s.tool == "search_manual"),
                model_calls=result.model_calls,
                seconds=round(seconds, 2),
                steps=[{"tool": s.tool, "arguments": s.arguments} for s in result.steps],
                answer=result.answer[:500],
                error=None,
            )
        rows.append(row)

    ok = [r for r in rows if not r["error"]]
    n = len(rows)

    def share(key):
        return round(sum(r[key] for r in rows) / n, 4) if n else 0.0

    def average(key):
        return round(mean(r[key] for r in ok), 2) if ok else 0.0

    summary = {
        "questions": n,
        "found_rate": share("found"),
        "cited_rate": share("cited"),
        "baseline_found_rate": share("baseline_found"),
        "avg_searches": average("searches"),
        "avg_model_calls": average("model_calls"),
        "avg_seconds": average("seconds"),
        "errors": n - len(ok),
    }
    return {"summary": summary, "per_question": rows}


def run(cfg, limit: int | None = None, agent_factory=None) -> dict[str, Any]:
    """Evaluate on the indexed manuals' questions and save reports/agent_eval.json."""
    from . import pipeline
    from .agent import Agent

    retriever = pipeline.build_retriever(cfg)
    indexed = set(pipeline.list_sources(retriever.collection))
    records = [r for r in pipeline.load_eval_set(cfg.path("eval_set"))
               if r.get("manual", "") in indexed]
    if not records:
        raise RuntimeError("No eval questions match the indexed manuals.")
    if limit:
        records = records[:limit]

    agent = (agent_factory or Agent)(cfg, retriever=retriever)
    result = evaluate_agent(agent, retriever, records)
    llm = cfg.section("llm") or {}
    result["settings"] = {"model": llm.get("model", ""), "max_steps": agent.max_steps,
                          "top_k": agent.top_k, "manual_hint": False}

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
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(run_params(cfg))
        mlflow.log_params({f"agent.{k}": v for k, v in (result.get("settings") or {}).items()})
        mlflow.log_params({"eval.questions": summary["questions"]})
        mlflow.log_metrics({f"agent.{k}": float(v) for k, v in summary.items() if k != "questions"})
        mlflow.set_tags({"stage": "agent-eval", **(tags or {})})
        path = cfg.path("results_dir") / "agent_eval.json"
        if path.exists():
            mlflow.log_artifact(str(path))
        return run.info.run_id
