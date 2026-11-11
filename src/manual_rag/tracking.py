"""Experiment tracking with MLflow.

Each evaluation run is logged with the settings that produced it (params),
its retrieval scores (metrics) and its report files (artifacts), so runs can
be compared side by side in the MLflow UI:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

MLflow is imported only when a run is logged, so the rest of the project
(and the unit tests) work without it installed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Config sections whose values change retrieval results, so every run records them
PARAM_SECTIONS = ("chunking", "embedding", "vector_store", "retrieval")
MODES = ("within_manual", "all_manuals")


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """{"a": {"b": 1}} -> {"a.b": 1}; lists become comma separated strings."""
    out: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{name}."))
        elif isinstance(value, (list, tuple)):
            out[name] = ",".join(map(str, value))
        else:
            out[name] = value
    return out


def run_params(cfg) -> dict[str, Any]:
    """The settings that produced a run, e.g. {"chunking.max_lines_per_chunk": 30}."""
    params: dict[str, Any] = {}
    for section in PARAM_SECTIONS:
        params.update(_flatten(cfg.section(section) or {}, f"{section}."))
    return params


def metric_name(mode: str, key: str) -> str:
    """MLflow only allows letters, digits and _ - . / in names: "hit@1" -> "hit_at_1"."""
    key = str(key).replace("@", "_at_")
    return f"{mode}.{re.sub(r'[^A-Za-z0-9_./-]', '_', key)}"


def run_metrics(result: dict) -> dict[str, float]:
    """Every numeric score from an evaluate() result, plus the number of misses."""
    metrics: dict[str, float] = {}
    for mode in MODES:
        section = result.get(mode) or {}
        for key, value in _flatten(section.get("metrics") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[metric_name(mode, key)] = float(value)
        metrics[f"{mode}.misses"] = float(len(section.get("misses") or []))
    return metrics


def artifact_paths(cfg) -> list[Path]:
    """Report files written by evaluate(): the full results and the two charts."""
    figures = cfg.path("figures_dir")
    return [
        cfg.path("results_dir") / "retrieval_metrics.json",
        figures / "retrieval_hit_at_k.png",
        figures / "retrieval_hit_at_k_all_manuals.png",
    ]


def log_evaluation(cfg, result: dict, run_name: str | None = None,
                   tags: dict[str, str] | None = None) -> str:
    """Log one evaluation run to MLflow and return its run id."""
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow isn't installed. Run: pip install mlflow") from exc

    tracking = cfg.section("tracking") or {}
    mlflow.set_tracking_uri(tracking.get("tracking_uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment(tracking.get("experiment_name", "manual-rag-retrieval"))

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(run_params(cfg))
        mlflow.log_params({
            "eval.manuals": ",".join(result.get("manuals") or []),
            "eval.questions": len((result.get("within_manual") or {}).get("per_question") or []),
        })
        mlflow.log_metrics(run_metrics(result))
        mlflow.set_tags({"stage": "retrieval-eval", **(tags or {})})
        for path in artifact_paths(cfg):
            if path.exists():
                mlflow.log_artifact(str(path))
        return run.info.run_id
