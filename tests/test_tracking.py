"""Tests for MLflow tracking. A fake mlflow module records the calls,
so MLflow doesn't need to be installed."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from manual_rag import tracking  # noqa: E402


class FakeCfg:
    def __init__(self, root: Path, tracking_section: dict | None = None):
        self.root = root
        self.sections = {
            "chunking": {"max_lines_per_chunk": 30, "keep_languages": ["en"], "contextual_headers": True},
            "embedding": {"model_name": "all-MiniLM-L6-v2"},
            "vector_store": {"distance": "cosine"},
            "retrieval": {"top_k": 5},
            "llm": {"api_key_env": "NVIDIA_API_KEY"},  # must NOT be logged
            "tracking": tracking_section or {},
        }

    def section(self, name):
        return self.sections.get(name, {})

    def path(self, name):
        return self.root / name


RESULT = {
    "manuals": ["ge.pdf", "coffee.pdf"],
    "within_manual": {"metrics": {"hit@1": 0.94, "hit@5": 1.0, "mrr": 0.96},
                      "misses": [{"q": "a"}, {"q": "b"}], "per_question": [{}] * 34},
    "all_manuals": {"metrics": {"hit@1": 0.94, "mrr": 0.95}, "misses": [], "per_question": [{}] * 34},
}


class FakeMlflow:
    def __init__(self):
        self.calls = {"params": {}, "metrics": {}, "tags": {}, "artifacts": []}

    def set_tracking_uri(self, uri):
        self.calls["uri"] = uri

    def set_experiment(self, name):
        self.calls["experiment"] = name

    def start_run(self, run_name=None):
        self.calls["run_name"] = run_name
        fake = self

        class Run:
            info = SimpleNamespace(run_id="run123")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                fake.calls["ended"] = True

        return Run()

    def log_params(self, params):
        self.calls["params"].update(params)

    def log_metrics(self, metrics):
        self.calls["metrics"].update(metrics)

    def set_tags(self, tags):
        self.calls["tags"].update(tags)

    def log_artifact(self, path):
        self.calls["artifacts"].append(Path(path).name)


def test_params_cover_retrieval_settings_only(tmp_path):
    params = tracking.run_params(FakeCfg(tmp_path))
    assert params["chunking.max_lines_per_chunk"] == 30
    assert params["chunking.keep_languages"] == "en"
    assert params["embedding.model_name"] == "all-MiniLM-L6-v2"
    assert params["retrieval.top_k"] == 5
    assert not any(k.startswith("llm.") for k in params)  # no secrets or unrelated settings


def test_metric_names_are_mlflow_safe():
    assert tracking.metric_name("within_manual", "hit@1") == "within_manual.hit_at_1"
    assert tracking.metric_name("all_manuals", "mrr") == "all_manuals.mrr"


def test_run_metrics_includes_scores_and_misses():
    metrics = tracking.run_metrics(RESULT)
    assert metrics["within_manual.hit_at_1"] == 0.94
    assert metrics["within_manual.mrr"] == 0.96
    assert metrics["all_manuals.hit_at_1"] == 0.94
    assert metrics["within_manual.misses"] == 2.0
    assert metrics["all_manuals.misses"] == 0.0


def test_run_metrics_handles_nested_scores():
    result = {"within_manual": {"metrics": {"hit_at_k": {"1": 0.8, "5": 1.0}, "mrr": 0.9}}}
    metrics = tracking.run_metrics(result)
    assert metrics["within_manual.hit_at_k.1"] == 0.8
    assert metrics["within_manual.mrr"] == 0.9


def test_log_evaluation_records_a_full_run(monkeypatch, tmp_path):
    (tmp_path / "results_dir").mkdir()
    (tmp_path / "results_dir" / "retrieval_metrics.json").write_text("{}")
    (tmp_path / "figures_dir").mkdir()
    (tmp_path / "figures_dir" / "retrieval_hit_at_k.png").write_bytes(b"png")
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    cfg = FakeCfg(tmp_path, {"tracking_uri": "sqlite:///test.db", "experiment_name": "exp"})
    run_id = tracking.log_evaluation(cfg, RESULT, run_name="headers on", tags={"note": "baseline"})

    assert run_id == "run123"
    assert fake.calls["uri"] == "sqlite:///test.db"
    assert fake.calls["experiment"] == "exp"
    assert fake.calls["run_name"] == "headers on"
    assert fake.calls["params"]["chunking.contextual_headers"] is True
    assert fake.calls["params"]["eval.questions"] == 34
    assert fake.calls["params"]["eval.manuals"] == "ge.pdf,coffee.pdf"
    assert fake.calls["metrics"]["within_manual.hit_at_1"] == 0.94
    assert fake.calls["tags"] == {"stage": "retrieval-eval", "note": "baseline"}
    # only files that exist are logged (the second chart is missing here)
    assert fake.calls["artifacts"] == ["retrieval_metrics.json", "retrieval_hit_at_k.png"]
    assert fake.calls["ended"]


def test_log_evaluation_explains_missing_mlflow(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "mlflow", None)  # makes "import mlflow" fail
    with pytest.raises(RuntimeError, match="pip install mlflow"):
        tracking.log_evaluation(FakeCfg(tmp_path), RESULT)
