"""GPU evaluation script, dry-run mode: real chunking, indexing, retrieval,
MLflow logging and registry promotion -- with a fake embedder and no LLM, so
it runs in CI. The judge path is exercised only on the GPU (Kaggle)."""

import json
import sys
from pathlib import Path

import mlflow
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import gpu_eval  # noqa: E402
from test_reindex_batch import FakeEmbedder  # noqa: E402


def article(n, ar, en, repealed=False):
    return {
        "article_number": n,
        "book": "",
        "chapter": "",
        "section": "",
        "topic": "",
        "text_ar": ar,
        "text_en": en,
        "is_repealed": repealed,
        "source_page": 1,
        "citation": f"Egyptian Civil Code, Article {n}",
    }


@pytest.fixture
def corpus_file(tmp_path):
    corpus = [
        article(
            147, "العقد شريعة المتعاقدين " * 5, "The contract makes the law of the parties. " * 5
        ),
        article(
            148,
            "يجب تنفيذ العقد طبقا لما اشتمل عليه " * 5,
            "A contract must be performed in good faith. " * 5,
        ),
        article(
            149,
            "(١) " + "نص " * 300 + "\n(٢) " + "نص " * 300,
            "(1) " + "text " * 300 + "\n(2) " + "text " * 300,
        ),
    ]
    f = tmp_path / "civil_code.json"
    f.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    return f


def run(tmp_path, corpus_file, *extra):
    return gpu_eval.main(
        [
            *extra,
            "--corpus", str(corpus_file),
            "--work-dir", str(tmp_path / "out"),
            "--skip-ragas",
            "--fake-generator",
            "--limit", "4",
            "--workers", "2",
        ],
        embed_model=FakeEmbedder(),
    )  # fmt: skip


def test_sweep_logs_one_mlflow_run_per_config(tmp_path, corpus_file):
    summary = run(tmp_path, corpus_file, "sweep")
    assert set(summary["runs"]) == {c.name for c in gpu_eval.SWEEP}
    assert len(gpu_eval.SWEEP) >= 5  # course checklist: >= 5 runs compared

    runs = mlflow.search_runs(experiment_names=["civil-code-rag-chunking"])
    assert len(runs) == len(gpu_eval.SWEEP)
    for col in ("params.chunk_size", "params.overlap", "params.embedding_model", "params.n_chunks"):
        assert col in runs.columns
    # different thresholds really produce different chunkings
    n_chunks = dict(zip(runs["params.config"], runs["params.n_chunks"]))
    assert n_chunks["split-400"] != n_chunks["whole-articles"]
    assert (tmp_path / "out" / "sweep_summary.json").exists()


def test_full_registers_best_config_as_production(tmp_path, corpus_file):
    run(tmp_path, corpus_file, "sweep")
    result = run(tmp_path, corpus_file, "full")

    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(gpu_eval.REGISTERED_MODEL, "production")
    assert mv.version == result["version"]
    assert mv.tags["config"] == result["config"]

    # the registered "model" hands back the exact params that were evaluated
    loaded = mlflow.pyfunc.load_model(f"models:/{gpu_eval.REGISTERED_MODEL}@production")
    params = loaded.predict(None)[0]
    cfg = next(c for c in gpu_eval.SWEEP if c.name == result["config"])
    assert params["chunking"]["paragraph_split_threshold_chars"] == cfg.paragraph_split_threshold

    report = json.loads((tmp_path / "out" / "reports" / "ragas_results.json").read_text())
    assert report["meta"]["config"] == result["config"]
    assert len(report["meta"]["corpus_md5"]) == 32
    assert report["rows"] and "_article_hit" in report["rows"][0]


def test_article_hit_rate_ignores_out_of_corpus_questions():
    rows = [{"_article_hit": True}, {"_article_hit": False}, {"_article_hit": None}]
    assert gpu_eval.article_hit_rate(rows) == 0.5
    assert gpu_eval.article_hit_rate([{"_article_hit": None}]) is None


def test_best_of_prefers_faithfulness_then_precision():
    def r(name, f, p, h):
        return {
            "config": gpu_eval.ChunkConfig(name, 1),
            "scores": {"faithfulness": f, "context_precision": p, "article_hit_rate": h},
        }

    best = gpu_eval.best_of([r("a", 0.8, 0.9, 0.5), r("b", 0.9, 0.1, 0.1), r("c", 0.9, 0.5, 0.2)])
    assert best["config"].name == "c"


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path, monkeypatch):
    # gpu_eval.main sets its own sqlite URI under --work-dir; make sure nothing
    # leaks into the developer's real ./mlruns
    monkeypatch.chdir(tmp_path)
    yield
    mlflow.set_tracking_uri(None)
