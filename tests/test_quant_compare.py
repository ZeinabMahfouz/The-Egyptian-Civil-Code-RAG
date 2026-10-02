"""AWQ vs fp16 comparison (scripts/quant_compare.py), run end to end with a
fake embedder, fake streaming models and a fake judge -- no GPU. The real
run is notebooks/kaggle_quantization.ipynb."""

import json
import sys
from pathlib import Path

import mlflow
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import quant_compare  # noqa: E402
from test_gpu_eval import article  # noqa: E402
from test_reindex_batch import FakeEmbedder  # noqa: E402


def fake_model(name, answer="Article 147 says the contract makes the law of the parties."):
    pieces = answer.split(" ")

    def gen(prompt):
        gen.last_usage = {"input": 100, "output": len(pieces)}
        return answer

    def stream(prompt):
        for i, p in enumerate(pieces):
            yield p if i == 0 else " " + p
        gen.last_usage = {"input": 100, "output": len(pieces)}

    gen.stream, gen.model_name, gen.last_usage = stream, name, None
    return gen


def fake_judge(faithfulness):
    def score(rows, *args, **kwargs):
        for r in rows:
            r.update(
                faithfulness=faithfulness,
                context_precision=0.9,
                context_recall=0.8,
                answer_relevancy=0.7,
            )
        return rows

    return score


@pytest.fixture
def corpus_file(tmp_path):
    corpus = [
        article(
            147, "العقد شريعة المتعاقدين " * 5, "The contract makes the law of the parties. " * 5
        ),
        article(148, "يجب تنفيذ العقد " * 5, "A contract must be performed in good faith. " * 5),
    ]
    f = tmp_path / "civil_code.json"
    f.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    return f


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    mlflow.set_tracking_uri(None)


def run_all(work, corpus_file, f16, fawq):
    common = ["--work-dir", str(work / "out")]
    for label in ("awq", "fp16"):
        quant_compare.main(
            ["generate", "--label", label, "--model", f"m-{label}", "--corpus", str(corpus_file),
             "--limit", "6", "--concurrency", "3", *common],
            embed_model=FakeEmbedder(),
            generate_fn=fake_model(f"m-{label}"),
        )  # fmt: skip
    quant_compare.main(["score", "--labels", "fp16", *common], score_fn=fake_judge(f16))
    quant_compare.main(["score", "--labels", "awq", *common], score_fn=fake_judge(fawq))
    return quant_compare.main(["report", *common, "--reports-dir", str(work / "reports")])


def test_generate_records_latency_per_question(work, corpus_file):
    out = quant_compare.main(
        ["generate", "--label", "awq", "--model", "m", "--corpus", str(corpus_file),
         "--limit", "4", "--concurrency", "2", "--work-dir", str(work / "out")],
        embed_model=FakeEmbedder(),
        generate_fn=fake_model("m"),
    )  # fmt: skip
    answered = [r for r in out["rows"] if r["_sources"]]
    assert answered and all(r["_latency_s"] > 0 and r["_ttft_s"] is not None for r in answered)
    assert all(r["_output_tokens"] == 11 for r in answered)  # 11 words streamed
    assert out["meta"]["concurrent"]["n_requests"] == len(answered)
    # RAGAS-facing fields are the same shape as the main evaluation
    assert {"user_input", "retrieved_contexts", "response", "reference"} <= set(out["rows"][0])


def test_small_drop_passes_and_is_logged(work, corpus_file):
    c = run_all(work, corpus_file, f16=0.90, fawq=0.88)
    assert c["faithfulness_drop"] == pytest.approx(0.02) and c["passed"]
    md = (work / "reports" / "quantization.md").read_text()
    assert "PASS" in md and "| faithfulness | 0.900 | 0.880 | -0.020 |" in md
    assert json.loads((work / "reports" / "quantization.json").read_text())["passed"]

    runs = mlflow.search_runs(experiment_names=[quant_compare.EXPERIMENT])
    assert set(runs["tags.mlflow.runName"]) == {"quant-fp16", "quant-awq"}
    assert "metrics.latency_p50_s" in runs.columns
    awq = runs[runs["tags.mlflow.runName"] == "quant-awq"].iloc[0]
    assert awq["metrics.faithfulness_drop_vs_fp16"] == pytest.approx(0.02)


def test_drop_at_threshold_fails(work, corpus_file):
    c = run_all(work, corpus_file, f16=0.90, fawq=0.85)
    assert not c["passed"]
    assert "FAIL" in (work / "reports" / "quantization.md").read_text()


def test_weights_parsed_from_vllm_log(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text("INFO ... Model loading took 2.85 GiB and 40.1 seconds\n")
    assert quant_compare.weights_gib_from_log(log) == 2.85
    log.write_text("INFO Loading model weights took 7.6719 GB\n")
    assert quant_compare.weights_gib_from_log(log) == pytest.approx(7.6719)
    assert quant_compare.weights_gib_from_log(tmp_path / "missing.log") is None


def test_percentile():
    assert quant_compare.percentile([1, 2, 3, 4], 50) == 2
    assert quant_compare.percentile([1, 2, 3, 4], 95) == 4
    assert quant_compare.percentile([None], 50) is None
