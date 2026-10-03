"""Query-embedding drift (scripts/embedding_drift.py): the measure, the
calibrated threshold, the reports, and the Prometheus export. Synthetic
vectors -- no model download."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import embedding_drift as ed  # noqa: E402

from egyptian_civil_code_rag import metrics  # noqa: E402

DIM = 32


def cluster(center, n, spread, seed):
    rng = np.random.default_rng(seed)
    return center + spread * rng.standard_normal((n, DIM))


@pytest.fixture
def baseline():
    return cluster(np.eye(DIM)[0] * 3, 54, 1.0, seed=1)


def test_same_distribution_is_not_drift_shifted_one_is(baseline):
    windows = {
        "same": cluster(np.eye(DIM)[0] * 3, 16, 1.0, seed=2),
        "shifted": cluster(np.eye(DIM)[1] * 3, 16, 1.0, seed=3),
    }
    r = ed.measure(baseline, windows, trials=300)
    assert not r["same"]["drifted"]
    assert r["shifted"]["drifted"]
    assert r["shifted"]["drift"] > r["same"]["drift"]
    assert (
        r["same"]["nearest_baseline_similarity_mean"]
        > r["shifted"]["nearest_baseline_similarity_mean"]
    )


def test_threshold_is_larger_for_smaller_windows(baseline):
    # fewer queries -> noisier centroid -> more drift by chance alone
    assert ed.calibrate_threshold(baseline, 5, trials=300) > ed.calibrate_threshold(
        baseline, 25, trials=300
    )


def test_identical_window_has_zero_drift(baseline):
    assert ed.centroid_cosine(baseline, baseline) == pytest.approx(1.0)


class TopicEmbedder:
    """Weather questions point one way, everything else (the legal eval
    questions included) another, plus noise."""

    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % 2**32)
            v = 0.3 * rng.standard_normal(DIM)
            v[1 if "weather" in t.lower() else 0] += 3
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def test_end_to_end_writes_reports(tmp_path):
    windows = tmp_path / "w.json"
    windows.write_text(
        json.dumps(
            {
                "windows": {
                    "legal": {
                        "queries": [
                            f"What does Article {i} of the contract law say?" for i in range(12)
                        ]
                    },
                    "weather": {
                        "queries": [f"Weather in city number {i} tomorrow?" for i in range(12)]
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    report = ed.main(
        [
            "--windows",
            str(windows),
            "--reports-dir",
            str(tmp_path),
            "--no-mlflow",
            "--trials",
            "200",
        ],
        embed_model=TopicEmbedder(),
    )
    assert report["windows"]["weather"]["drifted"]
    assert report["windows"]["weather"]["drift"] > report["windows"]["legal"]["drift"]
    md = (tmp_path / "drift.md").read_text()
    assert "| weather | 12 |" in md and "**YES**" in md
    assert json.loads((tmp_path / "drift.json").read_text())["baseline"]["n_queries"] == 54


def test_drift_report_exported_as_gauges(tmp_path):
    from prometheus_client import REGISTRY

    f = tmp_path / "drift.json"
    f.write_text(json.dumps({"windows": {"w1": {"drift": 0.12, "threshold": 0.03}}}))
    assert metrics.load_query_drift(f) is not None
    assert REGISTRY.get_sample_value("rag_query_embedding_drift", {"window": "w1"}) == 0.12
    assert (
        REGISTRY.get_sample_value("rag_query_embedding_drift_threshold", {"window": "w1"}) == 0.03
    )
    assert metrics.load_query_drift(tmp_path / "missing.json") is None


def test_windows_file_is_disjoint_from_eval_set():
    evalq = {
        q["question"]
        for q in json.loads((REPO / "tests/eval/eval_questions.json").read_text(encoding="utf-8"))[
            "questions"
        ]
    }
    windows = json.loads((REPO / "tests/eval/drift_windows.json").read_text(encoding="utf-8"))[
        "windows"
    ]
    for w in windows.values():
        assert not evalq & set(w["queries"])
        assert len(w["queries"]) >= 10
