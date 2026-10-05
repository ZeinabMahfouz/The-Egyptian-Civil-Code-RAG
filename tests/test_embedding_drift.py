"""Query-embedding drift (scripts/embedding_drift.py): the measure, the
calibrated threshold, the reports, and the Prometheus export. Synthetic
vectors -- no model download."""

import hashlib
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


@pytest.fixture(autouse=True)
def never_open_the_real_index(monkeypatch):
    # In CI (after dvc pull) and on a dev machine the real 1024-dim index
    # exists; these tests use 32-dim fake vectors and must never reach it.
    def fail(_params):
        raise AssertionError("test reached the real Qdrant index -- pass index= or --no-index")

    monkeypatch.setattr(ed, "open_index", fail)


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
            # hashlib, not hash(): Python randomizes str hashes per process,
            # which made this test pass or fail depending on the run (~3%).
            seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:4], "little")
            rng = np.random.default_rng(seed)
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
            "--no-index",
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


# --- off-corpus share: the alerting signal ---


def test_binomial_limit():
    # 16 queries at a 5% base rate: 3 low ones happen by chance (p~4%), 4 don't (p<1%)
    assert ed.binomial_limit(16) == 4
    assert ed.binomial_limit(100) > ed.binomial_limit(16)


def test_off_corpus_flags_questions_the_corpus_does_not_cover(tmp_path):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    emb = TopicEmbedder()
    client = QdrantClient(":memory:")
    client.create_collection("c", vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
    articles = [f"Article {i}: the contract binds the parties" for i in range(30)]
    client.upsert(
        "c",
        points=[PointStruct(id=i, vector=v.tolist()) for i, v in enumerate(emb.encode(articles))],
    )
    windows = tmp_path / "w.json"
    windows.write_text(
        json.dumps(
            {
                "windows": {
                    "legal": {
                        "queries": [f"Is contract {i} binding on both parties?" for i in range(16)]
                    },
                    "weather": {
                        "queries": [f"What is the weather in city {i}?" for i in range(16)]
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
            "100",
        ],
        embed_model=emb,
        index=(client, "c"),
    )
    w = report["windows"]
    assert not w["legal"]["off_corpus"] and w["legal"]["low_similarity_queries"] < 4
    assert w["weather"]["off_corpus"] and w["weather"]["low_similarity_queries"] == 16
    md = (tmp_path / "drift.md").read_text()
    assert "## Off-corpus share (alerting signal)" in md and "| weather | 16 |" in md
