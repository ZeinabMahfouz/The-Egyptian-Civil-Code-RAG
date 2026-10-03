"""Query drift: are users asking what the system was built and evaluated for?

    python scripts/embedding_drift.py        # BGE-M3 on CPU + the local index, ~1-2 min
                                             # (stop the API first: local Qdrant is single-process)

Two signals per window (a batch of incoming queries), both with thresholds
calibrated from data rather than picked by hand:

1. Off-corpus share (the one that alerts). For each query, the cosine
   similarity of its best match among the indexed articles: "does the Civil
   Code have anything close to this?". The cut-off is the 5th percentile of
   that score over the substantive evaluation questions, so about 5% of
   normal questions fall below it by chance. A window is flagged when more of
   its queries fall below the cut-off than a 5% rate would produce with
   probability >= 1% (binomial test).

2. Centroid drift (reported, not alerted on). 1 - cosine between the mean
   embedding of the window and of the 54 evaluation questions, with a
   threshold from 1000 random baseline samples of the window's size. It ranks
   windows correctly, but it also reacts to phrasing: a third of the
   evaluation set is templated "What does Article N say?" lookups, so a
   window of ordinary full-sentence Civil Code questions already looks
   "drifted". See docs/decisions.md.

Outputs reports/drift.json and reports/drift.md, logs one MLflow run per
window (experiment "civil-code-rag-drift"), and the API exports the result
as Prometheus gauges (metrics.load_query_drift) for the RagQueryDrift alert.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "tests" / "eval" / "eval_questions.json"
WINDOWS = REPO / "tests" / "eval" / "drift_windows.json"
EXPERIMENT = "civil-code-rag-drift"
LOW_PERCENTILE = 5  # % of normal questions expected below the corpus cut-off
ALPHA = 0.01  # false-alarm rate of the window test


# --- centroid drift ---------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def centroid(vectors: np.ndarray) -> np.ndarray:
    return _unit(_unit(vectors).mean(axis=0))


def centroid_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(centroid(a) @ centroid(b))


def calibrate_threshold(
    baseline: np.ndarray, n: int, trials: int = 1000, percentile: float = 99, seed: int = 0
) -> float:
    """Drift between a random n-question sample of the baseline and the rest:
    the drift you see by chance alone. Returns its `percentile`."""
    rng = np.random.default_rng(seed)
    n = min(n, len(baseline) - 1)
    drifts = []
    for _ in range(trials):
        idx = rng.permutation(len(baseline))
        drifts.append(1 - centroid_cosine(baseline[idx[:n]], baseline[idx[n:]]))
    return float(np.percentile(drifts, percentile))


def nearest_baseline_similarity(window: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """For each window query, cosine to its closest baseline question."""
    return (_unit(window) @ _unit(baseline).T).max(axis=1)


def measure(baseline: np.ndarray, windows: dict, trials: int = 1000) -> dict:
    results = {}
    for name, vectors in windows.items():
        drift = 1 - centroid_cosine(vectors, baseline)
        threshold = calibrate_threshold(baseline, len(vectors), trials=trials)
        nn = nearest_baseline_similarity(vectors, baseline)
        results[name] = {
            "n_queries": int(len(vectors)),
            "centroid_cosine": 1 - drift,
            "drift": drift,
            "threshold": threshold,
            "drifted": bool(drift > threshold),
            "nearest_baseline_similarity_mean": float(nn.mean()),
        }
    return results


# --- off-corpus share ---------------------------------------------------------------


def binomial_limit(n: int, p: float = LOW_PERCENTILE / 100, alpha: float = ALPHA) -> int:
    """Smallest k with P(X >= k) <= alpha for X ~ Binomial(n, p): at least
    this many low-similarity queries out of n is unlikely under normal traffic."""
    for k in range(n + 1):
        tail = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))
        if tail <= alpha:
            return k
    return n + 1


def top_corpus_scores(vectors: np.ndarray, client, collection: str) -> np.ndarray:
    """Cosine similarity of each query's best match among the indexed articles."""
    return np.array(
        [
            client.query_points(collection, query=v.tolist(), limit=1).points[0].score
            for v in vectors
        ]
    )


def measure_corpus(baseline_scores: np.ndarray, window_scores: dict) -> dict:
    cut = float(np.percentile(baseline_scores, LOW_PERCENTILE))
    out = {}
    for name, scores in window_scores.items():
        n_low = int((scores < cut).sum())
        limit = binomial_limit(len(scores))
        out[name] = {
            "corpus_score_mean": float(scores.mean()),
            "corpus_cutoff": cut,
            "low_similarity_queries": n_low,
            "low_similarity_limit": limit,
            "off_corpus_share": n_low / len(scores),
            "off_corpus": bool(n_low >= limit),
        }
    return out


def open_index(params_path: Path):
    """(client, collection) for the local index, or None if it isn't there."""
    import yaml
    from qdrant_client import QdrantClient

    params = yaml.safe_load(params_path.read_text(encoding="utf-8"))["qdrant"]
    path = REPO / params["storage_path"]
    if not path.exists():
        print(f"[warn] no index at {path} -- skipping the off-corpus signal (run dvc pull)")
        return None
    return QdrantClient(path=str(path)), params["collection_name"]


# --- reporting ------------------------------------------------------------------


def to_markdown(results: dict, model: str, n_baseline: int) -> str:
    lines = [
        "# Query drift",
        "",
        f"Embeddings: {model}. Baseline: the {n_baseline} evaluation questions.",
        "",
    ]
    corpus = [r for r in results.values() if "off_corpus" in r]
    if corpus:
        lines += [
            "## Off-corpus share (alerting signal)",
            "",
            f"Cut-off: best-match similarity to the indexed articles below "
            f"{corpus[0]['corpus_cutoff']:.4f} (the {LOW_PERCENTILE}th percentile of the "
            f"substantive evaluation questions). Flagged when the count reaches the "
            f"binomial limit (false-alarm rate {ALPHA:.0%}).",
            "",
            "| Window | Queries | Mean best-match similarity | Below cut-off | Limit "
            "| Off-corpus? |",
            "|---|---|---|---|---|---|",
        ]
        for name, r in results.items():
            lines.append(
                f"| {name} | {r['n_queries']} | {r['corpus_score_mean']:.4f} | "
                f"{r['low_similarity_queries']} | {r['low_similarity_limit']} | "
                f"{'**YES**' if r['off_corpus'] else 'no'} |"
            )
        lines.append("")
    lines += [
        "## Centroid drift (reported, not alerted on)",
        "",
        "Threshold: 99th percentile of drift between random evaluation samples of the same size.",
        "",
        "| Window | Queries | Centroid cosine | Drift | Threshold | Above threshold? "
        "| Mean nearest-baseline similarity |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['n_queries']} | {r['centroid_cosine']:.4f} | {r['drift']:.4f} | "
            f"{r['threshold']:.4f} | {'**YES**' if r['drifted'] else 'no'} | "
            f"{r['nearest_baseline_similarity_mean']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def log_mlflow(results: dict, model: str, uri: str) -> None:
    import mlflow

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)
    for name, r in results.items():
        with mlflow.start_run(run_name=f"drift-{name}"):
            mlflow.log_params({"window": name, "embedding_model": model, "baseline": BASELINE.name})
            mlflow.log_metrics(
                {k: float(v) for k, v in r.items() if isinstance(v, (int, float, bool))}
            )


# --- entry point ------------------------------------------------------------------


def main(argv=None, embed_model=None, index=None):
    """index: (client, collection) to use instead of the local Qdrant index (tests)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--windows", type=Path, default=WINDOWS)
    ap.add_argument("--embedding-model", default="BAAI/bge-m3")
    ap.add_argument("--reports-dir", type=Path, default=REPO / "reports")
    ap.add_argument("--mlflow-uri", default="sqlite:///eval_out/mlflow.db")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--params", type=Path, default=REPO / "params.yaml")
    ap.add_argument("--no-index", action="store_true", help="skip the off-corpus signal")
    args = ap.parse_args(argv)

    if embed_model is None:
        from sentence_transformers import SentenceTransformer

        embed_model = SentenceTransformer(args.embedding_model)

    def embed(texts):
        return np.asarray(embed_model.encode(texts, normalize_embeddings=True), dtype=np.float64)

    questions = json.loads(BASELINE.read_text(encoding="utf-8"))["questions"]
    baseline_q = [q["question"] for q in questions]
    windows_q = json.loads(args.windows.read_text(encoding="utf-8"))["windows"]
    baseline = embed(baseline_q)
    windows = {name: embed(w["queries"]) for name, w in windows_q.items()}

    results = measure(baseline, windows, trials=args.trials)

    if index is None and not args.no_index:
        index = open_index(args.params)
    if index is not None:
        client, collection = index
        # Substantive questions only: article-number lookups are answered by an
        # exact filter, not by similarity, so their similarity says nothing.
        substantive = [i for i, q in enumerate(questions) if q["category"] == "substantive"]
        corpus = measure_corpus(
            top_corpus_scores(baseline[substantive], client, collection),
            {name: top_corpus_scores(v, client, collection) for name, v in windows.items()},
        )
        for name in results:
            results[name].update(corpus[name])

    report = {
        "baseline": {"source": str(BASELINE.relative_to(REPO)), "n_queries": len(baseline_q)},
        "embedding_model": args.embedding_model,
        "windows": results,
    }
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "drift.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = to_markdown(results, args.embedding_model, len(baseline_q))
    (args.reports_dir / "drift.md").write_text(md, encoding="utf-8")
    if not args.no_mlflow:
        Path("eval_out").mkdir(exist_ok=True)
        log_mlflow(results, args.embedding_model, args.mlflow_uri)
    print(md)
    return report


if __name__ == "__main__":
    sys.exit(main() and 0)
