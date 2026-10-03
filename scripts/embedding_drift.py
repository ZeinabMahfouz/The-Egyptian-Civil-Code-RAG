"""Query-embedding drift: are users asking what the system was built and
evaluated for?

    python scripts/embedding_drift.py                    # BGE-M3 on CPU, ~1-2 min
    python scripts/embedding_drift.py --windows my_queries.json

Baseline: the 54 evaluation questions (tests/eval/eval_questions.json), the
query distribution the RAGAS scores were measured on. A "window" is a batch
of incoming queries. For each window:

    centroid_cosine = cos(mean embedding of the window, mean embedding of the baseline)
    drift           = 1 - centroid_cosine

How much drift is too much? The threshold isn't picked by hand. It's
calibrated from the baseline itself: draw n baseline questions at random
(n = the window size), compare their centroid with the centroid of the rest,
repeat 1000 times. That shows how much two samples of *the same*
distribution differ by chance. The threshold is the 99th percentile of that
drift, so a window above it is less similar to the baseline than 99% of
random samples of the baseline itself.

Outputs reports/drift.json and reports/drift.md, logs one MLflow run per
window (experiment "civil-code-rag-drift"), and the API exports the result
as Prometheus gauges (metrics.load_query_drift), with an alert rule in
deploy/monitoring/prometheus/alerts.yml.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "tests" / "eval" / "eval_questions.json"
WINDOWS = REPO / "tests" / "eval" / "drift_windows.json"
EXPERIMENT = "civil-code-rag-drift"


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


def to_markdown(results: dict, model: str, n_baseline: int) -> str:
    lines = [
        "# Query-embedding drift",
        "",
        f"Baseline: {n_baseline} evaluation questions. Embeddings: {model}. "
        "Threshold: 99th percentile of drift between random baseline samples of the same size.",
        "",
        "| Window | Queries | Centroid cosine | Drift | Threshold | Drifted? "
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


def main(argv=None, embed_model=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--windows", type=Path, default=WINDOWS)
    ap.add_argument("--embedding-model", default="BAAI/bge-m3")
    ap.add_argument("--reports-dir", type=Path, default=REPO / "reports")
    ap.add_argument("--mlflow-uri", default="sqlite:///eval_out/mlflow.db")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--trials", type=int, default=1000)
    args = ap.parse_args(argv)

    if embed_model is None:
        from sentence_transformers import SentenceTransformer

        embed_model = SentenceTransformer(args.embedding_model)

    def embed(texts):
        return np.asarray(embed_model.encode(texts, normalize_embeddings=True), dtype=np.float64)

    baseline_q = [
        q["question"] for q in json.loads(BASELINE.read_text(encoding="utf-8"))["questions"]
    ]
    windows_q = json.loads(args.windows.read_text(encoding="utf-8"))["windows"]
    baseline = embed(baseline_q)
    windows = {name: embed(w["queries"]) for name, w in windows_q.items()}

    results = measure(baseline, windows, trials=args.trials)
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
