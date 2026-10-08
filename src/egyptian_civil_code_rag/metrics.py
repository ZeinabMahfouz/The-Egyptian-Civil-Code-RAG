"""Prometheus metrics for /ask, scraped from GET /metrics on the FastAPI app.

Recorded in one place (TracedPipeline) so they can't drift from what the
Langfuse traces show. Grafana derives the rest from these series:

    request rate      rate(rag_requests_total[5m])
    p95 latency       histogram_quantile(0.95, rate(rag_request_latency_seconds_bucket[5m]))
    tokens / minute   rate(rag_tokens_total[5m]) * 60
    cost / hour       rate(rag_tokens_total[5m]) * 3600 / 1000 * <price per 1k tokens>

Label values are small, fixed sets (service, stage, status, direction, PII
entity type) -- never the question, an article number or anything a user
typed, which would both leak data and explode Prometheus's series count.
"""

import json
import math
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram

from egyptian_civil_code_rag.refusal import is_in_corpus

# CPU generation takes ~30s-2min; GPU/vLLM should land well under 10s.
# Buckets cover both, so the same histogram shows the before/after.
LATENCY_BUCKETS = (0.1, 0.5, 1, 2.5, 5, 10, 20, 30, 45, 60, 90, 120, 180, 300)

REQUESTS = Counter(
    "rag_requests_total",
    "Completed /ask requests",
    ["service", "status"],  # status: ok | no_context | refused | error
)
LATENCY = Histogram(
    "rag_request_latency_seconds",
    "Latency per pipeline stage (stage=total for the whole request)",
    ["service", "stage"],  # stage: total | retrieve | generate
    buckets=LATENCY_BUCKETS,
)
TIME_TO_FIRST_CHUNK = Histogram(
    "rag_time_to_first_chunk_seconds",
    "Streaming /ask: time from request start until the first answer text is sent. "
    "What a user actually waits for, versus total latency.",
    ["service"],
    buckets=LATENCY_BUCKETS,
)
TOKENS = Counter(
    "rag_tokens_total",
    "LLM tokens processed",
    ["service", "direction"],  # direction: input | output
)
PII_REDACTIONS = Counter(
    "rag_pii_redactions_total",
    "PII entities redacted from questions and answers",
    ["service", "where", "entity"],  # where: question | answer
)
RETRIEVAL_TOP_SCORE = Histogram(
    "rag_retrieval_top_score",
    "Cosine similarity of the best retrieved chunk -- a falling distribution "
    "means questions are drifting away from what the corpus covers",
    ["service"],
    buckets=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
# Labelled on purpose: a prometheus_client Gauge *without* labels is exported
# as 0.0 from the moment it's created -- which read as "faithfulness 0%" on
# the dashboard and fired the < 0.80 alert before any evaluation had run.
# A labelled gauge has no sample until .labels(...).set() is called.
RAGAS_FAITHFULNESS = Gauge(
    "rag_ragas_faithfulness",
    "Mean RAGAS faithfulness of the latest evaluation run. "
    "Absent until a run has produced real scores.",
    ["source"],
)
# Batch drift check (scripts/embedding_drift.py): drift of each query window
# from the evaluation questions, and the threshold calibrated for that window
# size. Labelled for the same reason as the faithfulness gauge: no sample
# until a drift report exists.
QUERY_DRIFT = Gauge(
    "rag_query_embedding_drift",
    "1 - cosine(centroid of a query window, centroid of the evaluation questions), BGE-M3",
    ["window"],
)
QUERY_DRIFT_THRESHOLD = Gauge(
    "rag_query_embedding_drift_threshold",
    "99th percentile of drift between random evaluation-set samples of the window's size",
    ["window"],
)
# The alerting signal (docs/decisions.md, Query drift): the share of a
# window's queries whose best match among the indexed articles is below the
# corpus cut-off, and the share at which the binomial test flags the window.
# Centroid drift above is kept as a reported signal only -- it reacts to
# phrasing and flagged ordinary Civil Code questions.
OFF_CORPUS_SHARE = Gauge(
    "rag_query_off_corpus_share",
    "Share of a query window with no close match among the indexed articles",
    ["window"],
)
OFF_CORPUS_LIMIT = Gauge(
    "rag_query_off_corpus_limit",
    "Share at which the window is flagged (binomial test, 1% false-alarm rate)",
    ["window"],
)


def load_query_drift(path: Path = Path("reports/drift.json")) -> dict | None:
    """Exports the latest drift report as gauges; None if there isn't one."""
    try:
        windows = json.loads(path.read_text(encoding="utf-8"))["windows"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    for name, w in windows.items():
        QUERY_DRIFT.labels(name).set(w["drift"])
        QUERY_DRIFT_THRESHOLD.labels(name).set(w["threshold"])
        if "off_corpus_share" in w:  # reports from before the off-corpus check lack it
            OFF_CORPUS_SHARE.labels(name).set(w["off_corpus_share"])
            OFF_CORPUS_LIMIT.labels(name).set(w["low_similarity_limit"] / w["n_queries"])
    return windows


def load_ragas_faithfulness(path: Path = Path("reports/ragas_results.json")) -> float | None:
    """Sets the faithfulness gauge from the latest RAGAS run, if it has real
    scores. The current file comes from the CPU run whose scores are all
    null -- that must leave the gauge unset (no sample, alert stays quiet),
    not report 0.0 and page someone about a judge that never ran."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    # GPU runs write {"meta", "aggregate", "rows"}; the old CPU run a bare list.
    rows = data.get("rows", []) if isinstance(data, dict) else data
    scores = [
        r["faithfulness"]
        for r in rows
        if isinstance(r, dict)
        and is_in_corpus(r)  # a declined out-of-corpus question isn't a faithfulness sample
        and isinstance(r.get("faithfulness"), (int, float))
        and not math.isnan(r["faithfulness"])
    ]
    if not scores:
        return None
    mean = sum(scores) / len(scores)
    RAGAS_FAITHFULNESS.labels(path.name).set(mean)
    return mean
