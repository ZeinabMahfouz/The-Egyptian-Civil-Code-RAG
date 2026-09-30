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

# CPU generation takes ~30s-2min; GPU/vLLM should land well under 10s.
# Buckets cover both, so the same histogram shows the before/after.
LATENCY_BUCKETS = (0.1, 0.5, 1, 2.5, 5, 10, 20, 30, 45, 60, 90, 120, 180, 300)

REQUESTS = Counter(
    "rag_requests_total",
    "Completed /ask requests",
    ["service", "status"],  # status: ok | no_context | error
)
LATENCY = Histogram(
    "rag_request_latency_seconds",
    "Latency per pipeline stage (stage=total for the whole request)",
    ["service", "stage"],  # stage: total | retrieve | generate
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


def load_ragas_faithfulness(path: Path = Path("reports/ragas_results.json")) -> float | None:
    """Sets the faithfulness gauge from the latest RAGAS run, if it has real
    scores. The current file comes from the CPU run whose scores are all
    null -- that must leave the gauge unset (no sample, alert stays quiet),
    not report 0.0 and page someone about a judge that never ran."""
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    scores = [
        r["faithfulness"]
        for r in rows
        if isinstance(r, dict)
        and isinstance(r.get("faithfulness"), (int, float))
        and not math.isnan(r["faithfulness"])
    ]
    if not scores:
        return None
    mean = sum(scores) / len(scores)
    RAGAS_FAITHFULNESS.labels(path.name).set(mean)
    return mean
