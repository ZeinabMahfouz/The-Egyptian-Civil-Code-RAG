"""Prometheus metrics: every /ask updates the counters Grafana is built on,
and /metrics serves them in Prometheus's text format.

Prometheus metrics are process-global, so each test measures the *change*
in a series across its own requests rather than absolute values.
"""

import json

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from test_api import FakeEngine

from egyptian_civil_code_rag import metrics
from egyptian_civil_code_rag.api import create_app
from egyptian_civil_code_rag.pii import PIIGuard
from egyptian_civil_code_rag.pipeline import TracedPipeline, make_langfuse


def value(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


class TokenEngine(FakeEngine):
    def __init__(self):
        def gen(prompt):
            return "answer"

        gen.last_usage = {"input": 100, "output": 25}
        self.generate_fn = gen


def pipeline(engine, service="test"):
    return TracedPipeline(engine, PIIGuard(use_guardrails=False), make_langfuse(), service)


def test_request_latency_and_tokens_counted():
    svc = "t-basic"
    p = pipeline(TokenEngine(), svc)
    for _ in range(3):
        p.ask("What does Article 147 say?")

    assert value("rag_requests_total", service=svc, status="ok") == 3
    assert value("rag_request_latency_seconds_count", service=svc, stage="total") == 3
    assert value("rag_request_latency_seconds_count", service=svc, stage="retrieve") == 3
    assert value("rag_request_latency_seconds_count", service=svc, stage="generate") == 3
    assert value("rag_tokens_total", service=svc, direction="input") == 300
    assert value("rag_tokens_total", service=svc, direction="output") == 75
    assert value("rag_retrieval_top_score_count", service=svc) == 3


def test_no_context_is_its_own_status():
    class EmptyEngine(FakeEngine):
        def retrieve(self, question):
            return []

    svc = "t-empty"
    pipeline(EmptyEngine(), svc).ask("Q?")
    assert value("rag_requests_total", service=svc, status="no_context") == 1
    assert value("rag_request_latency_seconds_count", service=svc, stage="generate") == 0


def test_errors_counted_and_reraised():
    class BrokenEngine(FakeEngine):
        def generate_fn(self, prompt):
            raise RuntimeError("model crashed")

    svc = "t-error"
    with pytest.raises(RuntimeError):
        pipeline(BrokenEngine(), svc).ask("Q?")
    assert value("rag_requests_total", service=svc, status="error") == 1
    # latency still recorded for the failed request
    assert value("rag_request_latency_seconds_count", service=svc, stage="total") == 1


def test_pii_counted_by_type_never_by_value():
    class LeakyEngine(FakeEngine):
        def generate_fn(self, prompt):
            return "Call 01012345678."

    svc = "t-pii"
    pipeline(LeakyEngine(), svc).ask("رقمي القومي 29801011234567")
    assert (
        value("rag_pii_redactions_total", service=svc, where="question", entity="EG_NATIONAL_ID")
        == 1
    )
    assert value("rag_pii_redactions_total", service=svc, where="answer", entity="EG_PHONE") == 1
    exposition = TestClient(create_app(engine=FakeEngine())).get("/metrics").text
    assert "29801011234567" not in exposition and "01012345678" not in exposition


def test_metrics_endpoint_serves_prometheus_format():
    client = TestClient(create_app(engine=FakeEngine()))
    client.post("/ask", json={"question": "Q?"})
    resp = client.get("/metrics")
    assert resp.status_code == 200  # a route, not a mount: no 307 to /metrics/
    assert resp.headers["content-type"].startswith("text/plain")
    for series in (
        "rag_requests_total",
        "rag_request_latency_seconds_bucket",
        "rag_tokens_total",
        "rag_retrieval_top_score_bucket",
    ):
        assert series in resp.text
    assert 'service="fastapi"' in resp.text


def test_faithfulness_not_exported_before_any_real_run():
    """Regression: an unlabelled Gauge is exported as 0.0 from creation, so
    the dashboard showed "0.00%" and RagFaithfulnessLow fired with no
    evaluation ever run. /metrics must carry no faithfulness *sample* until
    real scores exist (HELP/TYPE lines are fine)."""
    from prometheus_client import CollectorRegistry, Gauge, generate_latest

    reg = CollectorRegistry()
    g = Gauge("f", "doc", ["source"], registry=reg)  # same shape as the real gauge
    assert (
        "\nf{" not in generate_latest(reg).decode() and "\nf " not in generate_latest(reg).decode()
    )
    g.labels("x").set(0.5)
    assert 'f{source="x"} 0.5' in generate_latest(reg).decode()

    exposition = TestClient(create_app(engine=FakeEngine())).get("/metrics").text
    samples = [
        line
        for line in exposition.splitlines()
        if line.startswith('rag_ragas_faithfulness{source="ragas_results.json"}')
    ]
    # The repo's committed report: no sample while it has no real scores,
    # its real in-corpus mean once a GPU evaluation is committed -- never 0.0.
    expected = metrics.load_ragas_faithfulness()
    if expected is None:
        assert samples == []
    else:
        assert len(samples) == 1 and float(samples[0].split()[-1]) == pytest.approx(expected)
        assert expected > 0


def test_faithfulness_gauge_ignores_null_scores(tmp_path):
    # The CPU RAGAS run produced only nulls -- that must not become "0.0 faithfulness"
    f = tmp_path / "ragas.json"
    f.write_text(json.dumps([{"faithfulness": None}, {"faithfulness": None}]))
    assert metrics.load_ragas_faithfulness(f) is None


def test_faithfulness_gauge_from_real_scores(tmp_path):
    f = tmp_path / "ragas.json"
    f.write_text(json.dumps([{"faithfulness": 0.9}, {"faithfulness": 0.7}, {"faithfulness": None}]))
    assert metrics.load_ragas_faithfulness(f) == pytest.approx(0.8)
    assert value("rag_ragas_faithfulness", source="ragas.json") == pytest.approx(0.8)


def test_faithfulness_gauge_missing_file(tmp_path):
    assert metrics.load_ragas_faithfulness(tmp_path / "nope.json") is None


def test_faithfulness_gauge_ignores_out_of_corpus_rows(tmp_path):
    # A correctly declined out-of-corpus question scores ~0 in RAGAS; it is
    # gated by refusal rate, not averaged into faithfulness.
    f = tmp_path / "ragas.json"
    rows = [
        {"_category": "substantive", "faithfulness": 0.9},
        {"_category": "out_of_corpus", "faithfulness": 0.0},
    ]
    f.write_text(json.dumps({"meta": {}, "aggregate": {}, "rows": rows}))
    assert metrics.load_ragas_faithfulness(f) == pytest.approx(0.9)
