"""Langfuse tracing: every /ask produces one trace with the expected spans,
and no raw PII ever lands in any span.

Spans are captured with OpenTelemetry's in-memory exporter instead of being
sent to a Langfuse server -- so this runs in CI with no server and no keys.
"""

import json
import uuid

import pytest
from fastapi.testclient import TestClient
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from test_api import FakeEngine

from egyptian_civil_code_rag.api import create_app
from egyptian_civil_code_rag.pii import PIIGuard
from egyptian_civil_code_rag.pipeline import TracedPipeline, make_langfuse

NATIONAL_ID = "29801011234567"
PHONE = "01012345678"


@pytest.fixture
def traced(monkeypatch):
    """A Langfuse client whose spans go to memory. Returns (langfuse, exporter)."""
    # Langfuse keeps one client per public key for the whole process -- a
    # second Langfuse(...) with the same key silently reuses the first one's
    # exporter. A unique key per test gives each test its own exporter.
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", f"pk-lf-test-{uuid.uuid4()}")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:9")  # never contacted
    exporter = InMemorySpanExporter()
    # Own TracerProvider (not the global one) so tests don't share state;
    # Langfuse attaches its span processor to it, exporting to memory.
    lf = Langfuse(tracer_provider=TracerProvider(), span_exporter=exporter, tracing_enabled=True)
    yield lf, exporter
    lf.shutdown()


def spans_by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


def all_span_text(exporter) -> str:
    """Every attribute of every span, as one string -- for leak checks."""
    return json.dumps(
        [dict(s.attributes or {}) for s in exporter.get_finished_spans()],
        ensure_ascii=False,
        default=str,
    )


def test_one_trace_with_all_stages(traced):
    lf, exporter = traced
    result = TracedPipeline(FakeEngine(), PIIGuard(use_guardrails=False), lf, "fastapi").ask(
        "What does Article 147 say?"
    )
    lf.flush()
    spans = spans_by_name(exporter)
    assert {"ask", "pii-input", "retrieve", "generate", "pii-output"} <= set(spans)
    # all five in the same trace, children under "ask"
    trace_ids = {s.context.trace_id for s in exporter.get_finished_spans()}
    assert len(trace_ids) == 1
    root = spans["ask"]
    for child in ("pii-input", "retrieve", "generate", "pii-output"):
        assert spans[child].parent.span_id == root.context.span_id
    assert result.trace_id is not None
    assert "Egyptian Civil Code, Article 1" in all_span_text(exporter)


def test_raw_pii_never_enters_any_span(traced):
    lf, exporter = traced

    class LeakyEngine(FakeEngine):
        def generate_fn(self, prompt):  # model echoes a phone number
            return f"Call {PHONE} about it."

    result = TracedPipeline(LeakyEngine(), PIIGuard(use_guardrails=False), lf, "fastapi").ask(
        f"رقمي القومي {NATIONAL_ID}، ما حكم المادة 147؟"
    )
    lf.flush()
    text = all_span_text(exporter)
    assert NATIONAL_ID not in text  # question side
    assert PHONE not in text  # answer side, including the generation span
    assert "[EG_NATIONAL_ID]" in text and "[EG_PHONE]" in text
    assert result.pii_redacted == ["EG_NATIONAL_ID", "EG_PHONE"]


def test_token_usage_recorded_on_generation(traced):
    lf, exporter = traced

    class CountingEngine(FakeEngine):
        def __init__(self):
            def gen(prompt):
                return "answer"

            gen.model_name = "fake-model"
            gen.last_usage = {"input": 120, "output": 30}
            self.generate_fn = gen

    result = TracedPipeline(CountingEngine(), PIIGuard(use_guardrails=False), lf, "bentoml").ask(
        "Q?"
    )
    lf.flush()
    gen_attrs = json.dumps(dict(spans_by_name(exporter)["generate"].attributes), default=str)
    assert "fake-model" in gen_attrs
    assert "120" in gen_attrs and "30" in gen_attrs
    assert result.usage == {"input": 120, "output": 30}


def test_no_hits_skips_generation(traced):
    lf, exporter = traced

    class EmptyEngine(FakeEngine):
        def retrieve(self, question):
            return []

    result = TracedPipeline(EmptyEngine(), PIIGuard(use_guardrails=False), lf, "fastapi").ask("Q?")
    lf.flush()
    assert "generate" not in spans_by_name(exporter)
    assert result.sources == []


def test_api_traces_every_ask(traced):
    lf, exporter = traced
    client = TestClient(create_app(engine=FakeEngine(), langfuse=lf))
    for _ in range(3):
        assert client.post("/ask", json={"question": "Q?"}).status_code == 200
    lf.flush()
    roots = [s for s in exporter.get_finished_spans() if s.name == "ask"]
    assert len(roots) == 3
    assert len({s.context.trace_id for s in roots}) == 3  # one trace per request


def test_tracing_off_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    lf = make_langfuse()
    # still fully usable -- the pipeline runs, nothing is exported
    result = TracedPipeline(FakeEngine(), PIIGuard(use_guardrails=False), lf, "fastapi").ask("Q?")
    assert result.answer.startswith("Fake answer")
