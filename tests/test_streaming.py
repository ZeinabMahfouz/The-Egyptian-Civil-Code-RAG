"""Streaming /ask: tokens arrive progressively, PII is redacted even when a
number is split across tokens, and the stream is traced/measured like /ask."""

import json
import random
import uuid

import pytest
from fastapi.testclient import TestClient
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import REGISTRY
from test_api import FakeEngine

from egyptian_civil_code_rag.api import create_app
from egyptian_civil_code_rag.pii import StreamingRedactor, redact

PHONE = "01012345678"
NATIONAL_ID = "29801011234567"


class StreamingEngine(FakeEngine):
    """generate_fn with a .stream attribute, like transformers_backend's."""

    def __init__(self, answer="Article 147 says the contract makes the law of the parties."):
        pieces = [answer[i : i + 3] for i in range(0, len(answer), 3)]  # 3-char "tokens"

        def gen(prompt):
            return answer

        def stream(prompt):
            yield from pieces
            gen.last_usage = {"input": 50, "output": len(pieces)}

        gen.stream = stream
        gen.model_name = "fake-stream"
        gen.last_usage = None
        self.generate_fn = gen


def read_sse(resp):
    events = []
    for line in resp.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


# --- the incremental redactor ---


def test_streaming_redaction_equals_whole_text_redaction():
    text = (
        f"Call {PHONE[:3]} {PHONE[3:7]} {PHONE[7:]} or mail a.b@example.com. "
        f"ID ٢٩٨٠١٠١١٢٣٤٥٦٧. Article 147 applies. " * 3
    )
    rng = random.Random(0)
    for _ in range(200):
        r, out, i = StreamingRedactor(), "", 0
        while i < len(text):
            n = rng.randint(1, 7)
            out += r.feed(text[i : i + n])
            i += n
        out += r.flush()
        assert out == redact(text)[0]
    assert r.entities == {"EG_PHONE", "EMAIL", "EG_NATIONAL_ID"}


def test_no_prefix_of_a_split_number_is_released_early():
    r = StreamingRedactor()
    released = r.feed("x" * 100 + " call 010")  # number has only started
    assert "010" not in released
    released += r.feed("12345678 now")
    released += r.flush()
    assert PHONE not in released and "[EG_PHONE]" in released


# --- /ask/stream end to end ---


def test_stream_endpoint_sends_tokens_progressively_then_done():
    client = TestClient(create_app(engine=StreamingEngine("word " * 60)))
    with client.stream("POST", "/ask/stream", json={"question": "Q?"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = read_sse(resp)
    tokens = [e for e in events if e["type"] == "token"]
    assert len(tokens) > 1  # more than one piece => actually streamed
    assert events[-1]["type"] == "done"
    assert events[-1]["sources"] == ["Egyptian Civil Code, Article 1"]
    assert "".join(t["text"] for t in tokens).strip() == ("word " * 60).strip()


def test_stream_never_leaks_pii_from_the_model():
    answer = f"Contact the clerk on {PHONE} or check ID {NATIONAL_ID}. " + "padding " * 20
    client = TestClient(create_app(engine=StreamingEngine(answer)))
    with client.stream("POST", "/ask/stream", json={"question": f"my phone is {PHONE}"}) as resp:
        body = resp.read().decode()
    assert PHONE not in body and NATIONAL_ID not in body
    done = [json.loads(x[6:]) for x in body.split("\n\n") if x.startswith("data: ")][-1]
    assert done["pii_redacted"] == ["EG_NATIONAL_ID", "EG_PHONE"]


def test_stream_falls_back_when_backend_cannot_stream():
    client = TestClient(create_app(engine=FakeEngine()))  # plain generate_fn, no .stream
    with client.stream("POST", "/ask/stream", json={"question": "Q?"}) as resp:
        events = read_sse(resp)
    assert [e["type"] for e in events] == ["token", "done"]
    assert events[0]["text"].startswith("Fake answer")


def test_stream_reports_errors_without_internals():
    class Broken(FakeEngine):
        def generate_fn(self, prompt):
            raise RuntimeError("CUDA out of memory at 0xdeadbeef")

    client = TestClient(create_app(engine=Broken()))
    with client.stream("POST", "/ask/stream", json={"question": "Q?"}) as resp:
        body = resp.read().decode()
    assert '"type": "error"' in body and "deadbeef" not in body


def test_stream_rejects_empty_question():
    assert (
        TestClient(create_app(engine=FakeEngine()))
        .post("/ask/stream", json={"question": "  "})
        .status_code
        == 422
    )


def test_time_to_first_chunk_recorded():
    before = (
        REGISTRY.get_sample_value("rag_time_to_first_chunk_seconds_count", {"service": "fastapi"})
        or 0
    )
    client = TestClient(create_app(engine=StreamingEngine()))
    with client.stream("POST", "/ask/stream", json={"question": "Q?"}) as resp:
        resp.read()
    after = REGISTRY.get_sample_value(
        "rag_time_to_first_chunk_seconds_count", {"service": "fastapi"}
    )
    assert after == before + 1


@pytest.fixture
def traced(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", f"pk-lf-test-{uuid.uuid4()}")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    exporter = InMemorySpanExporter()
    lf = Langfuse(tracer_provider=TracerProvider(), span_exporter=exporter, tracing_enabled=True)
    yield lf, exporter
    lf.shutdown()


def test_streamed_request_is_one_complete_trace(traced):
    # The pipeline runs in a worker thread; all spans must still land in
    # one trace with the usual structure.
    lf, exporter = traced
    client = TestClient(create_app(engine=StreamingEngine(), langfuse=lf))
    with client.stream("POST", "/ask/stream", json={"question": "Q?"}) as resp:
        resp.read()
    lf.flush()
    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert {"ask", "pii-input", "retrieve", "generate", "pii-output"} <= names
    assert len({s.context.trace_id for s in spans}) == 1
    gen = next(s for s in spans if s.name == "generate")
    assert "fake-stream" in json.dumps(dict(gen.attributes), default=str)
