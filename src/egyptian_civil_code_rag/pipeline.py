"""The traced /ask pipeline, shared by the FastAPI app and the BentoML service.

One Langfuse trace per request, with a span for every stage:

    ask                         (span)       redacted question -> answer + sources
    |- pii-input                (guardrail)  entity types found in the question
    |- retrieve                 (retriever)  cited articles + similarity scores
    |- generate                 (generation) prompt -> answer, model, token usage
    `- pii-output               (guardrail)  entity types found in the answer

and the matching Prometheus metrics (see metrics.py) for every request.

Privacy rule: the *raw* question never enters a trace. PII is redacted
first, and every span only ever sees the redacted text -- otherwise
Langfuse would become the place where national IDs get stored.

Tracing is off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set,
so tests, CI and anyone running without a Langfuse server are unaffected.
Metrics are always on (they're just in-process counters until scraped).
"""

import os
import time
from dataclasses import dataclass

from langfuse import Langfuse, propagate_attributes

from egyptian_civil_code_rag import metrics
from egyptian_civil_code_rag.pii import PIIGuard

NO_CONTEXT_ANSWER = "No relevant articles found."


def make_langfuse(**overrides) -> Langfuse:
    """Langfuse client configured from the environment:
    LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL."""
    enabled = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))
    kwargs = {
        "tracing_enabled": enabled,
        "release": os.environ.get("APP_RELEASE", "dev"),
    }
    kwargs.update(overrides)
    return Langfuse(**kwargs)


@dataclass
class AskResult:
    answer: str
    sources: list[str]
    pii_redacted: list[str]
    trace_id: str | None
    usage: dict | None  # {"input": n, "output": m} tokens, when the backend reports it


class TracedPipeline:
    def __init__(self, engine, pii_guard: PIIGuard, langfuse: Langfuse, service: str):
        self.engine = engine
        self.pii = pii_guard
        self.langfuse = langfuse
        self.service = service  # "fastapi" or "bentoml" -- which server answered

    def ask(self, question: str) -> AskResult:
        started = time.perf_counter()
        try:
            result = self._ask(question)
        except Exception:
            metrics.REQUESTS.labels(self.service, "error").inc()
            raise
        finally:
            metrics.LATENCY.labels(self.service, "total").observe(time.perf_counter() - started)

        status = "ok" if result.sources else "no_context"
        metrics.REQUESTS.labels(self.service, status).inc()
        if result.usage:
            for direction in ("input", "output"):
                metrics.TOKENS.labels(self.service, direction).inc(result.usage.get(direction, 0))
        return result

    def _ask(self, question: str) -> AskResult:
        clean_question, q_found = self.pii(question)
        for entity in q_found:
            metrics.PII_REDACTIONS.labels(self.service, "question", entity).inc()

        with propagate_attributes(
            trace_name="ask",
            tags=[f"service:{self.service}"],
            metadata={"service": self.service},
        ):
            with self.langfuse.start_as_current_observation(
                name="ask", as_type="span", input={"question": clean_question}
            ) as root:
                with self.langfuse.start_as_current_observation(
                    name="pii-input", as_type="guardrail", input=clean_question
                ) as g:
                    g.update(output={"entities": q_found})

                with self.langfuse.start_as_current_observation(
                    name="retrieve", as_type="retriever", input=clean_question
                ) as r:
                    t0 = time.perf_counter()
                    hits = self.engine.retrieve(clean_question)
                    metrics.LATENCY.labels(self.service, "retrieve").observe(
                        time.perf_counter() - t0
                    )
                    if hits:
                        metrics.RETRIEVAL_TOP_SCORE.labels(self.service).observe(
                            max(float(h.score) for h in hits)
                        )
                    r.update(
                        output=[
                            {
                                "citation": h.payload["citation"],
                                "doc_id": h.payload.get("doc_id"),
                                "lang": h.payload.get("lang"),
                                "score": round(float(h.score), 4),
                            }
                            for h in hits
                        ]
                    )
                sources = [h.payload["citation"] for h in hits]

                usage = None
                if not hits:
                    raw_answer = NO_CONTEXT_ANSWER
                    answer, a_found = self.pii(raw_answer)
                else:
                    prompt = self.engine.build_prompt(clean_question, hits)
                    generate_fn = self.engine.generate_fn
                    with self.langfuse.start_as_current_observation(
                        name="generate",
                        as_type="generation",
                        input=prompt,
                        model=getattr(generate_fn, "model_name", None),
                    ) as gen:
                        t0 = time.perf_counter()
                        raw_answer = generate_fn(prompt).strip()
                        metrics.LATENCY.labels(self.service, "generate").observe(
                            time.perf_counter() - t0
                        )
                        usage = getattr(generate_fn, "last_usage", None)
                        # Redact before recording: if the model produced PII,
                        # the trace must be as clean as the response.
                        answer, a_found = self.pii(raw_answer)
                        gen.update(output=answer, usage_details=usage)

                with self.langfuse.start_as_current_observation(
                    name="pii-output", as_type="guardrail", input=answer
                ) as g:
                    g.update(output={"entities": a_found})

                for entity in a_found:
                    metrics.PII_REDACTIONS.labels(self.service, "answer", entity).inc()

                pii_redacted = sorted(set(q_found) | set(a_found))
                root.update(
                    output={"answer": answer, "sources": sources},
                    metadata={"pii_redacted": pii_redacted},
                )
                trace_id = root.trace_id

        return AskResult(answer, sources, pii_redacted, trace_id, usage)
