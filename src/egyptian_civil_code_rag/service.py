"""BentoML service wrapping the RAG pipeline with an async /ask endpoint.

Run from the repo root (params.yaml and the Qdrant index are relative paths):

    bentoml serve egyptian_civil_code_rag.service:CivilCodeRAG

Then open http://localhost:3000 for the interactive API page.
Same request contract as the FastAPI app: POST /ask {"question": "..."}.
Every request is traced to Langfuse when LANGFUSE_* env vars are set.
"""

import asyncio
import os

import bentoml
from bentoml.exceptions import InvalidArgument
from pydantic import BaseModel

from egyptian_civil_code_rag.pii import PIIGuard
from egyptian_civil_code_rag.pipeline import TracedPipeline, make_langfuse
from egyptian_civil_code_rag.query import RAGQueryEngine

MODEL_NAME = os.environ.get("GEN_MODEL", "Qwen/Qwen3-1.7B")
RELEASE = os.environ.get("APP_RELEASE", "dev")


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    pii_redacted: list[str] = []
    release: str
    trace_id: str | None = None


@bentoml.service(
    # Local (embedded) Qdrant allows one process per storage folder, so one
    # worker -- more would fail on the index lock. Concurrency comes from
    # async handling, not extra processes.
    workers=1,
    # CPU generation can take minutes; BentoML's default timeout is 60s.
    traffic={"timeout": 600, "max_concurrency": 8},
)
class CivilCodeRAG:
    def __init__(self) -> None:
        from egyptian_civil_code_rag.backends import transformers_backend

        print(f"[startup] loading {MODEL_NAME} (CPU dev backend)")
        self.engine = RAGQueryEngine(generate_fn=transformers_backend(MODEL_NAME))
        self.langfuse = make_langfuse()
        self.pipeline = TracedPipeline(self.engine, PIIGuard(), self.langfuse, service="bentoml")
        # One generation at a time: on CPU, parallel generations just fight
        # over the same cores and all finish later.
        self._generate_lock = asyncio.Lock()
        print("[startup] ready")

    @bentoml.api
    async def ask(self, question: str) -> AskResponse:
        if not question or not question.strip():
            raise InvalidArgument("question must not be empty")

        async with self._generate_lock:
            # pipeline.ask is blocking (embedding + generation); run it in a
            # thread so the event loop keeps serving /health and queuing.
            # PII redaction and the Langfuse trace happen inside it.
            result = await asyncio.to_thread(self.pipeline.ask, question)

        return AskResponse(
            answer=result.answer,
            sources=result.sources,
            pii_redacted=result.pii_redacted,
            release=RELEASE,
            trace_id=result.trace_id,
        )

    @bentoml.api
    async def health(self) -> dict:
        count = self.engine.client.count(collection_name=self.engine.collection).count
        return {"status": "healthy", "documents_indexed": count, "release": RELEASE}

    @bentoml.on_shutdown
    def shutdown(self) -> None:
        self.langfuse.flush()  # don't drop the last traces
        self.engine.close()
