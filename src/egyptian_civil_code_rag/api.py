import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, field_validator

from egyptian_civil_code_rag.metrics import load_ragas_faithfulness
from egyptian_civil_code_rag.pii import PIIGuard
from egyptian_civil_code_rag.pipeline import TracedPipeline, make_langfuse
from egyptian_civil_code_rag.query import RAGQueryEngine

DEV_MODEL_NAME = "Qwen/Qwen3-1.7B"
# Which build is answering -- set per deployment (image tag / git SHA) so a
# canary and the stable release can be told apart behind the load balancer.
RELEASE = os.environ.get("APP_RELEASE", "dev")


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question must not be empty")
        return v


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    # PII entity types redacted from the question and/or answer, e.g.
    # ["EG_NATIONAL_ID"]. Empty when nothing was found.
    pii_redacted: list[str] = []


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    release: str


def get_engine(request: Request) -> RAGQueryEngine:
    return request.app.state.engine


def create_app(
    engine: RAGQueryEngine | None = None,
    pii_guard: PIIGuard | None = None,
    langfuse=None,
) -> FastAPI:
    pii_guard = pii_guard or PIIGuard()
    langfuse = langfuse or make_langfuse()
    load_ragas_faithfulness()  # sets the faithfulness gauge only if real scores exist
    if engine is not None:
        app = FastAPI(title="Egyptian Civil Code RAG")
        app.state.engine = engine
    else:

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            from egyptian_civil_code_rag.backends import transformers_backend

            print(f"[startup] loading {DEV_MODEL_NAME} (dev backend -- see docs/decisions.md)")
            app.state.engine = RAGQueryEngine(generate_fn=transformers_backend(DEV_MODEL_NAME))
            print("[startup] ready")
            yield
            langfuse.flush()  # don't drop the last traces on shutdown
            app.state.engine.close()

        app = FastAPI(title="Egyptian Civil Code RAG", lifespan=lifespan)

    @app.middleware("http")
    async def add_release_header(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-App-Release"] = RELEASE
        return response

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest, engine: RAGQueryEngine = Depends(get_engine)):
        # PII redaction, retrieval, generation and the Langfuse trace all
        # live in TracedPipeline, shared with the BentoML service.
        pipeline = TracedPipeline(engine, pii_guard, langfuse, service="fastapi")
        result = pipeline.ask(payload.question)
        return {
            "answer": result.answer,
            "sources": result.sources,
            "pii_redacted": result.pii_redacted,
        }

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        # Scraped by Prometheus (deploy/monitoring/). Defined as a route, not
        # a mounted sub-app, so /metrics doesn't 307-redirect to /metrics/.
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health", response_model=HealthResponse)
    def health(engine: RAGQueryEngine = Depends(get_engine)):
        count = engine.client.count(collection_name=engine.collection).count
        return {"status": "healthy", "documents_indexed": count, "release": RELEASE}

    return app


app = create_app()
