import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, field_validator

from egyptian_civil_code_rag.pii import PIIGuard
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


def create_app(engine: RAGQueryEngine | None = None, pii_guard: PIIGuard | None = None) -> FastAPI:
    pii_guard = pii_guard or PIIGuard()
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
            app.state.engine.close()

        app = FastAPI(title="Egyptian Civil Code RAG", lifespan=lifespan)

    @app.middleware("http")
    async def add_release_header(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-App-Release"] = RELEASE
        return response

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest, engine: RAGQueryEngine = Depends(get_engine)):
        question, q_found = pii_guard(payload.question)
        result = engine.ask(question)
        answer, a_found = pii_guard(result["answer"])
        return {
            "answer": answer,
            "sources": result["sources"],
            "pii_redacted": sorted(set(q_found) | set(a_found)),
        }

    @app.get("/health", response_model=HealthResponse)
    def health(engine: RAGQueryEngine = Depends(get_engine)):
        count = engine.client.count(collection_name=engine.collection).count
        return {"status": "healthy", "documents_indexed": count, "release": RELEASE}

    return app


app = create_app()
