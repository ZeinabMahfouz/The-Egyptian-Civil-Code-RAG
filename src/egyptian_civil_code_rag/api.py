from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, field_validator

from egyptian_civil_code_rag.query import RAGQueryEngine

DEV_MODEL_NAME = "Qwen/Qwen3-1.7B"


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


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int


def get_engine(request: Request) -> RAGQueryEngine:
    return request.app.state.engine


def create_app(engine: RAGQueryEngine | None = None) -> FastAPI:
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

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest, engine: RAGQueryEngine = Depends(get_engine)):
        return engine.ask(payload.question)

    @app.get("/health", response_model=HealthResponse)
    def health(engine: RAGQueryEngine = Depends(get_engine)):
        count = engine.client.count(collection_name=engine.collection).count
        return {"status": "healthy", "documents_indexed": count}

    return app


app = create_app()
