from fastapi.testclient import TestClient

from egyptian_civil_code_rag.api import create_app


class FakeQdrantClient:
    def count(self, collection_name):
        class Result:
            count = 42

        return Result()


class FakeEngine:
    """Stands in for RAGQueryEngine: same interface (.ask, .client,
    .collection), no model loading, no Qdrant connection."""

    collection = "fake_collection"
    client = FakeQdrantClient()

    def ask(self, question: str) -> dict:
        return {
            "answer": f"Fake answer to: {question}",
            "sources": ["Egyptian Civil Code, Article 1"],
        }


def make_client():
    app = create_app(engine=FakeEngine())
    return TestClient(app)


def test_empty_question_returns_422():
    client = make_client()
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_whitespace_only_question_returns_422():
    client = make_client()
    resp = client.post("/ask", json={"question": "   "})
    assert resp.status_code == 422


def test_missing_question_field_returns_422():
    client = make_client()
    resp = client.post("/ask", json={})
    assert resp.status_code == 422


def test_valid_question_returns_200_with_answer_and_sources():
    client = make_client()
    resp = client.post("/ask", json={"question": "What does Article 147 say?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)
    for source in data["sources"]:
        assert "Article" in source


def test_health_endpoint():
    client = make_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["documents_indexed"] == 42
