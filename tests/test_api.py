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
    assert data["release"] == "dev"  # APP_RELEASE unset in tests


def test_every_response_names_its_release():
    # nginx logs this header per request -- it's how canary traffic is
    # told apart from stable in deploy/canary/canary_report.py
    client = make_client()
    assert client.get("/health").headers["X-App-Release"] == "dev"
    assert client.post("/ask", json={"question": "Q?"}).headers["X-App-Release"] == "dev"


# --- PII guardrails ---------------------------------------------------------


class RecordingEngine(FakeEngine):
    """Echoes the question into the answer and records what it received,
    so tests can check both sides of the redaction."""

    def __init__(self):
        self.received = []

    def ask(self, question: str) -> dict:
        self.received.append(question)
        return super().ask(question)


def test_pii_in_question_never_reaches_engine():
    engine = RecordingEngine()
    client = TestClient(create_app(engine=engine))
    resp = client.post(
        "/ask", json={"question": "رقمي القومي 29801011234567، هل يحق لي فسخ العقد؟"}
    )
    assert resp.status_code == 200
    assert "29801011234567" not in engine.received[0]
    assert "[EG_NATIONAL_ID]" in engine.received[0]
    assert "29801011234567" not in resp.json()["answer"]
    assert resp.json()["pii_redacted"] == ["EG_NATIONAL_ID"]


def test_pii_in_answer_is_redacted():
    class LeakyEngine(FakeEngine):
        def ask(self, question):
            return {"answer": "Contact 01012345678.", "sources": ["Egyptian Civil Code, Article 1"]}

    resp = TestClient(create_app(engine=LeakyEngine())).post("/ask", json={"question": "Hi?"})
    assert resp.json()["answer"] == "Contact [EG_PHONE]."
    assert resp.json()["pii_redacted"] == ["EG_PHONE"]


def test_legal_question_without_pii_is_untouched():
    engine = RecordingEngine()
    client = TestClient(create_app(engine=engine))
    q = "What does Article 147 say? Law 181 of 2018, Articles 54-80."
    resp = client.post("/ask", json={"question": q})
    assert engine.received[0] == q
    assert resp.json()["pii_redacted"] == []
