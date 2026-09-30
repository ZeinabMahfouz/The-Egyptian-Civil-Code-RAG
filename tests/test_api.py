from fastapi.testclient import TestClient

from egyptian_civil_code_rag.api import create_app


class FakeQdrantClient:
    def count(self, collection_name):
        class Result:
            count = 42

        return Result()


class FakeHit:
    def __init__(self, citation="Egyptian Civil Code, Article 1"):
        self.payload = {"citation": citation, "doc_id": "egyptian_civil_code", "lang": "en"}
        self.score = 0.9


class FakeEngine:
    """Stands in for RAGQueryEngine: same interface (.retrieve,
    .build_prompt, .generate_fn, .client, .collection), no model loading,
    no Qdrant connection."""

    collection = "fake_collection"
    client = FakeQdrantClient()

    def retrieve(self, question: str):
        return [FakeHit()]

    def build_prompt(self, question: str, hits) -> str:
        return f"PROMPT[{question}]"

    def generate_fn(self, prompt: str) -> str:
        return f"Fake answer to: {prompt}"


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
    """Records the question retrieval received (and echoes it into the
    answer), so tests can check both sides of the redaction."""

    def __init__(self):
        self.received = []

    def retrieve(self, question: str):
        self.received.append(question)
        return super().retrieve(question)


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
        def generate_fn(self, prompt):
            return "Contact 01012345678."

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
