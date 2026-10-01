"""Batch re-indexing, tested against an in-memory Qdrant and a fake
embedder -- no model download, no DVC data, runs in CI's `test` job.

The base index mimics the real one's shape (Civil Code points with
doc_id, including an Article 1 that a new document's Article 1 must not
collide with); the "new document" is tests/fixtures/synthetic_law.json.
"""

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from documents import DocumentValidationError, validate_document  # noqa: E402
from embed_and_index import build_points, point_id  # noqa: E402
from reindex_batch import count_doc, reindex_documents  # noqa: E402

from egyptian_civil_code_rag.query import CIVIL_CODE_DOC_ID, RAGQueryEngine  # noqa: E402

COLLECTION = "test_collection"
DIM = 16
FIXTURE = REPO / "tests" / "fixtures" / "synthetic_law.json"


class FakeEmbedder:
    """Deterministic text -> vector, same call shape as SentenceTransformer."""

    def __init__(self, dim=DIM):
        self.dim = dim

    def get_embedding_dimension(self):
        return self.dim

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=False):
        out = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:4], "little")
            out.append(np.random.default_rng(seed).standard_normal(self.dim))
        return np.array(out, dtype=np.float32)


def civil_chunk(n, text_ar, text_en):
    return {
        "chunk_id": str(n),
        "article_numbers": [n],
        "paragraph_index": None,
        "book": "",
        "chapter": "",
        "section": "",
        "topic": "",
        "text_ar": text_ar,
        "text_en": text_en,
        "is_repealed": False,
        "source_page": 1,
        "citation": f"Egyptian Civil Code, Article {n}",
        "flags": [],
    }


BASE_CHUNKS = [
    civil_chunk(1, "نص المادة الأولى من القانون المدني", "Civil Code Article 1 text"),
    civil_chunk(147, "العقد شريعة المتعاقدين", "The contract makes the law of the parties"),
]


@pytest.fixture
def model():
    return FakeEmbedder()


@pytest.fixture
def client(model):
    c = QdrantClient(":memory:")
    c.create_collection(COLLECTION, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
    c.upsert(COLLECTION, points=list(build_points(BASE_CHUNKS, model, batch_size=8)))
    yield c
    c.close()


@pytest.fixture
def doc():
    with open(FIXTURE, encoding="utf-8") as f:
        d = json.load(f)
    validate_document(d)
    return d


def run(client, docs, model):
    return reindex_documents(client, COLLECTION, docs, model, 8, 700, True)


def civil_snapshot(client):
    pts, _ = client.scroll(COLLECTION, limit=100, with_payload=True, with_vectors=True)
    return {
        str(p.id): (p.payload, p.vector) for p in pts if p.payload["doc_id"] == CIVIL_CODE_DOC_ID
    }


# --- the checklist item itself: one new document, indexed incrementally ---


def test_new_document_is_added_without_touching_civil_code(client, doc, model):
    civil_before = civil_snapshot(client)
    results, total_before, total_after = run(client, [doc], model)

    assert results[0].points_before == 0
    assert results[0].points_after == 5  # 3 articles; Article 3 has no English text
    assert total_after == total_before + 5
    assert civil_snapshot(client) == civil_before  # same IDs, payloads and vectors


def test_new_document_citations_use_its_own_prefix(client, doc, model):
    run(client, [doc], model)
    pts, _ = client.scroll(COLLECTION, limit=100, with_payload=True)
    citations = {p.payload["citation"] for p in pts if p.payload["doc_id"] == doc["doc_id"]}
    assert citations == {f"Synthetic Test Law, Article {n}" for n in (1, 2, 3)}


def test_article_number_collision_does_not_overwrite_civil_code(doc):
    # Both documents have an Article 1; their point IDs must differ.
    assert point_id("1", "ar") != point_id("1", "ar", doc["doc_id"])


def test_civil_code_point_ids_unchanged_by_doc_namespacing():
    # Existing index IDs must not move -- uuid5(ns, "<chunk_id>-<lang>") as before.
    import uuid

    from embed_and_index import ID_NAMESPACE

    assert point_id("147", "en") == str(uuid.uuid5(ID_NAMESPACE, "147-en"))


# --- idempotency and updates ---


def test_rerun_is_idempotent(client, doc, model):
    _, _, total_first = run(client, [doc], model)
    results, _, total_second = run(client, [doc], model)
    assert total_second == total_first
    assert results[0].points_before == results[0].points_after == 5


def test_edited_document_removes_stale_chunks(client, doc, model):
    run(client, [doc], model)
    shrunk = copy.deepcopy(doc)
    shrunk["articles"] = shrunk["articles"][:1]  # Articles 2 and 3 removed

    results, _, _ = run(client, [shrunk], model)
    assert results[0].points_after == 2
    pts, _ = client.scroll(COLLECTION, limit=100, with_payload=True)
    remaining = {
        n for p in pts if p.payload["doc_id"] == doc["doc_id"] for n in p.payload["article_numbers"]
    }
    assert remaining == {1}  # a removed article must not stay retrievable/citable


def test_edited_text_is_reembedded(client, doc, model):
    run(client, [doc], model)
    edited = copy.deepcopy(doc)
    edited["articles"][0]["text_en"] = "Amended test-only text."
    run(client, [edited], model)
    pid = point_id("1", "en", doc["doc_id"])
    p = client.retrieve(COLLECTION, ids=[pid], with_payload=True)[0]
    assert p.payload["text"] == "Amended test-only text."


# --- guards ---


def test_dimension_mismatch_is_refused(client, doc):
    with pytest.raises(RuntimeError, match="dim"):
        run(client, [doc], FakeEmbedder(dim=DIM * 2))


def test_index_without_doc_ids_is_refused(model, doc):
    c = QdrantClient(":memory:")
    c.create_collection(COLLECTION, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
    legacy = list(build_points(BASE_CHUNKS, model, batch_size=8))
    for p in legacy:
        p.payload.pop("doc_id")
    c.upsert(COLLECTION, points=legacy)
    with pytest.raises(RuntimeError, match="predates the doc_id"):
        run(c, [doc], model)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(doc_id=CIVIL_CODE_DOC_ID), "reserved"),
        (lambda d: d.update(doc_id="Bad ID!"), "slug"),
        (lambda d: d["articles"].append(copy.deepcopy(d["articles"][0])), "duplicate"),
        (lambda d: d["articles"][0].update(text_ar="   "), "empty text_ar"),
        (lambda d: d["articles"][0].update(article_number="1"), "positive int"),
        (lambda d: d["articles"][0].update(text_ar="x" * 6001), "failed article split"),
        (lambda d: d.update(articles=[]), "non-empty"),
        (lambda d: d["articles"][1].update(text_ar="<official text of Article 2>"), "placeholder"),
        (
            lambda d: d["articles"][0].update(text_en="<full official text of Article 1>"),
            "placeholder",
        ),
        (lambda d: d["articles"][0].update(text_ar="TODO"), "placeholder"),
        (lambda d: d["articles"][0].update(text_ar="نص قصير"), "stub"),
        (lambda d: d.update(source="<URL or Official Gazette issue>"), "source"),
    ],
)
def test_invalid_documents_are_rejected(doc, mutate, message):
    mutate(doc)
    with pytest.raises(DocumentValidationError, match=message):
        validate_document(doc)


# --- retrieval: exact lookup stays scoped to the Civil Code ---


def test_exact_article_lookup_ignores_other_documents(client, doc, model):
    run(client, [doc], model)
    engine = RAGQueryEngine.__new__(RAGQueryEngine)  # skip model/Qdrant loading in __init__
    engine.embed_model = model
    engine.client = client
    engine.collection = COLLECTION

    hits = engine.retrieve("What does Article 1 say?", top_k_distinct=1)
    assert hits[0].payload["doc_id"] == CIVIL_CODE_DOC_ID
    assert hits[0].payload["citation"] == "Egyptian Civil Code, Article 1"
    assert count_doc(client, COLLECTION, doc["doc_id"]) == 5


# --- retrieval: answer from the question's language ---


def test_retrieval_returns_text_in_the_question_language(client, doc, model):
    run(client, [doc], model)
    engine = RAGQueryEngine.__new__(RAGQueryEngine)
    engine.embed_model = model
    engine.client = client
    engine.collection = COLLECTION

    en = engine.retrieve("What does Article 1 say?", top_k_distinct=1)
    ar = engine.retrieve("ماذا تقول المادة 1؟", top_k_distinct=1)
    assert en[0].payload["lang"] == "en"
    assert ar[0].payload["lang"] == "ar"
    # same article either way -- only the language of the text changes
    assert en[0].payload["chunk_id"] == ar[0].payload["chunk_id"]
