"""Qdrant server mode: seeding the server from the local store, the client
switch (QDRANT_URL), and the ID-based same-language twin lookup.

Runs against in-memory Qdrant. With QDRANT_TEST_URL set (a real server,
e.g. `docker run -p 6333:6333 qdrant/qdrant`), the last test also runs the
copy and a retrieval against it."""

import os
import sys
import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from embed_and_index import build_points  # noqa: E402
from test_reindex_batch import DIM, FakeEmbedder, civil_chunk  # noqa: E402

from egyptian_civil_code_rag import qdrant_seed  # noqa: E402
from egyptian_civil_code_rag.ids import point_id  # noqa: E402
from egyptian_civil_code_rag.query import RAGQueryEngine, make_qdrant_client  # noqa: E402

COLLECTION = "civil"


def index(client, n_articles=40):
    client.create_collection(
        COLLECTION, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
    )
    chunks = [
        civil_chunk(n, f"نص المادة {n} " * 10, f"Text of Article {n}. " * 10)
        for n in range(1, n_articles + 1)
    ]
    client.upsert(COLLECTION, points=list(build_points(chunks, FakeEmbedder(), 16)))
    return client


def engine_on(client):
    e = RAGQueryEngine.__new__(RAGQueryEngine)  # skip model loading
    e.embed_model, e.client, e.collection, e.generate_fn = FakeEmbedder(), client, COLLECTION, None
    return e


def test_seed_copies_points_vectors_and_payloads():
    source, target = index(QdrantClient(":memory:")), QdrantClient(":memory:")
    result = qdrant_seed.seed(source, target, COLLECTION, batch_size=7)  # several batches
    assert result == {"copied": 80, "points": 80, "skipped": False}

    pid = point_id("12", "en")
    [a] = source.retrieve(COLLECTION, ids=[pid], with_vectors=True)
    [b] = target.retrieve(COLLECTION, ids=[pid], with_vectors=True)
    assert a.payload == b.payload and a.vector == pytest.approx(b.vector)


def test_seed_is_idempotent_and_recreate_forces_a_copy():
    source, target = index(QdrantClient(":memory:")), QdrantClient(":memory:")
    qdrant_seed.seed(source, target, COLLECTION)
    assert qdrant_seed.seed(source, target, COLLECTION)["skipped"]
    assert not qdrant_seed.seed(source, target, COLLECTION, recreate=True)["skipped"]
    assert target.count(COLLECTION).count == 80


def test_seed_replaces_a_stale_server_collection():
    source, target = index(QdrantClient(":memory:")), index(QdrantClient(":memory:"), 5)
    assert target.count(COLLECTION).count == 10
    assert qdrant_seed.seed(source, target, COLLECTION)["copied"] == 80


def test_qdrant_url_switches_to_a_server_client(monkeypatch, tmp_path):
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    c = make_qdrant_client(str(tmp_path / "unused"))
    assert c.init_options.get("url") == "http://qdrant:6333"
    assert not (tmp_path / "unused").exists()
    monkeypatch.delenv("QDRANT_URL")
    c = make_qdrant_client(str(tmp_path / "local"))
    assert (tmp_path / "local").exists()
    c.close()


class NoScroll:
    """Wraps a client; fails if the engine falls back to a filtered scroll."""

    def __init__(self, client):
        self._c = client

    def scroll(self, *a, **kw):
        raise AssertionError("twin lookup must be by ID, not a filtered scroll")

    def __getattr__(self, name):
        return getattr(self._c, name)


def test_twins_fetched_by_id_in_the_question_language():
    e = engine_on(NoScroll(index(QdrantClient(":memory:"))))
    for question, lang in (("What does Article 7 say?", "en"), ("ماذا تقول المادة 7؟", "ar")):
        hits = e.retrieve(question)
        assert hits and all(h.payload["lang"] == lang for h in hits)
        assert hits[0].payload["article_numbers"] == [7]


def test_hit_without_a_twin_is_kept_as_is():
    client = index(QdrantClient(":memory:"))
    client.delete(COLLECTION, points_selector=[point_id(str(n), "en") for n in range(1, 41)])
    hits = engine_on(client).retrieve("What does Article 7 say?")
    assert hits and all(h.payload["lang"] == "ar" for h in hits)


@pytest.mark.skipif(not os.environ.get("QDRANT_TEST_URL"), reason="needs a Qdrant server")
def test_against_a_real_server():
    name = f"test_{uuid.uuid4().hex[:8]}"
    source = QdrantClient(":memory:")
    global COLLECTION
    saved, COLLECTION = COLLECTION, name
    try:
        index(source)
        target = QdrantClient(url=os.environ["QDRANT_TEST_URL"])
        assert qdrant_seed.seed(source, target, name)["points"] == 80
        info = target.get_collection(name)
        assert set(qdrant_seed.PAYLOAD_INDEXES) <= set(info.payload_schema)
        hits = engine_on(target).retrieve("What does Article 7 say?")
        assert hits[0].payload["article_numbers"] == [7] and hits[0].payload["lang"] == "en"
        target.delete_collection(name)
    finally:
        COLLECTION = saved


def test_compose_override_seeds_then_serves_from_the_server():
    import yaml

    o = yaml.safe_load((REPO / "docker-compose.qdrant.yml").read_text(encoding="utf-8"))
    api, url = o["services"]["api"], o["services"]["api"]["environment"]["QDRANT_URL"]
    assert url == "http://qdrant:6333" and "qdrant" in o["services"]
    cmd = api["command"][-1]
    assert cmd.index(f"qdrant_seed --url {url}") < cmd.index("&& exec uvicorn")


def test_package_point_id_matches_the_indexer():
    # scripts/embed_and_index.py keeps its own copy: editing it would make
    # `dvc repro` re-embed the corpus. The query engine relies on both
    # computing the same IDs.
    import embed_and_index

    for args in (("147", "en"), ("389-417-repealed", "ar"), ("1", "en", "other_law")):
        assert point_id(*args) == embed_and_index.point_id(*args)
