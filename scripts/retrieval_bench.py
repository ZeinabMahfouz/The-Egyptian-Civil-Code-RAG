"""Retrieval throughput under concurrency: local (embedded) Qdrant vs a
Qdrant server, on a synthetic index with the real index's shape (2,197
points, 1024-dim, the same payload fields and point IDs).

Measures RAGQueryEngine.retrieve() only -- exact-article filter, semantic
search, dedup, same-language twin swap. The embedder is a fake (random
vectors) so the numbers isolate the vector store; embedding cost is
separate.

    python scripts/retrieval_bench.py                                 # local
    python scripts/retrieval_bench.py --url http://localhost:6333     # server
"""

import argparse
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from egyptian_civil_code_rag.ids import CIVIL_CODE_DOC_ID, point_id
from egyptian_civil_code_rag.qdrant_seed import PAYLOAD_INDEXES
from egyptian_civil_code_rag.query import RAGQueryEngine

DIM = 1024
N_POINTS = 2197  # the real index: 1,099 chunks x 2 languages, one chunk English-only
COLLECTION = "retrieval_bench"
QUESTIONS = [
    "What does Article 147 say?",
    "ما حكم المادة ١٤٧؟",
    "Can a contract be cancelled for unforeseen circumstances?",
    "ما هي شروط صحة العقد؟",
]


class RandomEmbedder:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def encode(self, texts, **kwargs):
        return self.rng.standard_normal((len(texts), DIM)).astype("float32")


def build_index(client: QdrantClient, server: bool) -> int:
    rng = np.random.default_rng(0)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(COLLECTION, VectorParams(size=DIM, distance=Distance.COSINE))
    if server:  # as qdrant_seed creates them
        for field, schema in PAYLOAD_INDEXES.items():
            client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)
    points = [
        PointStruct(
            id=point_id(str(n), lang),
            vector=rng.standard_normal(DIM).tolist(),
            payload={
                "doc_id": CIVIL_CODE_DOC_ID,
                "chunk_id": str(n),
                "article_numbers": [n],
                "lang": lang,
                "citation": f"Egyptian Civil Code, Article {n}",
                "is_repealed": False,
                "text": "x" * 1200,
            },
        )
        for n in range(1, 1100)
        for lang in ("ar", "en")
    ][:N_POINTS]
    client.upload_points(COLLECTION, points=points, batch_size=256, wait=True)
    return client.count(COLLECTION).count


def measure(engine, threads: int, calls: int) -> dict:
    def one(i):
        t = time.perf_counter()
        engine.retrieve(QUESTIONS[i % len(QUESTIONS)])
        return time.perf_counter() - t

    start = time.perf_counter()
    with ThreadPoolExecutor(threads) as ex:
        latencies = sorted(ex.map(one, range(calls)))
    wall = time.perf_counter() - start
    return {
        "threads": threads,
        "calls": calls,
        "mean_ms": round(statistics.mean(latencies) * 1000, 1),
        "p95_ms": round(latencies[max(0, int(0.95 * calls) - 1)] * 1000, 1),
        "per_s": round(calls / wall, 1),
    }


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", help="Qdrant server; omit for the local store")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 8, 50])
    ap.add_argument("--calls-per-thread", type=int, default=6)
    args = ap.parse_args(argv)

    tmp = None
    if args.url:
        client = QdrantClient(url=args.url, timeout=60)
    else:
        tmp = tempfile.TemporaryDirectory()
        client = QdrantClient(path=tmp.name)
    print(
        f"[bench] {build_index(client, bool(args.url))} points, {'server' if args.url else 'local'}"
    )

    engine = RAGQueryEngine.__new__(RAGQueryEngine)  # skip model loading
    engine.embed_model, engine.client, engine.collection = RandomEmbedder(), client, COLLECTION
    measure(engine, 1, 10)  # warm-up

    rows = []
    print("| threads | mean ms | p95 ms | retrievals/s |\n|---|---|---|---|")
    for t in args.threads:
        r = measure(engine, t, max(100, t * args.calls_per_thread))
        rows.append(r)
        print(f"| {r['threads']} | {r['mean_ms']} | {r['p95_ms']} | {r['per_s']} |", flush=True)
    if args.url:
        client.delete_collection(COLLECTION)
    client.close()
    if tmp:
        tmp.cleanup()
    return rows


if __name__ == "__main__":
    main()
