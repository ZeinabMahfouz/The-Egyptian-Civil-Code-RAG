"""Copy the local Qdrant store (the DVC output baked into the image) into a
Qdrant server, with payload indexes on the fields the query engine filters.

    python -m egyptian_civil_code_rag.qdrant_seed --url http://qdrant:6333

Idempotent: if the server collection already holds the same number of points
it is left alone, so it can run before every API start
(docker-compose.qdrant.yml). --recreate forces a full copy, e.g. after
`dvc repro` rebuilt the index with the same point count.
"""

import argparse
import sys
import time
from pathlib import Path

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType, PointStruct

# Fields the query engine filters on (query.py). Without these indexes a
# filter makes the server check every point's payload.
PAYLOAD_INDEXES = {
    "doc_id": PayloadSchemaType.KEYWORD,
    "lang": PayloadSchemaType.KEYWORD,
    "chunk_id": PayloadSchemaType.KEYWORD,
    "article_numbers": PayloadSchemaType.INTEGER,
}


def wait_for(client: QdrantClient, timeout_s: float = 60) -> None:
    """The server container can take a few seconds to accept requests."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            client.get_collections()
            return
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(1)


def seed(
    source: QdrantClient,
    target: QdrantClient,
    collection: str,
    batch_size: int = 256,
    recreate: bool = False,
) -> dict:
    n_source = source.count(collection).count
    if not recreate and target.collection_exists(collection):
        n_target = target.count(collection).count
        if n_target == n_source:
            return {"copied": 0, "points": n_target, "skipped": True}

    vectors = source.get_collection(collection).config.params.vectors
    if target.collection_exists(collection):
        target.delete_collection(collection)
    target.create_collection(collection, vectors_config=vectors)
    for field, schema in PAYLOAD_INDEXES.items():
        target.create_payload_index(collection, field_name=field, field_schema=schema)

    copied, offset = 0, None
    while True:
        records, offset = source.scroll(
            collection, limit=batch_size, offset=offset, with_payload=True, with_vectors=True
        )
        # Batched: one request with the whole index (~47 MB of JSON) is over
        # the server's 32 MB request limit.
        target.upsert(
            collection,
            points=[PointStruct(id=r.id, vector=r.vector, payload=r.payload) for r in records],
            wait=True,
        )
        copied += len(records)
        if offset is None:
            break

    n_target = target.count(collection, exact=True).count
    if n_target != n_source:
        raise RuntimeError(f"copied {copied} points; server holds {n_target}, source {n_source}")
    return {"copied": copied, "points": n_target, "skipped": False}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True, help="Qdrant server, e.g. http://qdrant:6333")
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--recreate", action="store_true")
    args = ap.parse_args(argv)

    params = yaml.safe_load(args.params.read_text(encoding="utf-8"))["qdrant"]
    source = QdrantClient(path=params["storage_path"])
    target = QdrantClient(url=args.url, timeout=60)
    try:
        wait_for(target)
        result = seed(source, target, params["collection_name"], args.batch_size, args.recreate)
    finally:
        source.close()
    what = "already up to date" if result["skipped"] else f"copied {result['copied']} points"
    print(f"[qdrant_seed] {params['collection_name']} on {args.url}: {what}", file=sys.stderr)
    return result


if __name__ == "__main__":
    main()
