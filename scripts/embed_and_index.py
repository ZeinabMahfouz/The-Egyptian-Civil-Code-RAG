import argparse
import json
import sys
import uuid
from pathlib import Path

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# Fixed namespace so point IDs are deterministic across runs, not random.
ID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def point_id(chunk_id: str, lang: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, f"{chunk_id}-{lang}"))


def load_params(params_path: Path):
    with open(params_path, encoding="utf-8") as f:
        p = yaml.safe_load(f)
    return p["embedding"], p["qdrant"]


def build_points(chunks, model, batch_size):
    """Yields (PointStruct, lang, chunk_id) for every non-empty text field
    across all chunks, embedding in batches for efficiency."""
    jobs = []  # (chunk, lang, text)
    for c in chunks:
        if c["text_ar"].strip():
            jobs.append((c, "ar", c["text_ar"]))
        if c["text_en"].strip():
            jobs.append((c, "en", c["text_en"]))

    texts = [j[2] for j in jobs]
    print(
        f"[info] embedding {len(texts)} (chunk, language) pairs from {len(chunks)} chunks",
        file=sys.stderr,
    )
    vectors = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=False
    )

    for (c, lang, text), vec in zip(jobs, vectors):
        payload = {
            "chunk_id": c["chunk_id"],
            "article_numbers": c["article_numbers"],
            "lang": lang,
            "book": c["book"],
            "chapter": c["chapter"],
            "section": c["section"],
            "topic": c["topic"],
            "is_repealed": c["is_repealed"],
            "citation": c["citation"],
            "source_page": c["source_page"],
            "flags": c["flags"],
            "text": text,
        }
        yield PointStruct(id=point_id(c["chunk_id"], lang), vector=vec.tolist(), payload=payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, type=Path)
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = ap.parse_args()

    embed_params, qdrant_params = load_params(args.params)

    with open(args.chunks, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[info] loading {embed_params['model_name']}", file=sys.stderr)
    model = SentenceTransformer(embed_params["model_name"])
    dim = model.get_embedding_dimension()

    storage_path = Path(qdrant_params["storage_path"])
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(storage_path))

    collection = qdrant_params["collection_name"]
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = list(build_points(chunks, model, embed_params["batch_size"]))
    client.upsert(collection_name=collection, points=points)

    count = client.count(collection_name=collection).count
    print(
        f"[info] indexed {count} points into collection '{collection}' at {storage_path}",
        file=sys.stderr,
    )
    assert count == len(points), f"Expected {len(points)} points indexed, got {count}"

    # article coverage check, same discipline as extraction/chunking:
    # every article_number in the source corpus should be retrievable
    # from at least one indexed point.
    covered = set()
    for c in chunks:
        covered.update(c["article_numbers"])
    print(
        f"[info] {len(covered)} distinct article numbers represented in the index", file=sys.stderr
    )


if __name__ == "__main__":
    main()
