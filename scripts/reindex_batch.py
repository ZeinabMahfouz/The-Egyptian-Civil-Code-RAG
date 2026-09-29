"""Batch re-indexing: add or update legal documents in the existing index
without rebuilding it from scratch.

    python scripts/reindex_batch.py data/documents/law_4_1996.json [more.json ...]
    python scripts/reindex_batch.py --all          # every data/documents/*.json
    python scripts/reindex_batch.py --dry-run ...  # validate + chunk only, touch nothing

Per document: validate -> chunk (same strategy as the Civil Code) -> embed
-> upsert -> delete that document's stale points -> verify counts. Re-running
with the same file is a no-op in effect (deterministic IDs); re-running
after editing the file replaces that document's points exactly, including
removing chunks for articles that no longer exist.

The Civil Code itself is never touched here -- its points are counted
before and after and must be identical.

Relationship to DVC: this is the fast operational path. The source of
truth is still `dvc repro`, whose embed_index stage indexes the Civil
Code *plus* every file in data/documents/, so a document added here must
also be committed there, or the next full rebuild drops it.

Qdrant local (embedded) mode allows one process per storage folder: stop
the API (`docker compose down` / Ctrl-C uvicorn) before running this.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from documents import chunk_document, list_document_paths, load_document
from embed_and_index import build_points, load_chunking_params, load_params
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointIdsList

from egyptian_civil_code_rag.query import CIVIL_CODE_DOC_ID

DEFAULT_DOCS_DIR = Path("data/documents")


@dataclass
class ReindexResult:
    doc_id: str
    articles: int
    chunks: int
    points_before: int
    points_after: int


def doc_filter(doc_id: str) -> Filter:
    return Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])


def count_doc(client: QdrantClient, collection: str, doc_id: str) -> int:
    return client.count(collection_name=collection, count_filter=doc_filter(doc_id)).count


def existing_ids(client: QdrantClient, collection: str, doc_id: str) -> list[str]:
    ids, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            scroll_filter=doc_filter(doc_id),
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.extend(str(p.id) for p in batch)
        if offset is None:
            return ids


def check_collection(client: QdrantClient, collection: str, model) -> None:
    if not client.collection_exists(collection):
        raise RuntimeError(
            f"Collection '{collection}' does not exist -- build the base index first "
            "(`dvc pull` or `dvc repro`)."
        )
    index_dim = client.get_collection(collection).config.params.vectors.size
    model_dim = model.get_embedding_dimension()
    if index_dim != model_dim:
        raise RuntimeError(
            f"Embedding model produces {model_dim}-dim vectors but the index holds "
            f"{index_dim}-dim vectors -- params.yaml's embedding model changed since the "
            "index was built. Mixed embedding spaces make retrieval meaningless: run a "
            "full `dvc repro` instead."
        )
    if count_doc(client, collection, CIVIL_CODE_DOC_ID) == 0:
        raise RuntimeError(
            f"No points with doc_id={CIVIL_CODE_DOC_ID!r} -- the index predates the doc_id "
            "field. Rebuild it with `dvc repro` first."
        )


def reindex_document(
    client: QdrantClient,
    collection: str,
    doc: dict,
    model,
    batch_size: int,
    threshold: int,
    dedupe_repealed: bool,
) -> ReindexResult:
    doc_id = doc["doc_id"]
    chunks = chunk_document(doc, threshold, dedupe_repealed)
    points = list(build_points(chunks, model, batch_size, doc_id))

    before = count_doc(client, collection, doc_id)
    # Upsert, then delete this document's stale points. Upsert alone isn't
    # enough: if an edited document now has fewer chunks (an article
    # removed, a paragraph split undone), the old chunks would stay
    # retrievable -- and citable. Upserting *first* means a crash midway
    # leaves old+new points, never a document that has vanished.
    client.upsert(collection_name=collection, points=points)
    new_ids = {p.id for p in points}
    stale = [pid for pid in existing_ids(client, collection, doc_id) if pid not in new_ids]
    if stale:
        client.delete(collection_name=collection, points_selector=PointIdsList(points=stale))
    after = count_doc(client, collection, doc_id)

    if after != len(points):
        raise RuntimeError(f"[{doc_id}] expected {len(points)} points after upsert, found {after}")

    return ReindexResult(doc_id, len(doc["articles"]), len(chunks), before, after)


def reindex_documents(client, collection, docs, model, batch_size, threshold, dedupe_repealed):
    check_collection(client, collection, model)
    civil_before = count_doc(client, collection, CIVIL_CODE_DOC_ID)
    total_before = client.count(collection_name=collection).count

    results = [
        reindex_document(client, collection, d, model, batch_size, threshold, dedupe_repealed)
        for d in docs
    ]

    civil_after = count_doc(client, collection, CIVIL_CODE_DOC_ID)
    total_after = client.count(collection_name=collection).count
    if civil_after != civil_before:
        raise RuntimeError(
            f"Civil Code point count changed ({civil_before} -> {civil_after}) -- "
            "batch re-indexing must never touch it"
        )
    expected_total = total_before + sum(r.points_after - r.points_before for r in results)
    if total_after != expected_total:
        raise RuntimeError(f"Total point count {total_after}, expected {expected_total}")
    return results, total_before, total_after


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("docs", nargs="*", type=Path, help="document JSON files to (re-)index")
    ap.add_argument("--all", action="store_true", help=f"every *.json in {DEFAULT_DOCS_DIR}/")
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="validate + chunk only")
    args = ap.parse_args()

    paths = list(args.docs) + (list_document_paths(DEFAULT_DOCS_DIR) if args.all else [])
    if not paths:
        ap.error("no documents given (pass file paths or --all)")

    docs = [load_document(p) for p in paths]  # validates every file before touching the index
    ids = [d["doc_id"] for d in docs]
    if len(ids) != len(set(ids)):
        ap.error(f"duplicate doc_id across input files: {ids}")

    threshold, dedupe = load_chunking_params(args.params)
    for path, doc in zip(paths, docs):
        n_chunks = len(chunk_document(doc, threshold, dedupe))
        print(f"[ok] {path}: {doc['doc_id']} -- {len(doc['articles'])} articles, {n_chunks} chunks")
        if path.parent.resolve() != DEFAULT_DOCS_DIR.resolve():
            print(
                f"[warn] {path} is outside {DEFAULT_DOCS_DIR}/ -- the next `dvc repro` "
                "will drop it from the index unless you move it there",
                file=sys.stderr,
            )
    if args.dry_run:
        return

    embed_params, qdrant_params = load_params(args.params)
    # Open the index before loading the model: if the API still holds the
    # lock, fail in a second rather than after a minute of loading BGE-M3.
    try:
        client = QdrantClient(path=qdrant_params["storage_path"])
    except RuntimeError as e:
        sys.exit(f"[error] {e}\n[hint] is the API still running? Local Qdrant is single-process.")

    from sentence_transformers import SentenceTransformer

    print(f"[info] loading {embed_params['model_name']}", file=sys.stderr)
    model = SentenceTransformer(embed_params["model_name"])

    try:
        results, total_before, total_after = reindex_documents(
            client,
            qdrant_params["collection_name"],
            docs,
            model,
            embed_params["batch_size"],
            threshold,
            dedupe,
        )
    finally:
        client.close()

    for r in results:
        print(
            f"[done] {r.doc_id}: {r.articles} articles -> {r.chunks} chunks -> "
            f"{r.points_after} points (was {r.points_before})"
        )
    print(f"[done] collection total: {total_before} -> {total_after} points")


if __name__ == "__main__":
    main()
