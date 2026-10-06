"""Point IDs, computed exactly as scripts/embed_and_index.py assigns them.

IDs are deterministic, so the query engine can fetch a chunk's other-language
copy directly by ID instead of searching for it.

scripts/embed_and_index.py has the same function. It isn't imported from
here because that script is a `dvc repro` dependency: editing it would re-run
the embedding stage. tests/test_qdrant_seed.py checks the two agree."""

import uuid

# Every indexed point carries a doc_id payload. The Civil Code's is fixed;
# additional laws added via scripts/reindex_batch.py get their own.
CIVIL_CODE_DOC_ID = "egyptian_civil_code"

# Fixed namespace so point IDs are deterministic across runs, not random.
ID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def point_id(chunk_id: str, lang: str, doc_id: str = CIVIL_CODE_DOC_ID) -> str:
    """Civil Code IDs keep their original form (chunk_id-lang), so existing
    point IDs are unchanged. Every other document is namespaced by doc_id:
    without it, Article 1 of a new law would get the *same* ID as Civil
    Code Article 1 and the upsert would silently overwrite it."""
    key = f"{chunk_id}-{lang}" if doc_id == CIVIL_CODE_DOC_ID else f"{doc_id}:{chunk_id}-{lang}"
    return str(uuid.uuid5(ID_NAMESPACE, key))
