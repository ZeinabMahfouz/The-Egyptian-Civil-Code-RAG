"""Additional legal documents indexed alongside the Egyptian Civil Code.

A document lives in data/documents/<doc_id>.json and uses the same
per-article record schema as data/interim/civil_code.json, wrapped with
document-level metadata:

    {
      "doc_id": "law_4_1996",                 # slug, unique, never the Civil Code's
      "title": "Law No. 4 of 1996 ...",
      "citation_prefix": "Egyptian Law No. 4 of 1996",
      "source": "where the text was taken from (URL / gazette issue)",
      "articles": [
        {"article_number": 1, "book": "", "chapter": "", "section": "", "topic": "",
         "text_ar": "...", "text_en": "...", "is_repealed": false, "source_page": 1},
        ...
      ]
    }

Used by both paths that write to the index, so they can't drift apart:
  - scripts/embed_and_index.py  (full rebuild, `dvc repro`)
  - scripts/reindex_batch.py    (incremental add/update of one or more documents)
"""

import json
import re
from dataclasses import asdict
from pathlib import Path

from chunk_corpus import build_chunks

from egyptian_civil_code_rag.query import CIVIL_CODE_DOC_ID

# Same threshold as tests/test_corpus_validation.py: a record past this
# almost certainly means a failed article split, not a genuinely long article.
MAX_REASONABLE_TEXT_LENGTH = 6000
RE_DOC_ID = re.compile(r"^[a-z0-9][a-z0-9_]{2,63}$")
REQUIRED_DOC_FIELDS = {"doc_id", "title", "citation_prefix", "source", "articles"}
REQUIRED_ARTICLE_FIELDS = {
    "article_number",
    "book",
    "chapter",
    "section",
    "topic",
    "text_ar",
    "text_en",
    "is_repealed",
    "source_page",
}


class DocumentValidationError(ValueError):
    pass


def validate_document(doc: dict) -> None:
    """Fail loudly before anything is embedded -- same discipline as the
    corpus validation gate: a silent parsing bug here becomes a
    hallucination three steps later."""
    missing = REQUIRED_DOC_FIELDS - doc.keys()
    if missing:
        raise DocumentValidationError(f"document missing fields: {sorted(missing)}")

    doc_id = doc["doc_id"]
    if not isinstance(doc_id, str) or not RE_DOC_ID.match(doc_id):
        raise DocumentValidationError(
            f"doc_id {doc_id!r} must be a lowercase slug (a-z, 0-9, _; 3-64 chars)"
        )
    if doc_id == CIVIL_CODE_DOC_ID:
        raise DocumentValidationError(
            f"doc_id {CIVIL_CODE_DOC_ID!r} is reserved -- the Civil Code is rebuilt "
            "from the PDF by `dvc repro`, never through this path"
        )
    if not doc["citation_prefix"].strip():
        raise DocumentValidationError(f"[{doc_id}] citation_prefix must not be empty")

    articles = doc["articles"]
    if not isinstance(articles, list) or not articles:
        raise DocumentValidationError(f"[{doc_id}] 'articles' must be a non-empty list")

    seen = set()
    for i, rec in enumerate(articles):
        missing = REQUIRED_ARTICLE_FIELDS - rec.keys()
        if missing:
            raise DocumentValidationError(f"[{doc_id}] article #{i} missing {sorted(missing)}")
        n = rec["article_number"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise DocumentValidationError(
                f"[{doc_id}] article #{i}: article_number must be a positive int, got {n!r}"
            )
        if n in seen:
            raise DocumentValidationError(f"[{doc_id}] duplicate article_number {n}")
        seen.add(n)
        if not rec["text_ar"].strip():
            raise DocumentValidationError(f"[{doc_id}] article {n}: empty text_ar")
        for field in ("text_ar", "text_en"):
            if len(rec[field]) > MAX_REASONABLE_TEXT_LENGTH:
                raise DocumentValidationError(
                    f"[{doc_id}] article {n}: {field} is {len(rec[field])} chars "
                    f"(> {MAX_REASONABLE_TEXT_LENGTH}) -- likely a failed article split"
                )


def load_document(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    validate_document(doc)
    return doc


def list_document_paths(docs_dir: Path) -> list[Path]:
    """Every *.json in docs_dir, sorted so indexing order is deterministic.
    A missing directory means "no extra documents", not an error."""
    if not docs_dir.exists():
        return []
    return sorted(docs_dir.glob("*.json"))


def chunk_document(doc: dict, threshold: int, dedupe_repealed: bool) -> list[dict]:
    """Chunk with exactly the Civil Code's strategy (article-level,
    paragraph-split past the threshold), but with the document's own
    citation prefix."""
    chunks = build_chunks(
        doc["articles"], threshold, dedupe_repealed, citation_prefix=doc["citation_prefix"]
    )
    return [asdict(c) for c in chunks]
