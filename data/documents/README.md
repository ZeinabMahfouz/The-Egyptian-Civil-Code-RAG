# Additional legal documents

Legal texts indexed alongside the Egyptian Civil Code: one `<doc_id>.json`
per document. The schema is documented in `scripts/documents.py`. It uses
the same per-article records as `data/interim/civil_code.json`, plus
`doc_id`, `title`, `citation_prefix` and `source`.

Two paths write these documents to the index. Both use the same
validation and chunking code:

- **Incremental:** `python scripts/reindex_batch.py data/documents/<doc_id>.json`
  adds or updates one document in the existing index. It does not re-embed
  the Civil Code. Stop the API first, because local Qdrant is single-process.
- **Full rebuild:** `dvc repro`. The `embed_index` stage indexes the Civil
  Code plus every `*.json` in this folder.

Files here are tracked in git, which keeps them small and reviewable in a PR.
A document that exists only in the index, and not in this folder, is lost
on the next `dvc repro`.

Only put real legal text here, with a verifiable `source`. The test fixture
is in `tests/fixtures/`, not here, so that it never reaches a real index.
