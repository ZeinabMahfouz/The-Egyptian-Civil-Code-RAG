# Design decisions

Lightweight log of decisions that needed real justification, not just
"we picked something" -- for the project write-up and for future-me.

---

## Chunking: article-level, not fixed token windows

**Decision:** one article = one chunk by default. Long articles (>700
chars) with genuine numbered-paragraph markers split by paragraph,
every sub-chunk keeps the parent `article_number`. Contiguous repealed
articles sharing identical placeholder text (54-80, 389-417) collapsed
into 2 deduped chunks instead of 56 near-duplicate vectors.

**Why:** a legal article is a self-contained unit of meaning with a
natural citation (course Module 2 checklist). Fixed 512-token windows
would cut mid-provision and destroy the citation. Corpus length stats
(median 199 chars AR / 315 EN, max 1417/2141) showed no article
actually needs splitting for embedding context-window reasons --
paragraph splitting exists for retrieval precision on the long tail,
not necessity.

**Result:** 1149 articles -> 1102 chunks. See `diagnostics/README.md`
for the analysis and spot-check scripts.

**Revisit:** chunk_size/overlap thresholds are tracked params
(`params.yaml`) meant to be swept properly via MLflow once RAGAS
faithfulness scoring exists end-to-end -- this is the reasoned
starting point, not the final tuned value.

---

## Embedding model: BGE-M3

**Decision:** BAAI/bge-m3 over intfloat/multilingual-e5-large.

**Why:**
- A direct empirical study on Arabic RAG pipelines specifically
  ("Optimizing RAG Pipelines for Arabic: A Systematic Analysis of Core
  Components") found bge-m3 and multilingual-e5-large as the two
  strongest performers for this exact task, ahead of Arabic-specialized
  models -- narrowed the field to these two rather than picking from
  general MTEB leaderboards alone.
- A 4-query smoke test (2 known-ground-truth articles, AR + EN each)
  scored an identical 3/4 recall@3 for both models -- both missed the
  same query (Article 147 vs. the topically-adjacent Article 658),
  which is a real semantic near-miss between two related legal
  concepts, not a broken model.
- Tie-broken on practical grounds: BGE-M3's 8192-token context is far
  more headroom than the corpus needs today but is real
  future-proofing; it natively produces dense + sparse + ColBERT
  embeddings in one pass, leaving hybrid retrieval on the table without
  a model swap later; both are MIT-licensed and self-hostable, which
  matters for routing legal document content through the pipeline
  without an external API call.

**Caveat:** the 4-query smoke test is a sanity gate, not a statistical
comparison -- it exists to rule out an obviously broken candidate
before investing in the full pipeline around it, not to make the final
call. See `diagnostics/compare_embedding_models.py`.

**Revisit:** formal comparison via MLflow + RAGAS on >=50 questions
once `/ask` exists end-to-end, per the course checklist. This decision
may be confirmed or overturned by that data.

---

## Vector store: Qdrant

**Decision:** Qdrant over Chroma.

**Why:** the corpus itself (1102 chunks) is small enough that either
would work today, but the project's later sessions explicitly require
production-shaped serving (Locust load testing at 50 concurrent users,
quantization, canary rollout, Prometheus/Grafana monitoring) that
Chroma is not built for -- multiple current sources describe Chroma as
the prototype/notebook choice and Qdrant as the one that goes from
laptop to production without a second migration. Picking Chroma now
would likely mean migrating mid-project once those sessions hit.
Additional fit: Qdrant's native hybrid dense+sparse retrieval (named
vectors + sparse vector API) directly matches BGE-M3's dense+sparse+
ColBERT output, leaving hybrid retrieval available later without a
database swap; its metadata filtering (rich, indexed, pre/post-filter)
fits the citation-heavy retrieval this project needs (`is_repealed`,
`article_numbers`, `book`/`chapter`/`section`) better than Chroma's
basic key-value filtering.

**Practical note:** using Qdrant's local/embedded mode (no server
process) for pipeline development; the same client API switches to a
real Docker-run server for the serving stages later.

---

## Embedding indexing: separate points per language, not combined

**Decision:** each chunk produces up to two Qdrant points (one for
`text_ar`, one for `text_en`), both carrying identical citation
metadata plus a `lang` field, rather than concatenating both languages
into a single embedding per chunk.

**Why:** deferred from the chunking stage. The corpus is small enough
(~2200 points either way) that doubling costs nothing operationally,
and combining two languages into one embedding risks diluting either
language's match precision for a task where citation precision is the
whole point. This also naturally handles the asymmetric chunks from
Articles 238/658 (Arabic-only paragraph chunks, one English-whole
chunk) without special-casing.

**Reproducibility note:** point IDs are deterministic (`uuid5` of
`chunk_id + lang`, not random), so re-running the indexing script is
an idempotent upsert rather than a duplicate insert -- required for
the "batch re-indexing with a new document" checklist item later.
