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
