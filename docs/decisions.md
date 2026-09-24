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

---

## Generative model: Qwen3 family, dev/production split

**Decision:** Qwen3-1.7B (bfloat16, `transformers` backend) for local
CPU development of the retrieval/citation logic; Qwen3-8B (vLLM
backend) planned for actual serving once GPU access exists.

**Why Qwen3 specifically:** multiple independent sources repeatedly
highlighted the Qwen3 family for Arabic specifically (not just
"multilingual" as a checkbox) -- strong quality across 100+ languages
including Arabic, and separately ranked near the top for RAG
faithfulness in general benchmarks. Same reasoning pattern as the
BGE-M3 embedding choice: prefer evidence naming this exact combination
(Arabic + RAG) over generic leaderboards.

**Why a dev/production split:** `check_gpu.py` confirmed this machine
is CPU-only (`torch.cuda.is_available() == False`). vLLM's design
(PagedAttention, continuous batching) targets GPU inference -- running
it now, on CPU, just to iterate on prompt design would be friction
with no benefit, and the later Locust-at-50-concurrent-users test
wouldn't be meaningful on CPU regardless. Staying within the same
Qwen3 family across both tiers keeps the chat template and prompt
format identical, so the eventual swap to Qwen3-8B + vLLM is a config
change (model name, generation backend), not a rewrite -- `generate_fn`
is injected into `RAGQueryEngine` specifically to make that swap clean.

**Memory constraint found along the way:** this machine has 7.6GB RAM
total, ~6.2GB available to WSL. Qwen3-1.7B in float32 (~6.8GB) plus
BGE-M3 (~2.2GB) exceeded that and crashed the WSL connection outright.
Switched to bfloat16 (~3.4GB for the 1.7B model), bringing total usage
to ~5.6GB -- comfortably inside the available 6.2GB. Documented here
because "the model that scores best on paper" isn't usable if it
doesn't fit the actual hardware -- a real constraint, not a
theoretical one.

**Also fixed:** Qwen3 generates an internal `<think>...</think>`
reasoning block by default, which was leaking into the `answer` field.
Disabled via `enable_thinking=False` at generation time, with a
defensive regex strip as backup in case a future model swap doesn't
fully respect that flag.

**Validated on real queries, not just retrieval in isolation:** the
force-majeure question correctly retrieved and cited Article 147; the
repealed-associations question correctly retrieved the deduped
54-80 range chunk (as a range citation, not 27 separate hits) and
correctly flagged it as no longer in force -- confirming the
repealed-range dedup and citation-by-range design (see chunking
decision above) works end-to-end, not just in the chunking-stage
spot-check.

**Revisit:** formal comparison via MLflow + RAGAS on >=50 questions,
per the course checklist, once vLLM serving and FastAPI exist.

---

## API test isolation: factory pattern, not a bare app instance

**Decision:** `create_app(engine=None)` factory instead of a single
module-level `app` with model loading baked into its lifespan.
Production use calls `create_app()` (real models load via lifespan);
tests call `create_app(engine=FakeEngine())`, which skips the lifespan
-- and therefore all model loading -- entirely.

**Why:** given this project's CPU-only, memory-constrained development
environment (see the generative-model decision above), a test suite
that loads real Qwen3 + BGE-M3 weights just to check that an empty
question returns 422 would be slow and a poor fit for CI. Validated:
the full 5-test suite (three 422 cases, a valid-request check, and
`/health`) runs in under 7 seconds with zero model downloads -- versus
several minutes if real models loaded, based on this project's own
BGE-M3 embedding timings.

**Also caught along the way:** `api.py` initially imported
`backends.py` (and therefore `torch`) at module level, which meant
even importing the API module for a pure-validation test required
torch installed. Fixed by moving that import inside the lifespan
function, where it's actually used -- `qdrant_client` and
`sentence_transformers` remain module-level imports (via
`query.py`, needed for the `RAGQueryEngine` type itself), but those
are already required by the embedding stage, so this doesn't add new
install weight, only avoids the unnecessary torch dependency for a
test path that never touches it.

---

## Docker: bake the vector store in, mount model weights instead

**Decision:** `Dockerfile` `COPY`s `data/processed/qdrant_storage`
(the built vector store) into the image at build time. It does NOT
bake in LLM/embedding model weights (Qwen3-1.7B, BGE-M3) -- those
download on first `docker compose up` into a named volume
(`hf_cache`), persisted across container restarts.

**Why the split:** the course checklist asks specifically for
"the vector store and embedded documents" baked in, not model weights.
There's also a real practical reason to keep them separate: the vector
store is small (tens of MB) and project-specific -- it's the actual
output of this project's work, so it belongs in the image. The model
weights are ~5.6GB combined, generic (reusable by any project using
the same models, not specific to this corpus), and would bloat every
image rebuild if baked in, re-downloading on every `docker build`
during development. A volume-mounted HF cache downloads them once,
ever, regardless of how many times the image gets rebuilt afterward.

**Why CPU-only torch, pinned explicitly:** `pip install torch` on
Linux can resolve to a CUDA-bundled build several GB larger than the
CPU-only wheel, for a machine that -- per this project's own
CPU-only development environment -- will never use the GPU build
anyway. Installed explicitly via the CPU wheel index, before
`requirements.txt`, with `requirements.txt`'s own (locally-frozen,
not guaranteed CPU-only) torch line excluded at build time so it can't
override the explicit choice.

**Revisit:** once GPU access exists for the vLLM/Qwen3-8B serving
target, this Dockerfile needs a GPU-enabled variant (CUDA base image,
GPU-build torch, `--gpus` at run time) -- not a modification of this
one, since the two have genuinely different base image and dependency
requirements.

---

## Retrieval: exact article-number lookup alongside semantic search

**Decision:** `RAGQueryEngine.retrieve()` is two-stage, not purely
semantic. If the question contains an explicit article-number
reference ("Article 147", "مادة ١٤٧"), an exact Qdrant filter lookup
for that article runs first and is prioritized; semantic (dense
embedding) search fills any remaining context slots, or all of them
if no explicit reference is found.

**Why:** caught live, in the Docker deployment's own first smoke test
-- `"What does Article 147 say?"` retrieved Articles 491, 54-80, and
223, but not 147 itself. Root cause: a bare "look up article N"
question carries almost no semantic content for a dense embedding
model to match against -- there's no substantive statement to embed,
just a structured reference. That's a well-known weak point for pure
semantic retrieval, and also plausibly one of the *most* common query
shapes a real user (particularly a lawyer doing a direct lookup,
per this project's stated design goal of citing by article number
"the way a lawyer verifies an answer") would actually type -- worth
fixing properly rather than documenting as an accepted limitation.

**How it's implemented:** `extract_referenced_article_numbers()`
regex-matches "Article"/"article"/"مادة"/"المادة" followed by a number
in either Western or Arabic-Indic digits (reusing the same digit
translation approach as the extraction pipeline), then Qdrant's
`Filter`/`FieldCondition`/`MatchAny` restricts the search to points
whose `article_numbers` payload field contains one of the referenced
numbers. This runs as a genuine vector query with a filter attached
(not a payload-only scan), so it still ranks by relevance among
matching points rather than returning them in arbitrary order.

**Validated:** after the fix, the same failing query correctly
retrieved Article 147 first, and the generated answer was an accurate
summary of its actual content (the force-majeure/unforeseen-
circumstances doctrine, independently verified against the source PDF
earlier in this project) -- not just a citation-format fix, a real
retrieval-quality fix.

**Left as-is, not urgent:** Article 491 still appears as a secondary
source for this query. Worth investigating later whether it's a
genuine topical near-miss (similar to Article 658's near-miss for the
force-majeure semantic query, noted in the embedding-model decision
above) or something else -- not blocking, since the primary result is
now correct.
