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

---

## RAGAS full-scale scoring: deferred until GPU/stronger judge available

**Decision:** the RAGAS + MLflow harness (`run_ragas_eval.py`) is built,
integrated, and verified structurally correct -- but a full scoring
run against the 54-question evaluation set is deferred until GPU
access and a stronger judge model exist (the planned Qwen3-8B, per the
generative-model decision above), rather than forced through on the
current CPU dev setup.

**Why -- traced to a specific, reproducible failure, not a vague
"it's slow":**
- A single `Faithfulness` judge call took **27 minutes 55 seconds**
  and still failed: `"Prompt statement_generator_prompt failed to
  parse output: the output parser failed to parse the output
  including retries."`
- Root cause: RAGAS's `Faithfulness` metric first asks the judge to
  decompose the generated answer into individual factual statements
  in a specific structured format it parses programmatically, before
  it can check each statement against retrieved context. Our judge
  setup feeds RAGAS's prompts to Qwen3 through a bare
  `HuggingFacePipeline`, which does not apply Qwen3's chat/instruct
  template -- meaning an instruct-tuned model was effectively being
  used as a raw text-completion model, a plausible reason its output
  didn't conform to the format RAGAS needed.
- Even a corrected version of this (via `ChatHuggingFace` applying
  the proper template) doesn't remove the deeper tension: RAGAS's
  structured-output decomposition step was designed assuming judge
  models roughly GPT-3.5/4-class. A 1.7B-parameter local model
  reliably producing exact parseable output for nuanced semantic
  judgment is a real capability stretch, not purely a formatting
  problem -- and at ~28 minutes per attempt just to find out whether
  a fix worked, further blind iteration today was a poor use of time
  against an uncertain payoff.

**What's genuinely proven, not blocked:** the harness's plumbing is
correct -- `import ragas` works (after the vertexai stub workaround),
our own retrieval/generation pipeline runs correctly inside it
(confirmed via the actual answer/citation content in
`reports/ragas_results.json` before scoring failed), RAGAS accepts
our `RunConfig` and custom LLM/embedding wrappers without error, and
the MLflow logging path is wired and ready. What's missing is a judge
capable of completing the scoring step, not code.

**Revisit:** once Qwen3-8B + vLLM (or another sufficiently capable
self-hosted model) is available, retry with `ChatHuggingFace` applying
the proper chat template, and run the full 54-question set across all
four metrics -- both fixes should be attempted together at that point,
not incrementally re-tested on CPU at ~28 minutes per data point.

---

## DVC remote: Google Drive via personal OAuth, not a service account

**Decision:** the project's DVC remote moved from a local folder
(solo-dev only, unreachable by CI) to Google Drive, authenticated via
a personally-registered OAuth client -- not a service account, despite
that being the initially-planned and more commonly-tutorialized
approach for CI use.

**Why not a service account, despite it being the standard tutorial
pattern:** confirmed via direct testing, not assumption -- `dvc push`
failed with Google's own API error: *"Service Accounts do not have
storage quota. Leverage shared drives... or use OAuth delegation
instead."* This is a real, current restriction on personal
(non-Google-Workspace) Drive accounts: a service account can be
granted Editor access to a folder and will correctly authenticate, but
genuinely cannot upload new files into it, regardless of sharing
permissions -- multiple community tutorials describing the
service-account approach as working were either written before Google
tightened this enforcement, or assume Workspace access (Shared
Drives), which a personal account doesn't have.

**Why a personally-registered OAuth client, not DVC's built-in
default one:** DVC ships a shared OAuth client for the standard
interactive-login flow, but Google has been increasingly flagging that
shared client as unverified for many users. Registering an
application-specific OAuth client (Google Cloud Console -> APIs &
Services -> Credentials -> OAuth client ID -> Desktop app) avoided
this entirely.

**WSL-specific issue, not a DVC/Google issue:** the interactive OAuth
flow tries to auto-launch a browser via `gio`, which doesn't work
inside WSL (`gio: ... Operation not supported`) -- it prints a valid
auth URL but nothing opens, and the flow hangs waiting for a redirect
that never arrives. Fix: manually copy the printed URL into a Windows
browser, complete the Google login/consent there, and let the
already-running WSL process pick up the redirect via WSL2's
localhost-forwarding -- no code or config change needed, just a manual
step.

**Credential hygiene, applied throughout:**
- The service account's JSON key and the OAuth client secret were
  kept in `.dvc/config.local` (gitignored by default) via `dvc remote
  modify --local`, never in the committed `.dvc/config`. Caught one
  near-miss directly: an early `dvc remote modify` command was run
  without `--local`, landing the client ID/secret in the tracked file
  -- caught via `git diff .dvc/config` before committing, then moved
  correctly and the client secret was regenerated rather than trusting
  a secret that had briefly been mistyped/exposed in a terminal.
- The downloaded service-account JSON key was moved outside the
  project directory entirely (`~/.secrets/`, `chmod 600`), not stored
  inside the repo tree even with a .gitignore rule protecting it --
  removes the risk class entirely rather than relying on remembering
  an exclusion rule correctly.

**CI wiring:** three GitHub Actions secrets (`GDRIVE_CLIENT_ID`,
`GDRIVE_CLIENT_SECRET`, `GDRIVE_USER_CREDENTIALS`) reconstruct the
same local setup on the runner: client ID/secret via `dvc remote
modify --local`, and the pre-authorized token (from the *local*
one-time interactive login, so CI never has to do interactive OAuth
itself) written to the exact path dvc-gdrive expects --
`~/.cache/pydrive2fs/<client_id>/default.json` -- confirmed via the
developer's own local warning message before ever writing the CI step,
not guessed from documentation alone.

**Result:** `lint`, `test`, `rebuild_index` (full `dvc repro` from
source, including a real ~2.2GB model download and re-embedding), and
`build_and_push_image` (to GHCR, using the automatically-provided
`GITHUB_TOKEN` rather than a fourth credential) all run and pass in
GitHub Actions. The RAGAS faithfulness gate remains the one
intentionally deferred CI stage (see the RAGAS decision above) --
everything else in the course's "lint -> test -> rebuild index -> push
Docker image" checklist item is now real and green, not just
configured.

---

## Batch re-indexing: an incremental path, with DVC still the source of truth

**Decision:** `scripts/reindex_batch.py` adds or updates additional legal
documents (`data/documents/<doc_id>.json`) in the existing Qdrant
collection without rebuilding it. The `embed_index` DVC stage also
indexes every file in `data/documents/`, so `dvc repro` produces the same
index from tracked files. The incremental script is the operational
shortcut. DVC remains the reproducible definition.

**Problems found while designing it:**

- **Point-ID collision.** Point IDs were `uuid5(chunk_id + lang)`, and
  `chunk_id` is the article number. Article 1 of any new law would get
  the same ID as Civil Code Article 1, and the upsert would overwrite the
  Civil Code article without any error. The fix namespaces IDs by
  `doc_id` for every document except the Civil Code. Civil Code IDs keep
  their original form, which a test pins down.
- **Exact article lookup crossed documents.** "What does Article 1
  say?" would match Article 1 of every indexed law. The exact-lookup
  filter is now scoped to `doc_id == egyptian_civil_code`. Semantic
  search still spans all documents, and every citation names its
  source law (`citation_prefix`), so a retrieved non-Civil-Code article
  is never cited as the Civil Code.
- **Stale chunks after an edit.** A plain upsert of an edited document
  leaves points for removed articles in the index, where they can still
  be retrieved and cited. The script upserts first and then deletes that
  document's leftover IDs. It does not delete first, because a crash
  between the two steps would then leave the document missing from the
  index; with this order, a crash leaves old and new points side by side.
- **Mixed embedding spaces.** If `params.yaml`'s embedding model changes
  and the base index is not rebuilt, new vectors would be added to an
  index built by another model. The script checks that the model's
  dimension matches the index and refuses to run on a mismatch. This
  catches a model swap only when the dimension also changes; a full
  `dvc repro` is the real fix.

**Guards:** the script validates every input file before it touches the
index (schema, positive integer article numbers without duplicates,
non-empty Arabic text, and the same 6000-char split-failure threshold as
the corpus gate). The Civil Code's `doc_id` is reserved. The Civil
Code's point count is checked before and after the run and must not
change. The engine refuses to start on an index that has no `doc_id`
payloads. Without that check, an index built before this change would
silently turn exact lookup back into pure semantic search, which is the
Article 147 bug above.

**Known limitation:** local (embedded) Qdrant allows only one process per
storage folder, so the API must be stopped while re-indexing. This was
confirmed live: the script fails fast with a hint rather than after the
model load. Re-indexing while the API keeps serving would need Qdrant in
server mode. That fits the BentoML/serving work, but it is not done yet.

**Tested:** `tests/test_reindex_batch.py` runs against in-memory Qdrant
with a deterministic fake embedder, so CI's `test` job runs it without
downloading a model. It covers: a new document is added and Civil Code
IDs, payloads and vectors are byte-identical; citation prefix; ID
collision; re-runs are idempotent; an edited document drops removed
articles; edited text is re-embedded; the dimension, legacy-index and
validation guards; and exact lookup scoped to the Civil Code. The tests
were confirmed to fail when each fix is reverted. The fixture is
explicitly synthetic (`tests/fixtures/synthetic_law.json`); the real
"new document" run uses a real law in `data/documents/`.
---

## First real additional document: what it exposed

**Document:** Egyptian Consumer Protection Law No. 181 of 2018, Articles 1-2
(`data/documents/consumer_protection_law_181_2018.json`), taken from the
official PDF hosted by the Egyptian Economic Courts (elec.eecourts.gov.eg).
Its purpose is the "batch re-indexing tested with at least one new document"
checklist item. The document is small on purpose: the goal is to prove the
incremental path end to end, not to extend the corpus.

**Result:** `reindex_batch.py` added 2 points (2195 -> 2197). The Civil
Code count did not change. For the query "ما هي حقوق المستهلك عند استخدام
السلع والخدمات؟" the top two sources are Consumer Protection Law Articles 2
and 1. Before the document was added, the same query returned only
unrelated Civil Code articles (86, 804, 802), and the model correctly
answered that it had no relevant information.

**Problems found along the way. Each one is a failure the pipeline could
not catch on its own:**

1. **Placeholder text was embedded.** The first indexed version still
   contained template values (`<official text of Article 2>`). Validation
   only checked that `text_ar` was non-empty, so the placeholders were
   embedded, and they were never retrieved (cosine similarity about 0.33
   against the query, below every Civil Code article in the top 15).
   Diagnosed by reading the stored payloads directly. Fix: `validate_document`
   now rejects `<...>` placeholders, TODO/TBD, trailing ellipses, `text_ar`
   under 20 characters, and a missing or placeholder `source`. Each case has
   a test.
2. **The first draft contradicted the real law.** An early hand-typed
   version of Article 1 defined the consumer by "personal **or
   professional**" needs. The official text says "**غير المهنية**"
   (non-professional), which is the opposite meaning. The early Article 2
   also did not match the official text. Only checking against the official
   PDF caught this: nothing in the pipeline can detect a plausible but wrong
   legal text. From now on every document in `data/documents/` must name
   its source and be copied from it, not retyped.
3. **Arabic PDF copy broke the لا ligature.** Copying from the PDF produced
   "االختيار" and "األضرار" instead of "الاختيار" and "الأضرار": lam and
   alef swapped. This is the same class of problem the Civil Code
   extraction had to handle. The corrupted words were embedded and quoted
   in answers. Fixed in the document with a regex (`ا([اأإآ])ل` -> `ال\1`),
   then re-indexed. It is not yet enforced in validation. A future
   improvement would be to reuse the extraction pipeline's Arabic
   normalization for added documents.
4. **Invalid JSON from raw line breaks.** Text pasted from the PDF brought
   literal newlines into JSON strings, so the file would not load. Fixed by
   re-serializing with `json.loads(..., strict=False)`.

**Generation issues seen in the answer (the CPU dev model, Qwen3-1.7B, not
the retrieval):**

- **Citation misattribution:** the list of consumer rights comes from
  Article 2, but every point was cited as Article 1. Retrieval returned the
  right articles; the model cited them wrongly. RAGAS faithfulness and
  context precision should measure this in the GPU session.
- **Language drift:** a fragment of Chinese ("享有以下权利") appeared in an
  Arabic answer. It is recorded as a baseline to compare against Qwen3-8B.

---

## PII guardrails on /ask: own Egyptian-aware detector, redaction both ways

**Decision:** `src/egyptian_civil_code_rag/pii.py` detects Egyptian
national IDs, Egyptian mobile numbers, EG IBANs, card numbers
(Luhn-checked) and emails. It matches Western and Arabic-Indic digits.
`/ask` redacts the **question** before retrieval and generation, and the
**answer** before it is returned. Matches are replaced with `[ENTITY]`,
and the response lists what was removed in `pii_redacted`.

**Why a custom detector:** Presidio-based detectors, including Guardrails
Hub's `DetectPII`, are built around English entities. They have no
recognizer for a 14-digit Egyptian national ID or an 01x mobile number,
and they do not treat ٠١٢٣٤٥٦٧٨٩ as digits. For this project's users, that
is most of the PII that matters.

**Why redact the question as well:** PII in a question would otherwise
reach the embedding model and the LLM prompt, and every Langfuse trace
once tracing is added. The checklist only requires filtering responses.
Filtering the input is what actually keeps PII out of the system.

**False positives are the main risk in a legal corpus:** questions and
answers are full of article, law and year numbers ("Article 147", "Law
181 of 2018", "Articles 54-80"). Each pattern is anchored on a structure
that those numbers cannot match: a national ID needs a valid century and
birth date, a phone number needs the 01[0125] prefix and 11 digits, and a
card number must pass Luhn. `tests/test_pii.py` includes a set of
legal-text negatives.

**Guardrails library: optional, because of a dependency conflict.** The
detector is wrapped as a Guardrails AI validator (`EgyptianPII`, applied
with `OnFailAction.FIX`), but `guardrails-ai` cannot be installed next to
this repo's pinned stack. This was checked with `uv pip compile` against
`requirements.txt`:
- guardrails-ai 0.8.x-0.10.x pin `click<=8.2.0`, but `huggingface-hub
  1.32` needs `click>=8.4.2`.
- guardrails-ai 0.11.0 needs `openai>=2`, but `instructor==1.3.2` (pulled
  in by RAGAS) needs `openai<2`.

So `PIIGuard` uses the Guardrails `Guard` when the library is installed
and calls the same detector directly when it is not. The output is
identical in both cases, and both paths are tested. Plan: add
`guardrails-ai` to the serving-only environment when BentoML splits
serving requirements from pipeline requirements, since serving does not
need RAGAS.

**Found while testing the wrapper:** Guardrails sends telemetry
(OpenTelemetry spans to an AWS execute-api host) on every `validate()`
by default. This was observed directly as outbound DNS lookups in
testing. A PII guard must never send data out, so `PIIGuard` disables it
in code rather than relying on a `~/.guardrailsrc` that Docker and CI
would not have. A regression test asserts that no network lookups happen.

**Verified live:** asking "رقمي القومي 29801011234567 وموبايلي 01012345678،
هل يحق لي فسخ العقد؟" returned `"pii_redacted": ["EG_NATIONAL_ID",
"EG_PHONE"]`, and neither number appeared in the answer. Side effect: the
1.7B model focused on the `[EG_NATIONAL_ID]` / `[EG_PHONE]` tags instead
of the legal question. A prompt instruction to ignore redaction tags is
planned for the Qwen3-8B prompt work.

---

## Canary rollout: nginx weighted split, gates computed from its access log

**Decision:** `deploy/canary/` runs two CI-built images (stable and canary,
each pinned by git SHA) behind nginx with a weighted `upstream`. Each API
instance labels its responses with `X-App-Release`, which nginx writes to a
JSON access log. `canary_report.py` turns that log into pass/fail promotion
gates: error rate, p95 latency, and a minimum amount of canary traffic.

**Why nginx and not a service mesh:** the course checklist specifies an nginx
canary config, and a single-host Docker Compose deployment has nothing for
Istio, Argo Rollouts or similar tools to manage. The official nginx image's
`templates/` + envsubst mechanism means the weights come from `.env`, so
moving to the next stage is a config change plus an nginx-only restart. The
APIs keep their loaded models.

**Design details that came from testing, not assumptions:**
- **Rollback can't use `weight=0`,** because nginx rejects it. The canary
  line takes `${CANARY_STATE}`, which is empty in normal operation and
  `down` to remove the canary from rotation.
- **`proxy_next_upstream off`:** by default nginx retries a failed request
  on the other upstream. That is kind to users, but it would erase the
  canary's errors from the log, so a broken canary would pass the error
  gate. Retries are therefore off.
- **Header-less failures:** an unhandled 500 or an nginx 502 carries no
  `X-App-Release`. Attributing requests by header alone would file canary
  crashes under "unknown". The report learns which upstream address serves
  which release from the successful responses and attributes failures by
  address. There is a test for this.
- **The split is exact, but only while both upstreams are healthy.**
  Measured with local nginx and two instances of the real FastAPI app (fake
  engine): 25 of 500 requests (5.0%) went to the canary at 95/5. An earlier
  run where the upstreams had been failing showed 6.8%, because `max_fails`
  state skews routing until `fail_timeout` expires. Gates should therefore
  be judged on a window that starts after the stage is stable.
- **Verified live:** after the canary process was killed mid-run, the
  report attributed its 502s correctly (16.7% canary error rate against 0%
  for stable), both the error and traffic gates failed, and it exited 1.
  With `CANARY_STATE=down` and a reload, 200 of 200 requests went to stable.

**Not tested here:** the full `docker-compose.canary.yml` with two real
model-loading API containers. The compose file validates with
`docker compose config`, and nginx was run from the same template with
upstreams substituted for local processes. The full stack needs about
12 GB of RAM.

**Answer-quality gate:** error rate and latency cannot catch a release that
answers quickly and wrongly. For this RAG system that is the more likely
failure mode, for example a retrieval regression or a worse prompt. That
gate is RAGAS faithfulness on the canary image (stage 0 in the README).
It becomes real when the GPU-based RAGAS run and its CI gate land.

---

## Serving: BentoML service with a separate serving environment

**Decision:** `src/egyptian_civil_code_rag/service.py` wraps the same
`RAGQueryEngine` and `PIIGuard` as the FastAPI app in a BentoML service
with an async `/ask`. It is started with
`bentoml serve egyptian_civil_code_rag.service:CivilCodeRAG`. It has its own
environment (`.venv-serve`, `requirements-serve.txt`), separate from the
pipeline's `requirements.txt`.

**Why a separate environment:** serving does not need RAGAS, DVC or MLflow,
and those pins already blocked `guardrails-ai` (see the PII decision). One
environment per role keeps each dependency set solvable. BentoML now has an
environment where it does not compete with RAGAS's `openai<2` pin. The
model weights are shared through `~/.cache/huggingface`, so a second
environment does not download them again.

**Design details:**
- **`workers=1`:** local (embedded) Qdrant allows one process per storage
  folder, so a second worker would fail on the index lock. Concurrency comes
  from async request handling, not from extra processes.
- **Blocking work runs in a thread:** `engine.ask` (embedding plus
  generation) runs under `asyncio.to_thread`, so the event loop keeps
  serving `/health` and queuing requests while a generation is in progress.
- **One generation at a time (`asyncio.Lock`):** on CPU, parallel
  generations compete for the same cores and all finish later. Requests
  queue instead, up to `max_concurrency=8`.
- **`timeout=600`:** BentoML's default of 60 s is shorter than a CPU
  generation.
- **Same contract as FastAPI:** `POST /ask {"question": ...}` returns
  `answer`, `sources` and `pii_redacted`, plus `release`. Two differences:
  an empty question returns 400 (BentoML's `InvalidArgument`) rather than
  FastAPI's 422, and `/health` is a POST, because BentoML APIs are POST.

**Verified live (CPU, Qwen3-1.7B):**
- "What does Article 147 say?" retrieved Article 147 first, and the
  answer matched the official English text almost word for word.
- An Arabic question containing a national ID returned
  `pii_redacted: ["EG_NATIONAL_ID"]` and an Arabic answer about Article 147.
  The answer had some garbled Arabic wording, a known limit of the 1.7B
  model that is logged for the Qwen3-8B comparison.
- A whitespace-only question returned 400.
- `/health` reported 2197 documents indexed.
- The Swagger UI at `localhost:3000` also works as a manual test client.

**Next:** swap the generation backend to vLLM + Qwen3-8B (GPU session),
add streaming, and consider installing `guardrails-ai` in this serving
environment, now that it no longer shares an environment with RAGAS.

---

## Observability: Langfuse self-hosted, one trace per /ask

**Decision:** `src/egyptian_civil_code_rag/pipeline.py` (`TracedPipeline`) runs
the whole `/ask` flow and records one Langfuse trace per request, with a span
for each stage: `ask` → `pii-input` (guardrail), `retrieve` (retriever),
`generate` (generation), `pii-output` (guardrail). The FastAPI app and the
BentoML service both call it, so the two servers trace identically, and each
trace is tagged with the server that answered (`service:fastapi` /
`service:bentoml`). Langfuse runs self-hosted from its official
docker-compose file.

**What each span records:**
- `retrieve`: every cited article with its `doc_id`, language and
  similarity score.
- `generate`: the full prompt, the answer, the model name and token usage.
  The backend exposes `generate.last_usage` as an attribute rather than
  changing its return type, so the RAGAS harness keeps working unchanged.

**Privacy:** the raw question never enters a trace. The question is redacted
before the first span opens. The generation span records the answer *after*
output redaction, so PII the model produces is not stored either. A test
checks the text of every span for a raw national ID and phone number.
Verified live: the traced question containing `29801011234567` has zero
matches anywhere in Langfuse, only `[EG_NATIONAL_ID]`.

**Off by default:** tracing is enabled only when `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` are set, so CI, the tests, and anyone running without
a Langfuse server are unaffected. The tests capture spans with
OpenTelemetry's in-memory exporter instead of a server.

**Found while testing:** Langfuse keeps one client per public key for the
whole process. A second `Langfuse(...)` with the same key silently reuses the
first client's exporter, which made one test pass vacuously (it asserted
that PII was absent from an empty span list). The tests now use a unique key
each, and the leak test also asserts that the redacted tags *are* present,
so an empty result can no longer pass.

**Local setup issues (self-hosting on Windows + WSL + Docker Desktop):**
Langfuse's compose file binds host ports 9000 (ClickHouse) and 5432
(Postgres). Both were already taken on this machine, by a MinIO container and
another project's Postgres. A `docker-compose.override.yml` in the Langfuse
checkout moves them to 19000 and 15432. The Langfuse services communicate
over Docker's internal network, so nothing else needed changing. The
Postgres clash first appeared as "Can't reach database server at
postgres:5432" from the web container, not as a port error.

**Verified live (CPU, Qwen3-1.7B):** one traced request took 1m 38s in
total: generation 1m 35s (489 tokens), retrieval 2.54 s, PII checks under
0.01 s. Generation is effectively the whole latency budget, which is the
case for the vLLM/GPU work.

---

## Monitoring: Prometheus metrics from the pipeline, Grafana provisioned from files

**Decision:** `src/egyptian_civil_code_rag/metrics.py` defines the Prometheus
series. `TracedPipeline` updates them in the same place that writes the
Langfuse spans, so metrics and traces cannot disagree. The FastAPI app serves
them at `GET /metrics`. `deploy/monitoring/` runs Prometheus and Grafana with
the datasource, dashboard and alert rules all provisioned from files in the
repo. Nothing is configured by clicking in the UI, so a reviewer gets the
same dashboard.

**Series:** `rag_requests_total{status}` (ok / no_context / error),
`rag_request_latency_seconds{stage}` (total / retrieve / generate),
`rag_tokens_total{direction}`, `rag_pii_redactions_total{where,entity}`,
`rag_retrieval_top_score`, `rag_ragas_faithfulness`. Label values are small
fixed sets. A question, article number or PII value never becomes a label,
because labels are stored verbatim and each distinct value creates a new
series. A test checks that a redacted national ID does not appear in
`/metrics`.

**Cost per hour:** computed in Grafana as tokens/hour / 1000 x
`$price_per_1k_tokens`, a dashboard variable. The model is self-hosted, so
there is no real per-token price. The variable makes the number an explicit
assumption rather than a constant hidden in code.

**Faithfulness gauge:** read from `reports/ragas_results.json` at startup,
but only when the file has real scores. The committed file comes from the
failed CPU run, where every score is null. Exporting that as 0.0 would fire
the faithfulness alert over an evaluation that never happened. So the gauge
stays unset (no data, alert silent) until the GPU session produces real
scores. There are tests for the null, real and missing-file cases.

**Found live, on the first real dashboard:** the panel showed **0.00%**
anyway. A `prometheus_client` `Gauge` without labels is exported as `0.0`
from the moment it is created, whether or not `.set()` is ever called. So
"no evaluation yet" was published as "faithfulness 0%", and
`RagFaithfulnessLow` would have fired. The unit tests had checked the loader
function but not what `/metrics` actually exports. The fix gives the gauge a
`source` label, because a labelled gauge has no sample until
`.labels(...).set()` is called. A regression test now checks the
`/metrics` output itself, and it fails against the unlabelled version.

**Latency buckets** run from 0.1 s to 300 s, so the same histogram covers
both the CPU backend (about 30 s to 2 min) and the planned vLLM backend
(seconds). The before/after comparison will then be one dashboard, not two.

**Ports:** Prometheus 19090 and Grafana 3001. The defaults are taken on the
dev machine by Langfuse (3000) and its MinIO (9090), the same class of
clash as the Langfuse setup itself.

**Verified here, not assumed:** the Prometheus config and rules pass
`promtool`. A real Prometheus 3.5 scraped the real FastAPI app (with a
simulated engine: 0.2 to 0.8 s generation, a 10% error rate, and a RAGAS
file with a mean of 0.71). The target was `up`. All ten dashboard queries
returned data, including cost/hour ($0.44 at the default price, from
about 3,700 tokens/min) and per-stage p95. `RagFaithfulnessLow` went to
**firing** at 0.71, and `RagHighErrorRate` went to **pending** at the
simulated 10% error rate (it fires after its 5-minute `for:`). The Grafana
dashboard JSON could not be loaded into a real Grafana in that
environment. It is verified on the developer machine (screenshot in the
README).

**Scope:** metrics come from the FastAPI app. BentoML has its own built-in
`/metrics` (request counts and latency per endpoint) but not these
pipeline-level series. Its multiprocess Prometheus setup needs
`bentoml.metrics` rather than plain `prometheus_client`, which is left for
when BentoML becomes the production server.

---

## Streaming: one pipeline, an incremental redactor, a worker thread

**Decision:** `TracedPipeline.ask(question, on_chunk=...)` streams through
the same code path as `/ask`, with the same spans, metrics and PII checks. It
is not a separate streaming implementation. The transformers backend
gained `generate.stream(prompt)` (a `TextIteratorStreamer`, with
`model.generate` running in a thread). FastAPI exposes `POST /ask/stream` as
Server-Sent Events: token events, then a `done` event with sources and
`pii_redacted`. BentoML exposes `ask_stream` as an async generator of text
chunks.

**The hard part is PII redaction on a stream.** A phone number can arrive as
`010` + `1234` + `5678`. Redacting each piece misses it, and sending `010`
before the rest arrives leaks part of it. `StreamingRedactor` always holds
back the last 64 characters, and it never cuts a detected PII match in half:
the release point moves to before the match. The longest realistic PII
value (a spaced EG IBAN, about 35 characters) is therefore seen whole before
any of it is released. It is tested two ways: the streamed output must be
byte-identical to whole-text redaction across hundreds of random token
splits, and a half-arrived number must not be released early. Cost: the
client sees text about 64 characters behind the model. On CPU, where the
first token already takes tens of seconds, that is negligible.

**Why a worker thread:** Langfuse spans use thread-local (contextvar)
context. A streaming response iterated directly would run successive
tokens on different threadpool threads, so spans would be opened in one
thread and closed in another. Instead, the whole pipeline runs start to
finish in one worker thread and pushes redacted chunks into a queue that
the response drains. A test checks that a streamed request produces one
complete trace (ask → pii-input → retrieve → generate → pii-output).

**New metric:** `rag_time_to_first_chunk_seconds`, the time from request
start to the first text the user sees. For streaming, this is the latency
that matters, rather than the total time.

**Errors** mid-stream end with `{"type": "error", "message": "generation
failed"}`. Exception text stays in the trace and logs, so internals such as
memory addresses and paths never reach the client. This is tested.

**Verified here:** both servers were run over a real socket with a
simulated streaming backend (4-character chunks, 50 ms apart). FastAPI
delivered token events about 50 ms apart, then `done`. BentoML delivered
text in about 0.25 s steps, with a phone number split across chunks
arriving as `[EG_PHONE]`. `rag_time_to_first_chunk_seconds` recorded the
request. Not verified here: the real `TextIteratorStreamer` path with
Qwen3, because model downloads are blocked in that environment. That is
checked on the developer machine.

---

## Arabic extraction: reverse glyphs, not characters (the lam-alef bug)

**Found:** every lam-alef ligature in the extracted Civil Code (لا / لأ /
لإ / لآ, about 5,000 occurrences) came out flipped to ال / أل / إل / آل.
Words that appear in the law hundreds of times never appeared correctly:
`إلا` had 0 correct and 288 broken ("إال"), `فلا` 0 vs 148 ("فال"),
`الالتزام` 0 vs 82 ("االلتزام"), `خلال` 0 vs 70 ("خالل"). The bug first
showed up in the added Consumer Protection Law document, then in the Civil
Code contexts inside `reports/ragas_results.json`. All three Arabic contexts
there were affected. It was then measured across the whole extracted text.

**Root cause:** the PDF stores Arabic in visual (left-to-right) order, and
`fix_arabic_word` converted each word to logical order by reversing its
*text string*. A lam-alef ligature is a **single glyph** whose Unicode text
is two characters already in logical order ("لا"). String reversal flipped
those two characters too. The ~5,000 other alef-lam pairs are genuine (the
definite article, "المال", "حالة") and are correct.

**Why not a find-and-replace:** once flattened to text, a flipped ligature
and a genuine alef+lam are identical. For example, "مالك" is correct but
"إال" is broken. Any regex would corrupt correct words. (The small
Consumer Protection Law document was fixed that way, by hand-checking its
two articles. That is not possible for 1,149 articles.) The difference
only exists at the glyph level.

**Fix:** extract words with `return_chars=True` and reverse the word's
*glyphs*, joining each glyph's text unchanged. Verified on a Chromium-rendered
PDF with real ligature shaping: the old method reproduces exactly the broken
forms found in the corpus ("إال خالل االلتزام فال … إلبطال األرض"). The new
method gives "إلا خلال الالتزام فلا … لإبطال الأرض", and genuine "المال" /
"حالة" are unchanged. The Arabic-word check also accepts presentation-form
code points, for PDFs that map glyphs to those instead of base letters.

**Gates:** `tests/test_extract_arabic.py` holds glyph-level unit tests that
run without the PDF. `tests/test_corpus_validation.py` now fails the DVC
`validate_corpus` stage if any of the broken canary words ("إال", "خالل",
"االلتزام", "فال") appears in `text_ar`, or if none of the correct forms
do. This bug went unnoticed through extraction, chunking, indexing and
live answers. The gate means it cannot return silently.

**Impact:** every Arabic chunk and embedding was built from partly
corrupted text. Arabic retrieval quality and any Arabic RAGAS score before
this fix are not comparable with scores after it. The corpus and index are
rebuilt with `dvc repro`, and the GPU RAGAS run uses the fixed corpus.

---

## GPU evaluation: vLLM on Kaggle, the same model as judge, a gate that can go stale

**Decision:** Kaggle runs the GPU work (T4 x2, Internet on). vLLM serves
Qwen3-8B (fp16, tensor-parallel 2) behind its OpenAI-compatible API.
`scripts/gpu_eval.py` builds one index per chunking config with the same
chunking and embedding code as `dvc repro`, answers with the vLLM model,
scores with RAGAS, and logs every run to MLflow. The best config is then
scored on all 54 questions and registered.

**Fixing the CPU judging failure:** the earlier failure (27m55s for one
unparseable faithfulness call) came from a bare `HuggingFacePipeline` with no
chat template, on a 1.7B model. The judge now goes through `ChatOpenAI` to
vLLM. The server applies Qwen3's chat template, and thinking mode is off via
`chat_template_kwargs`. It is an 8B model.

**Known weakness, stated up front:** Qwen3-8B judges its own answers. A
model grading itself tends to be lenient. The alternative was an external
API judge, which would send the evaluation questions and legal context to a
third party and cost money. The mitigation is a second, judge-free metric
logged on every run: `article_hit_rate`, the share of questions whose
expected article appears in the cited sources. Where RAGAS and
`article_hit_rate` disagree, that is a signal about the judge, not
necessarily about retrieval.

**Separate environments, again:** vLLM pins its own torch and transformers.
RAGAS pins `openai<2` through `instructor`. vLLM therefore runs from its own
venv as a server, and evaluation runs in the notebook's Python. This is the
same resolution as guardrails-ai versus the pipeline stack.

**What gets registered:** the "model" in the MLflow Registry is the winning
chunking/embedding config (a params.yaml wrapped as a pyfunc model). The
LLM is a fixed public checkpoint and the config is the part being optimised.
MLflow 3 replaced stages with aliases, so promotion means the alias
`production` (`models:/civil-code-rag-chunking@production`).

**Overlap:** article-level chunks never overlap, so `overlap=0` is logged
explicitly rather than left out. This is a design property, not a missing
experiment.

**CI gate without a GPU:** CI cannot regenerate and judge answers on every
push. `scripts/ragas_gate.py` instead gates on the committed evaluation:
faithfulness on the 20-question CI subset must be at least 0.75, and the
evaluation must still describe the repo. If its corpus md5 (from dvc.lock)
or its chunking/embedding params differ from what the repo now builds, the
gate fails as stale. Changing the corpus or chunker therefore keeps CI red
until the GPU evaluation is re-run and committed. Without the staleness
check, the gate would be a badge, not a gate.

**Tested here (no GPU):** the sweep end to end in dry-run mode, with a fake
embedder and no LLM. That covers real chunking per config, indexing,
retrieval, `article_hit_rate`, one MLflow run per config, best-config
selection, registry with alias, and loading the registered config back.
The vLLM client backend is tested against a mock OpenAI-compatible server
(full answers, streaming, token usage, thinking off). The gate is tested for
pass, below threshold, stale params, stale corpus, too few scored questions,
and rejecting the old CPU report format. **Not tested here:** vLLM on T4
itself and the RAGAS judge calls. That is the Kaggle run.
**Verified live after the rebuild:** the same Arabic question ("ما حكم المادة
147 من القانون المدني؟") on the same model (Qwen3-1.7B). Before the fix, the
answer paraphrased the article with garbled wording ("إداء الالتزام مُعَلَّمًا
أو مُعَلَّمًا بشكل غير مُمكن"). After it, the answer reproduces the official
text: "العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين
أو للأسباب التي يقررها القانون". Three of those words (فلا، ولا، إلا) are
lam-alef words that were corrupted in the old index. The answer still covers
only the first paragraph of the article, which is a limit of the 1.7B model.

## First GPU sweep: what the scores were actually measuring

The first Kaggle sweep (Qwen3-8B on vLLM, 20 questions, 5 configs) gave
faithfulness 0.54–0.60 and answer relevancy 0.48–0.56. Retrieval was fine
(article hit rate 0.875 in every config). Reading the per-question rows
showed four separate causes. Two were real bugs, two were measurement.

| Cause | Example | Kind | Fix |
|---|---|---|---|
| Live articles flagged repealed | Article 2 ("a provision can only be **repealed** by a subsequent law") | data bug | repeal = a notice whose own range covers the article (`is_repeal_notice`) |
| Wrong-language context | English question, Arabic text of Article 43 | retrieval bug | swap each hit for its same-language twin |
| Judge couldn't see article labels | "according to Article 44" counted as unsupported | measurement | RAGAS gets the context exactly as the model saw it (`format_context`) |
| Correct refusals scored ~0 | "the provided articles do not contain information about the minimum wage" | measurement | out-of-corpus questions gated on `refusal_rate`, not averaged into faithfulness |

**The repeal bug was the most serious.** Any article whose text *contained*
"repealed", "abolished" or "ألغي" was flagged. That marked 3 live articles
(2, 388, 1034) as repealed, so the system told users that valid law was "no
longer in force". The flagged set is now exactly the two notice ranges
(54–80 and 389–417, 56 articles), and `test_corpus_validation.py` fails if
anything outside them is flagged.

**Refusals are scored, not hidden.** Excluding out-of-corpus questions from
the faithfulness mean could hide a system that declines everything. Two
metrics prevent that: `refusal_rate` (out-of-corpus questions declined; the
CI gate requires ≥ 75%) and `false_refusal_rate` (in-corpus questions wrongly
declined). Both are logged to MLflow for every config.

**Chunk size made no measurable difference.** All five configs retrieved the
same articles for these 20 questions, and faithfulness spread was 0.07, about
two answers. `best_of` now keeps the incumbent (`baseline-700`, what
`params.yaml` builds) unless a challenger beats it by at least 0.05. The sweep
picked `split-400` on +0.037, which is noise. Re-chunking, re-indexing and
re-deploying for noise is cost and risk with no benefit.

**Still open:** RAGAS asks for 3 generations per answer for answer relevancy;
vLLM through LangChain returns 1 ("LLM returned 1 generations instead of
requested 3"). The metric is still computed, from one generated question
instead of three, so it is noisier.

## GPU evaluation results (Kaggle, after the fixes)

Generator and judge: Qwen3-8B on vLLM (2x T4, fp16). Embeddings: BGE-M3.
Corpus md5 `b5678806de7597f927f542e9ec159392` (corrected repeal flags).

**Sweep: 5 chunking configs on the 20-question CI subset** (16 in-corpus
questions for the RAGAS means, 4 out-of-corpus for refusal):

| Config | Split threshold | Repealed dedupe | Faithfulness | Context precision | Context recall | Refusal rate |
|---|---|---|---|---|---|---|
| **baseline-700** (production) | 700 | yes | 0.857 | 0.812 | 0.812 | 0.75 |
| split-400 | 400 | yes | 0.824 | 0.833 | 0.812 | 0.75 |
| split-1200 | 1200 | yes | 0.902 | 0.812 | 0.812 | 0.75 |
| whole-articles | never | yes | 0.879 | 0.812 | 0.812 | 0.75 |
| no-repealed-dedupe | 700 | no | 0.889 | 0.812 | 0.812 | 0.75 |

Retrieval was identical across configs (article hit rate 0.875 in all five),
so chunk size barely matters for this question set. The best challenger
(split-1200) beat the baseline by 0.044 faithfulness, below the 0.05
`MIN_FAITHFULNESS_GAIN`, so `best_of` kept **baseline-700**. With 15 scored
answers, 0.044 is less than one answer's difference.

**Full run: baseline-700 on all 54 questions**, registered as
`civil-code-rag-chunking` v1, alias `@production`:

| Metric | Score |
|---|---|
| Faithfulness (in-corpus) | **0.896** |
| Context precision | 0.908 |
| Context recall | 0.854 |
| Article hit rate | 0.917 |
| Refusal rate (out-of-corpus declined) | 0.667 (4 of 6) |
| False refusal rate (in-corpus wrongly declined) | 0.021 (1 of 48) |

Answer relevancy and the per-question scores are in
`reports/ragas_results.json`. The CI gate on the same report: faithfulness
**0.868** on the CI subset (threshold 0.75), 3 of 4 out-of-corpus questions
declined: **PASS**.

**Before vs after the fixes** (baseline-700, CI subset): faithfulness
0.567 → 0.857, context precision 0.679 → 0.812, context recall 0.725 →
0.812. The gain comes from fixing two real bugs (repeal flags, retrieval
language) and two measurement errors (judge context, refusal scoring), not
from tuning. See the previous section.

**Known limits.**
- 2 of 6 out-of-corpus questions were answered instead of declined in the
  full run. The prompt says to decline, but the 8B model sometimes answers
  from general knowledge. This is the main remaining hallucination risk.
- Generator and judge are the same model. A model judging its own answers
  may be lenient; a different judge would be a stronger check.
- One faithfulness score per sweep config is missing: the judge hit its
  1024-token output limit (`LLMDidNotFinishException`). The gate requires
  80% of questions to be scored, and 15 of 16 is above that.
- MLflow run durations show only the logging step (milliseconds). The
  evaluation runs before the MLflow run is opened; `eval_seconds` is the
  real duration.

**Evidence:** `reports/mlflow_comparison.png` (all runs with metrics and
params) and `reports/mlflow_registry.png` (the registered model).

## Quantization: AWQ 4-bit serves the same answers ~2.7x faster

**Setup.** `Qwen/Qwen3-8B` (fp16) vs `Qwen/Qwen3-8B-AWQ` (Qwen's official
4-bit AWQ checkpoint). Both run on vLLM with 2x T4 and tensor parallel 2,
with the same retrieval (baseline-700 index) and the same 54 questions.
`scripts/quant_compare.py` runs it, `notebooks/kaggle_quantization.ipynb`
holds the steps, and the results are in `reports/quantization.md`.

**The judge was fp16 for both answer sets.** Letting AWQ judge its own
answers would change the generator and the judge at once, so a difference
could come from either. Both servers can't fit on the GPUs together, so the
run was: AWQ answers → stop → fp16 answers → fp16 judges both.

| | fp16 | AWQ 4-bit | |
|---|---|---|---|
| Faithfulness | 0.886 | 0.888 | drop −0.002 (limit 0.03): **PASS** |
| Context precision / recall | 0.908 / 0.854 | 0.917 / 0.854 | same retrieval |
| Answer relevancy | 0.701 | 0.706 | |
| Out-of-corpus declined | 4 of 6 | 4 of 6 | |
| Latency p50 / p95 (one user) | 2.81 s / 9.42 s | **1.06 s / 2.51 s** | 2.7x / 3.8x faster |
| Time to first token p50 | 0.24 s | 0.17 s | |
| Decode speed p50 | 18 tok/s | **55 tok/s** | 3.0x |
| 8 concurrent: requests/s | 2.24 | **4.21** | 1.9x |
| 8 concurrent: latency p95 | 8.40 s | 3.92 s | |
| Weights per GPU | 7.64 GiB | **2.85 GiB** | 2.7x less |

**Why it's faster, not slower.** On the T4, vLLM can't use the Marlin
kernels (they need a newer GPU), so the 4-bit weights are unpacked by the
plain AWQ kernel. That kernel costs compute. But generating tokens one at a
time is limited by how fast the weights are read from GPU memory, not by
compute, and 4-bit weights are about a quarter of the fp16 bytes. Reading
less memory per token outweighed the unpacking cost.

**Quality didn't move.** The +0.002 faithfulness is noise (about a tenth of
one answer). The honest claim is "no measurable loss", not "AWQ is better".

**Decision:** serve AWQ. It gives the same answers at a third of the
latency with a third of the weight memory. For serving, set
`GEN_MODEL=Qwen/Qwen3-8B-AWQ` with `VLLM_BASE_URL`. fp16 stays the RAGAS
judge, so evaluation keeps one fixed yardstick.

**Limits.**
- One run on one day: the latency numbers are a single measurement, not an
  average over runs.
- 54 questions: "no measurable loss" holds at this scale. A small
  degradation (under ~0.02) would not be visible.
- Fitting AWQ on a single T4 (it would need about 5.7 GiB in total) is
  plausible from the weight size, but was not measured.
- Throughput is counted in streamed chunks, which vLLM emits at about one
  token each, so it is close to tokens/s but not exact.

## Repealed ranges: state the arithmetic, don't ask the model to do it

**Found during the reviewer dry run** (a fresh clone, the published image,
CPU model Qwen3-1.7B). Asked "Is Article 400 still in force?", the system
retrieved the right chunk, `Articles 389-417 (REPEALED)`, and still answered
"the provided articles do not contain information about Article 400". The
small model didn't infer that 400 lies between 389 and 417. Qwen3-8B on
Kaggle got the same question right, so the eval didn't catch it; the image
reviewers run uses the small model.

**Fix:** `format_context` now adds an explicit line when the question names an
article inside a repealed range: "Note: Article 400 is within this range, so
Article 400 is REPEALED." This is deterministic code, not something we hope
the model works out. The RAGAS judge sees the same line, because the contexts
are built by the same function.

**Effect on the recorded evaluation:** this changes the prompt only for
questions that name an article inside a repealed range (3 of the 54: r03_en, r04_ar, r09_en). The
committed GPU results predate the change, so for those rows they understate
the current system slightly. They don't overstate it.

## Load test: 50 users, no failures, saturated at ~5.7 requests/s

**Setup.** The configuration chosen for serving: the FastAPI app (PII
redaction, retrieval, prompt, metrics; the same code as the Docker image)
calling Qwen3-8B-AWQ on vLLM (2x T4, tensor parallel 2). Locust ran on the
same Kaggle machine (`load_test/locustfile.py`): users ask random questions
from the 54-question eval set, waiting 0.5–2 s between requests. An empty
answer counts as a failure. Run with `notebooks/kaggle_load_test.ipynb`; the
raw Locust HTML and CSV are in `reports/`.

| Users | Duration | Requests | Failures | Throughput | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 min | 47 | 0 | 0.40 req/s | 1.2 s | 2.2 s | 2.5 s | 2.5 s |
| **50** | 5 min | 1,692 | **0** | **5.65 req/s** | 7.0 s | **12.0 s** | 15.0 s | 18.1 s |

**Reading it.** Throughput rose 14x and nothing failed, but the server is
saturated: requests queue, so latency grows. The numbers are consistent with
a closed system at capacity: 50 users ÷ 5.65 req/s ≈ 8.8 s per cycle, which is
about 7.6 s average latency plus 1.25 s average think time.

**Where the time goes under load.** These are the API's own Prometheus
histograms (`reports/load_metrics.txt`), averaged over all 1,778 requests:

| Stage | Mean | Share |
|---|---|---|
| Retrieve (embed the question + Qdrant search) | 2.45 s | 38% |
| Generate (vLLM) | 4.02 s | 62% |
| Total | 6.47 s | |

Generation dominating is expected. **Retrieval taking 2.45 s is not:** it is
one embedding plus a search over 2,197 points, which should take tens of
milliseconds (not measured separately here: the histogram mixes both runs). The likely causes are that the
local (embedded) Qdrant client and the BGE-M3 encoder run inside the API
process and serialize across FastAPI's worker threads, and that the
same-language swap does a filtered `scroll` per hit, which local Qdrant
evaluates by scanning. This is the first thing to fix for more capacity:
run Qdrant as its own server (the official `qdrant/qdrant` image), and batch
question embeddings. That should return about a
third of the latency without touching the GPU.

**Limits.** Locust ran on the same machine as the server, so it competed for
CPU. A single 5-minute run is one measurement, not an average. The 1-user p95
(2.2 s) is the latency one user waits; the 50-user p95 (12 s) is the one that
matters for capacity planning.

## Query drift: alert on "does the Civil Code cover this?", not on phrasing

**Question.** Are users still asking what the system was built and evaluated
for? If not, the RAGAS scores no longer describe real traffic, and refusals
will rise. `scripts/embedding_drift.py` checks a batch ("window") of queries
against the 54 evaluation questions, with BGE-M3. Three test windows of 16
new questions each, half Arabic, none from the eval set
(`tests/eval/drift_windows.json`): `in_domain` (new Civil Code questions),
`other_jurisdiction` (Saudi labour law and VAT; Egyptian criminal, tax and
company law), and `off_topic` (weather, recipes, football).

**First attempt: centroid drift.** 1 − cosine between the mean embedding of
the window and of the eval set, with the threshold calibrated by drawing 1000
random eval samples of the same size (the 99th percentile of drift that
happens by chance).

| Window | Drift | Threshold | Flagged |
|---|---|---|---|
| in_domain | 0.120 | 0.090 | yes ✗ |
| other_jurisdiction | 0.231 | 0.090 | yes |
| off_topic | 0.298 | 0.090 | yes |

It ranks the windows correctly, but it **flags ordinary Civil Code
questions**. The reason is the baseline's mix: a third of the eval set is
templated lookups ("What does Article 43 say?", "Is Article 400 still in
force?"), which pull the eval centroid toward that template. A window of
normal full-sentence questions differs in *phrasing*, and the threshold only
accounts for sampling noise, not for a different mix of question types. An
alert that fires on normal traffic gets ignored, so this signal is reported
but not alerted on.

**What alerts instead: off-corpus share.** For each query, the cosine
similarity of its best match among the *indexed articles*: does the Civil
Code contain anything close to this? The cut-off (0.646) is the 5th
percentile of that score over the 26 substantive eval questions, so about 5%
of normal questions fall below it by chance. Article lookups are left out
because they are answered by an exact filter, not by similarity. A window is
flagged when the number of queries below the cut-off reaches the binomial
limit for a 5% base rate at a 1% false-alarm rate: 4 of 16.

| Window | Mean best-match similarity | Below cut-off | Limit | Flagged |
|---|---|---|---|---|
| in_domain | 0.693 | 2 of 16 | 4 | no ✓ |
| other_jurisdiction | 0.498 | **16 of 16** | 4 | **yes** ✓ |
| off_topic | 0.449 | **16 of 16** | 4 | **yes** ✓ |

The separation is wide: every question from another jurisdiction or off
topic falls below the cut-off, and normal questions stay under the limit.
Each window is half Arabic and half English; the results weren't split by
language.

**Where it shows up.** `reports/drift.md` and `drift.json`; an MLflow
experiment `civil-code-rag-drift` (one run per window); Prometheus gauges
`rag_query_off_corpus_share` and `rag_query_off_corpus_limit` (plus the
centroid gauges) loaded when the API starts; the `RagQueryDrift` alert in
`deploy/monitoring/prometheus/alerts.yml`; and a Grafana panel. The live
counterpart is the existing "Retrieval top score" panel, which tracks the
same similarity on every real request.

**Limits.**
- The windows are written by hand to simulate drift. Real traffic would come
  from logged (PII-redacted) questions.
- 16 queries per window: the binomial test needs at least 4 low-similarity
  queries to flag, so slow drift is caught later than sudden drift.
- The cut-off comes from 26 questions, so it is approximate. With more real
  traffic it should be recalibrated.
- In-domain had 2 of 16 below the cut-off (the expected rate is 5%, about
  1 of 16). That is within chance, but the margin to the limit is small.

## Alert delivery: Alertmanager with a local webhook receiver

**Problem.** The alert rules fired in Prometheus, but nothing was delivered.
An alert that only shows on a web page nobody watches doesn't help.

**Decision.** Prometheus → Alertmanager → a webhook. The webhook target is
a ~60-line standard-library receiver (`deploy/monitoring/alert-receiver/`)
that writes one line per alert to stdout and to a log file.
- **Why a webhook and not email or Slack.** Both need an account and a
  secret, which a reviewer can't run and which would sit in the repo. The
  routing (group by alert name, 10 s wait, critical repeated hourly,
  warnings every 4 h, resolved notifications on) is the part worth
  reviewing. Swapping the receiver for `slack_configs` doesn't change it.
- **How to show it without breaking anything.** The real RAGAS report
  scores 0.896, so the faithfulness alert is quiet. The API now reads the
  report path from `RAGAS_REPORT` (default unchanged), and
  `deploy/monitoring/demo/ragas_low_faithfulness.json` (marked DEMO ONLY)
  scores 0.62. The demo report gets its own gauge label, so it can't be
  mistaken for the real report on the dashboard.

**Bug found while wiring it.** The drift section above says the alert
fires on off-corpus share, but the rule and gauges were still the centroid
ones. Under Alertmanager that would have delivered a false alert for the
in_domain window (centroid drift 0.120 against a threshold of 0.090). The
rule now compares `rag_query_off_corpus_share` with
`rag_query_off_corpus_limit`. A test checks that the alert flags exactly
the windows `reports/drift.json` flags.

**Verified.** Ran locally with the release binaries (Prometheus 3.5.0,
Alertmanager 0.28.1) and the API on the demo report. Within a minute the
receiver got `RagFaithfulnessLow` (critical) and `RagQueryDrift` for
other_jurisdiction and off_topic, but not in_domain. Restarting the API
without the demo report produced a RESOLVED line one group interval later.
`amtool check-config` and `promtool check rules` pass.

**Limits.** One receiver for every alert. A team would route quality alerts
(faithfulness, drift) and serving alerts (errors, latency, API down) to
different people. There are no silences or inhibit rules: for example,
`RagApiDown` should mute the latency alert.

## Next steps (not done, in priority order)

1. **Fix the retrieval bottleneck.** Under 50 users, retrieval is 38% of
   request time. Run Qdrant as a server instead of the in-process client,
   and batch query embeddings.
2. **Refuse below the drift cut-off.** The 0.646 corpus cut-off from the
   drift check could also decline a single question. 2 of 6 out-of-corpus
   questions were answered instead of declined. This needs its own
   evaluation, because it also declines some valid questions.
3. **Re-ranker** (e.g. BGE reranker) over the top-k. Check it against the
   RAGAS gate, not by eye.
4. **RAGAS on live traffic.** Score a sample of Langfuse traces and attach
   the scores to the traces, so faithfulness is measured on real questions
   and not only the 54-question set.
5. **Trend in MLflow.** Plot gated RAGAS metrics across evaluation runs, so
   slow regressions show before they cross the gate.
6. **CI access to the DVC remote.** The Google OAuth app is in Testing, so
   its refresh token expires every 7 days and `dvc pull` in CI fails until
   it is renewed. A service account, or a published OAuth app, removes the
   manual step.
7. **Package GPU serving.** The container runs the CPU model. A vLLM image
   with the AWQ model would match what was evaluated.
