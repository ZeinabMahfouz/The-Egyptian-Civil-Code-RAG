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
