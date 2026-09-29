# The Egyptian Civil Code RAG

An Arabic legal document RAG (Retrieval-Augmented Generation) system built
on Egypt's Civil Code (القانون المدني المصري) -- a bilingual (Arabic/English),
170-page, 1149-article legal text. Built as the final MLOps project for the
ITI MLOps course.

## What this is, concretely

A reproducible pipeline that takes the raw bilingual PDF and produces a
validated, citable, retrieval-ready corpus:

```
data/raw/civil_code.pdf
        |
        v
[extract_corpus]   -- pdfplumber word-position column splitting (AR/EN are
        |              side-by-side columns, not stacked blocks), article
        |              boundary detection, repealed-range expansion,
        |              Arabic/English header cross-checking
        v
data/interim/civil_code.json   (1149 articles, one record each)
        |
        v
[validate_corpus]  -- pytest suite: numbering contiguity, non-empty text,
        |              no oversized records, repealed articles correctly
        |              flagged. Hard gate -- nothing downstream can run
        |              against unvalidated data.
        v
data/interim/.validated
        |
        v
[chunk_corpus]     -- article-level chunking (not fixed token windows --
        |              a legal article is a self-contained citable unit).
        |              Long multi-paragraph articles split by paragraph,
        |              parent article number preserved on every sub-chunk.
        |              Repealed-range duplicates (56 near-identical
        |              placeholder articles) collapsed to 2 chunks.
        v
data/interim/chunks.json   (1102 chunks)
```

Every stage is a `dvc.yaml` stage. `dvc repro` rebuilds the entire corpus
from the raw PDF deterministically -- no manual steps, no hidden state.

## Why this was harder than "run pdftotext and parse regex"

The source PDF's Arabic and English columns are laid out side-by-side
per physical line, not as separate blocks -- naive text extraction
interleaves both languages on every line. Getting a clean bilingual
corpus out of it required word-position-based column splitting
(pdfplumber), correcting for RTL character-reversal in the Arabic
extraction, and handling several real defects in the source document
itself (a promulgation-law preamble that reuses article numbers 1-2
before the real code starts, a repealed-article block with no header
line, a genuinely empty table cell for one article's Arabic text). Each
of these was diagnosed from real extracted data before being fixed --
see `diagnostics/README.md` for the full trail.

## Stack

Python, pdfplumber, BGE-M3, Qdrant, Qwen3, transformers (dev) / vLLM
(serving), RAGAS, Langfuse, BentoML, MLflow, DVC, Docker, GitHub Actions.

## Project structure

```
data/
  raw/            source PDF, DVC-tracked
  interim/        pipeline intermediates (civil_code.json, chunks.json)
  processed/      pipeline outputs -- Qdrant vector index (data/processed/qdrant_storage)
  documents/      additional legal documents indexed alongside the Civil Code
scripts/          pipeline code only (extract_corpus.py, chunk_corpus.py,
                  embed_and_index.py, reindex_batch.py, documents.py,
                  topic_overrides.json, manual_patches.json)
diagnostics/      one-off debugging/analysis scripts, not part of the
                  pipeline -- documents how failures were found and fixed
tests/            pytest validation suite
notebooks/        exploratory work
src/              installable package (pip install -e .) -- RAG query engine
                  (egyptian_civil_code_rag/query.py: retrieval, dedup, citation)
dvc.yaml          pipeline stage definitions
params.yaml       tunable pipeline parameters (chunking thresholds, etc.)
```

## Reproducing this

## Adding a legal document

Additional laws are indexed alongside the Civil Code from `data/documents/`
(the schema is in `scripts/documents.py`, and the folder has its own README).
Copy the text from an official source and name that source in the file.

```bash
python scripts/reindex_batch.py --dry-run data/documents/<doc_id>.json   # validate only
# stop the API first -- local Qdrant is single-process
python scripts/reindex_batch.py data/documents/<doc_id>.json             # add or update
```

`dvc repro` rebuilds the same index from scratch, including every file in
`data/documents/`.

**Full pipeline, from source (rebuilds extraction -> chunking -> embedding -> index):**

```bash
git clone <repo-url>
cd The-Egyptian-Civil-Code-RAG
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt install poppler-utils   # pdftotext/pdfinfo, required by extraction

dvc pull      # fetch the DVC-tracked raw PDF and pipeline outputs
dvc repro     # rebuild everything from source, verifying it reproduces
```

**Just run the Q&A API on any machine, without rebuilding anything** --
3 commands, per the course checklist:

```bash
dvc pull                # fetches the pre-built vector store this Dockerfile bakes in
docker compose up --build
curl http://localhost:8000/health   # wait for {"status":"healthy",...}, then:
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What does Article 147 say?"}'
```

First run downloads ~5.6GB of model weights (Qwen3-1.7B + BGE-M3) into
a persistent Docker volume -- slow once, instant on every run after.
The vector store itself (your corpus, already embedded) is baked into
the image at build time, so no DVC access is needed once the image
exists; `dvc pull` is only how you get that data onto disk *before*
`docker build` runs.

Configure your own DVC remote first if you're not pulling from the
project's existing one. The remote is Google Drive via a personal
OAuth client, not a service account -- see the "DVC remote" entry in
`docs/decisions.md` for why, and for the exact setup steps if you're
reproducing this from scratch on a new machine.

**Note on the generative model:** the Docker image currently runs the
same CPU-based `transformers` backend used for local development
(Qwen3-1.7B), not the Qwen3-8B + vLLM production target -- this
project's development environment is CPU-only (see
`docs/decisions.md`), and vLLM's GPU-oriented design isn't a good fit
without one. Swapping to vLLM + GPU is a planned follow-up, not yet
done.

## Canary rollout

A new release does not replace the running one all at once. It is
deployed next to it, and nginx sends it a small, fixed share of traffic
while the two are compared. Everything is in `deploy/canary/`:

| File | Purpose |
|---|---|
| `docker-compose.canary.yml` | `api-stable` + `api-canary` (two CI-built images, pinned by git SHA) behind `nginx` on port 8080 |
| `default.conf.template` | Weighted upstream (`STABLE_WEIGHT` / `CANARY_WEIGHT`), JSON access log recording which release served each request |
| `.env.example` | Images, release names, and the current stage's weights |
| `canary_report.py` | Reads the access log, compares canary vs stable, exits 0 = promote / 1 = hold or roll back |

Every API response carries an `X-App-Release` header, and `/health` includes
`release`. That is how the log, and you, can tell which build answered.

**Rollout stages:**

| Stage | `STABLE_WEIGHT` / `CANARY_WEIGHT` | Minimum before moving on |
|---|---|---|
| 0. Pre-flight | - | CI green on the canary SHA; RAGAS faithfulness on the canary image >= 0.75 (the CI gate) |
| 1. Canary 5% | 95 / 5 | >= 20 canary `/ask` requests and all `canary_report.py` gates pass |
| 2. Canary 25% | 75 / 25 | gates pass again on the new window |
| 3. Canary 50% | 50 / 50 | gates pass; spot-check 10 canary answers for correct citations |
| 4. Promote | set `STABLE_IMAGE` to the canary SHA, `CANARY_STATE=down` | - |

**Gates** (`canary_report.py`, adjustable by flag):

- The canary's 5xx rate is at most the stable rate plus 1 percentage point.
- The canary's `/ask` p95 latency is at most 1.2 times the stable p95.
- There is enough canary traffic to judge.

A failure that has no release header (an app crash or an nginx 502) is
attributed by upstream address, so a crashing canary can't hide from the
error gate.

**Commands:**

```bash
cp deploy/canary/.env.example deploy/canary/.env      # fill in the two SHAs
docker login ghcr.io                                   # GHCR images are private by default
docker compose -f deploy/canary/docker-compose.canary.yml --env-file deploy/canary/.env up -d
# ...send traffic to http://localhost:8080 ...
python deploy/canary/canary_report.py deploy/canary/logs/canary_access.log
# next stage: edit the weights in .env, then apply without restarting the APIs:
docker compose -f deploy/canary/docker-compose.canary.yml --env-file deploy/canary/.env up -d nginx
```

**Rollback:** set `CANARY_STATE=down` in `.env` and re-run the last command.
nginx then stops routing to the canary. The canary container keeps running
so its logs can be inspected. If the canary crashes outright, `max_fails=3`
takes it out of rotation automatically for 30 s at a time, which limits
users to about 3 failed requests per window until you roll back.

**Resource note:** each API container loads Qwen3-1.7B and BGE-M3 (about
6 GB of RAM), so the full stack needs roughly twice the memory of the
single-container setup.

## Status

- [x] Data extraction: 1149/1149 articles, fully validated
- [x] Corpus validation: automated pytest gate, wired into DVC
- [x] Chunking: article-level, paragraph-split for long articles,
      repealed-range deduplication -- 1102 chunks
- [x] Embedding model selection: BGE-M3 (see docs/decisions.md)
- [x] Vector database: Qdrant, 1149 articles -> 2195 indexed points
- [x] Query engine (`src/egyptian_civil_code_rag/query.py`): retrieval +
      dedup + citation-only sources, validated end-to-end on real
      queries (force-majeure article, repealed-range status check)
- [x] FastAPI `/ask` + `/health` endpoints (`src/egyptian_civil_code_rag/api.py`):
      Pydantic-validated request (empty/whitespace/missing question -> 422),
      testable via injected fake engine with no model loading (tests/test_api.py)
- [x] Docker: built, verified live (retrieval, generation, citation,
      the exact-article-lookup fix -- all confirmed through the real
      running container, not just unit tests)
- [x] GitHub Actions CI: `lint` -> `test` -> `rebuild_index` (full
      `dvc repro` from source) -> `build_and_push_image` (to GHCR),
      all real and passing -- DVC remote is Google Drive (personal
      OAuth, not a service account -- see docs/decisions.md), CI
      authenticates via three repo secrets reconstructing the same
      local credential setup
- [ ] Generative model for serving: Qwen3-8B via vLLM (Qwen3-1.7B via
      `transformers` used for CPU-based development so far)
- [x] RAGAS evaluation harness: built, integrated, verified
      structurally correct (real retrieval + generation, correct
      MLflow logging path) -- full-scale scoring deferred until
      GPU/stronger judge available, see docs/decisions.md for the
      specific failure (chat-template mismatch + judge capability
      limits, not a config bug) and the 27m55s data point that
      informed the decision to stop iterating on CPU
- [x] 54-question evaluation set (`tests/eval/eval_questions.json`),
      stratified across substantive questions, direct article lookups,
      repealed-status checks (including individual articles inside a
      repealed range, not just the range itself), and out-of-corpus
      edge cases
- [x] Batch re-indexing (`scripts/reindex_batch.py`): adds or updates a
      document in the live index without re-embedding the Civil Code;
      `dvc repro` rebuilds the same result from `data/documents/`. See
      docs/decisions.md for the ID-collision and stale-chunk issues it
      had to handle
- [x] PII guardrails on `/ask` (`src/egyptian_civil_code_rag/pii.py`):
      Egyptian national ID / mobile / IBAN / card / email, Arabic-Indic
      digits, redacted in both the question and the answer; response lists
      `pii_redacted`. See docs/decisions.md
- [x] Canary rollout (`deploy/canary/`): nginx weighted split between two
      CI-built images, rollout stages + measurable promotion gates
      (`canary_report.py`), rollback by config -- see "Canary rollout" above
- [ ] MLflow experiment tracking (chunking/embedding parameter sweeps)
- [ ] BentoML serving
- [ ] Langfuse observability
