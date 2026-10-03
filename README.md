# The Egyptian Civil Code RAG

> **Reviewing this project?** Start with [PEER_REVIEW.md](PEER_REVIEW.md): it
> covers how to run it in 3 commands, where the evidence for each rubric point
> is, and the review template.

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
3 commands, no DVC access needed (the image CI publishes from `main` has the
vector store baked in):

```bash
git clone https://github.com/ZeinabMahfouz/The-Egyptian-Civil-Code-RAG.git && cd The-Egyptian-Civil-Code-RAG
docker compose up -d
curl http://localhost:8000/health   # wait for {"status":"healthy",...}, then:
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What does Article 147 say?"}'
```

To build the image yourself instead: `dvc pull` (fetches the vector store
the Dockerfile bakes in), then `docker compose up --build`.

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

**Note on the generative model:** the Docker image runs Qwen3-1.7B on CPU
with `transformers`, so it works on any laptop (30-90 s per answer). With a
GPU, point it at a vLLM server instead: set `VLLM_BASE_URL` and `GEN_MODEL`
(e.g. `Qwen/Qwen3-8B-AWQ`). That is the configuration the RAGAS and
quantization results were measured on (Kaggle, 2x T4).

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

## Serving with BentoML

The BentoML service (`src/egyptian_civil_code_rag/service.py`) wraps the same
RAG engine and PII guard behind an async `/ask`. It runs in its own
environment, separate from the pipeline's (see docs/decisions.md):

```bash
python3 -m venv .venv-serve && source .venv-serve/bin/activate
pip install -r requirements-serve.txt && pip install -e . --no-deps
bentoml serve egyptian_civil_code_rag.service:CivilCodeRAG   # stop any other API first
```

Open http://localhost:3000 for the interactive API page, or:

```bash
curl -X POST http://localhost:3000/ask -H "Content-Type: application/json" \
     -d '{"question": "What does Article 147 say?"}'
curl -X POST http://localhost:3000/health
```

## Streaming answers

Both servers can stream the answer as it's generated instead of waiting a
minute or more for the whole thing:

```bash
# FastAPI: Server-Sent Events -- token events, then a final "done" event with sources
curl -N -X POST http://localhost:8000/ask/stream -H "Content-Type: application/json" \
     -d '{"question": "What does Article 147 say?"}'

# BentoML: plain text chunks, citations on the last line
curl -N -X POST http://localhost:3000/ask_stream -H "Content-Type: application/json" \
     -d '{"question": "What does Article 147 say?"}'
```

PII is redacted in the stream too, even when a number is split across
tokens (see docs/decisions.md). The text therefore arrives about 64
characters behind the model.

## Observability (Langfuse)

Every `/ask` is traced to a self-hosted Langfuse instance when its keys are set:

```bash
# Langfuse itself: official compose file, outside this repo
git clone https://github.com/langfuse/langfuse.git ~/langfuse && cd ~/langfuse
docker compose up -d                       # UI at http://localhost:3000

# this repo: put the project's API keys in .env (git-ignored)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000

set -a; source .env; set +a
uvicorn egyptian_civil_code_rag.api:app --port 8000
```

![Langfuse trace of one /ask request](reports/langfuse_trace.png)

## Monitoring (Prometheus + Grafana)

The FastAPI app exposes Prometheus metrics at `GET /metrics`: request count
by status, latency per stage (total / retrieve / generate), LLM tokens in and
out, PII redactions by entity type, best-chunk retrieval similarity, and the
latest RAGAS faithfulness. `deploy/monitoring/` runs Prometheus and a
pre-provisioned Grafana dashboard against it:

```bash
uvicorn egyptian_civil_code_rag.api:app --host 0.0.0.0 --port 8000   # the API, on the host
docker compose -f deploy/monitoring/docker-compose.monitoring.yml up -d
```

- Grafana: http://localhost:3001 (admin / admin), dashboard **Egyptian Civil Code RAG**
- Prometheus: http://localhost:19090 (`/targets` for the scrape, `/alerts` for the rules)

**Cost per hour** is `tokens per hour / 1000 x price per 1k tokens`. The price
is a dashboard variable (default $0.002) because a self-hosted model has no
per-token bill. Set it to a hosted-API equivalent or to your GPU cost per
1k tokens.

**Alerts** (`deploy/monitoring/prometheus/alerts.yml`):

| Alert | Condition |
|---|---|
| `RagFaithfulnessLow` | latest RAGAS faithfulness < 0.80 |
| `RagHighErrorRate` | > 5% of `/ask` requests failing for 5 min |
| `RagSlowP95` | p95 latency > 120 s for 5 min (CPU backend; tighten for vLLM) |
| `RagApiDown` | `/metrics` unreachable for 1 min |

The alerts fire in Prometheus (`/alerts`). Sending them to email or Slack would
add Alertmanager, which is not set up here.

![Grafana dashboard](reports/grafana_dashboard.png)

## GPU evaluation (Kaggle): vLLM, RAGAS, MLflow

The laptop has no GPU, so generation with Qwen3-8B, RAGAS scoring and the
MLflow chunking sweep run in `notebooks/kaggle_gpu_eval.ipynb` on Kaggle
(T4 x2):

- **vLLM** serves `Qwen/Qwen3-8B` (fp16, tensor-parallel 2) behind an
  OpenAI-compatible API, as both the generator and the RAGAS judge.
- **`scripts/gpu_eval.py sweep`**: 5 chunking configs (paragraph-split
  threshold 400 / 700 / 1200 / whole articles, repealed-range dedupe on/off),
  each scored on the 20-question CI subset and logged to MLflow.
- **`scripts/gpu_eval.py full`**: the best config on all 54 questions and all
  4 RAGAS metrics, registered as `civil-code-rag-chunking@production`.

The API can use the same vLLM server: set `VLLM_BASE_URL=http://<host>:8000/v1`
and `GEN_MODEL=Qwen/Qwen3-8B` before starting uvicorn or BentoML. Without
them, it falls back to the local CPU model.

**Results** (full run, 54 questions, Qwen3-8B as generator and judge):

| Faithfulness | Context precision | Context recall | Article hit rate | Out-of-corpus declined |
|---|---|---|---|---|
| **0.896** | 0.908 | 0.854 | 0.917 | 4 of 6 |

The sweep found no chunking config meaningfully better than the current
one (best challenger +0.044 faithfulness, below the 0.05 needed), so
`baseline-700` stayed in production. Full tables, the before/after of the
evaluation fixes, and known limits are in `docs/decisions.md`. MLflow
screenshots: `reports/mlflow_comparison.png`, `reports/mlflow_registry.png`.

**CI gate** (`ragas_gate` job): `scripts/ragas_gate.py` fails the build if
faithfulness on the 20-question CI subset is below 0.75, or if fewer than
75% of its out-of-corpus questions were declined. It also fails if the
committed evaluation is stale, meaning it was run against a different
corpus or different chunking/embedding params than the repo now has.
Current result: faithfulness 0.868, 3 of 4 declined, **PASS**.

## Quantization (AWQ 4-bit)

`Qwen/Qwen3-8B-AWQ` against fp16 on the same 54 questions, both judged by the
fp16 model (`scripts/quant_compare.py`, `notebooks/kaggle_quantization.ipynb`):

| | fp16 | AWQ 4-bit |
|---|---|---|
| Faithfulness | 0.886 | 0.888 (drop −0.002 < 0.03: **PASS**) |
| Latency p50 / p95 | 2.81 s / 9.42 s | 1.06 s / 2.51 s |
| Throughput, 8 concurrent users | 2.24 req/s | 4.21 req/s |
| Weights per GPU | 7.64 GiB | 2.85 GiB |

No measurable quality loss, ~2.7x lower latency, ~2.7x less weight memory,
so AWQ is the serving model. Full table and reasoning:
`reports/quantization.md`, `docs/decisions.md`.

## Load test (Locust, 50 users)

FastAPI + Qwen3-8B-AWQ on vLLM (2x T4), questions from the eval set, 0.5–2 s
think time (`load_test/locustfile.py`, `notebooks/kaggle_load_test.ipynb`):

| Users | Requests | Failures | Throughput | p50 | p95 | p99 |
|---|---|---|---|---|---|---|
| 1 | 47 | 0 | 0.40 req/s | 1.2 s | 2.2 s | 2.5 s |
| 50 | 1,692 | 0 | 5.65 req/s | 7.0 s | 12.0 s | 15.0 s |

No failures. It saturates at about 5.7 requests/s; under load, retrieval
takes 38% of the time, which is the first thing to fix. Reports:
`reports/locust_u50.html`, `reports/load_test.md`; analysis in
`docs/decisions.md`.

## Query drift

Are users asking what the system was evaluated on? `scripts/embedding_drift.py`
compares batches of queries with the eval set (BGE-M3). The alerting signal is
the share of questions with no close match among the indexed articles, with a
cut-off and limit calibrated on the eval questions:

| Window (16 questions each) | Below cut-off | Flagged |
|---|---|---|
| New Civil Code questions | 2 | no |
| Other jurisdictions (Saudi labour law, Egyptian criminal/tax law) | 16 | **yes** |
| Off topic | 16 | **yes** |

Plain centroid drift was tried first and rejected for alerting: it flagged
normal questions because it reacts to phrasing. Details in `docs/decisions.md`,
report in `reports/drift.md`, alert `RagQueryDrift`.

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
- [x] Generative model: Qwen3-8B via vLLM on Kaggle (2x T4); the API and
      BentoML use it when `VLLM_BASE_URL` is set, Qwen3-1.7B on CPU otherwise
- [x] RAGAS evaluation on GPU: 54 questions, faithfulness 0.896, judged by
      Qwen3-8B (the CPU attempt and why it failed are in docs/decisions.md)
- [x] MLflow: 5-config chunking sweep + full run, best config registered as
      `civil-code-rag-chunking@production`
- [x] CI RAGAS gate: faithfulness >= 0.75, out-of-corpus declined, and
      fails when the evaluation is stale against the corpus or params
- [x] Query drift check: off-corpus share with a calibrated limit, MLflow,
      Prometheus alert, Grafana panel
- [x] Locust load test: 50 users, 1,692 requests, 0 failures, p95 12.0 s
- [x] AWQ 4-bit quantization: no faithfulness loss (−0.002), 2.7x faster,
      2.7x less weight memory; logged to MLflow (`civil-code-rag-quantization`)
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
- [x] BentoML serving
- [x] Langfuse tracing (self-hosted): every `/ask` creates a trace with
      spans for PII check, retrieval, generation (with token usage) and
      output PII check; raw PII never enters a trace. See docs/decisions.md
- [x] Prometheus + Grafana: `/metrics` (requests, stage latency, tokens,
      PII, retrieval similarity, RAGAS faithfulness), provisioned dashboard
      with p95 latency and cost/hour, alert rules incl. faithfulness < 0.80
- [x] Arabic extraction fix: lam-alef ligatures (لا) were extracted
      flipped (ال) in ~5,000 places; now reversed per glyph, with a DVC
      validation gate. See docs/decisions.md
- [x] Streaming: `POST /ask/stream` (FastAPI, SSE) and `ask_stream`
      (BentoML); PII-safe incremental redaction; traced and measured
      (time-to-first-chunk metric) like `/ask`
