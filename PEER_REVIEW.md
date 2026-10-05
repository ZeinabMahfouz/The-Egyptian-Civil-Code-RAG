# Peer review guide

Thank you for reviewing this project. This page tells you how to run it,
where to find the evidence for each rubric point, and how to submit your
review. You should not need to ask me anything. If you do, that is a
finding worth writing down.

**The project in one paragraph.** A question-answering system over the
Egyptian Civil Code (1149 articles, Arabic and English). You ask a question;
it retrieves the relevant articles and answers *only* from them, with
article citations, because a hallucinated legal answer is unacceptable. It
declines when the code doesn't cover the question. Around it: DVC data
pipeline, MLflow experiments and registry, CI with a RAGAS quality gate,
PII redaction, Langfuse tracing, Prometheus/Grafana monitoring, BentoML,
vLLM, and AWQ 4-bit quantization.

## Time budget

| Part | What | Time | Needs |
|---|---|---|---|
| **A** | Run the API with Docker, ask questions | 20–30 min (mostly one-time download) | Docker |
| **B** | Read the evidence for each rubric point | 15–20 min | a browser |
| **C** | Optional: install for development, run the tests | 15 min | Python 3.11+ |
| **D** | Write the review (template at the end) | 30 min | |

---

## Part A: run it (3 commands)

**You need:** Docker Desktop running, about **8 GB of RAM** available to
Docker (Docker Desktop → Settings → Resources), about **12 GB of free disk**,
and internet. You don't need a GPU or any account.

```bash
git clone https://github.com/ZeinabMahfouz/The-Egyptian-Civil-Code-RAG.git && cd The-Egyptian-Civil-Code-RAG
docker compose up -d
curl http://localhost:8000/health
```

What happens:

- `docker compose up` pulls the pre-built image from the GitHub Container
  Registry. The vector index of all 1149 articles is already baked in, so
  you need no data download and no DVC access.
- On first start the container downloads the models (about 5.6 GB:
  Qwen3-1.7B and BGE-M3) into a Docker volume. This takes **5–15 minutes**
  depending on your connection, and only happens once.
- `/health` fails until the models are loaded. Repeat it every minute, or
  watch progress with `docker compose logs -f api`. It's ready when you get:

```json
{"status":"healthy","documents_indexed":2197,"release":"dev"}
```

Two lines in the startup log look like errors but are expected:
`Langfuse client initialized without public_key` (tracing is off when no
Langfuse server is configured) and `unauthenticated requests to the HF Hub`
(the model download works without a token). The models then download with no
progress output, so a few quiet minutes are normal; wait for
`Application startup complete`.

### Ask it questions

**The easiest way:** open **http://localhost:8000/docs** in a browser, then
**POST /ask** → **Try it out** → replace the body → **Execute**.

**Or with curl:**

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What does Article 147 say?"}'
```

The Docker image runs on CPU, so **each answer takes about 30–90 seconds**.
That's expected (see "What runs where" below).

Questions worth trying, and what a correct answer looks like:

| Try | Question | What to check |
|---|---|---|
| Article lookup | `What does Article 147 say?` | "The contract makes the law of the parties"; `sources` contains Article 147 |
| Arabic | `ما حكم المادة 147 من القانون المدني؟` | An Arabic answer starting `العقد شريعة المتعاقدين` |
| Meaning question | `Can a judge reduce an obligation if unforeseen circumstances make it too burdensome?` | Cites Article 147 (paragraph 2, *imprévision*) |
| Repealed article | `Is Article 400 still in force?` | Says it is **repealed** (Articles 389–417 were repealed) |
| Out of scope | `What does Egyptian criminal law say about theft?` | **Declines**: the Civil Code doesn't cover criminal law |
| PII redaction | `My national ID is 29801011234567, can I cancel my contract?` | `"pii_redacted": ["EG_NATIONAL_ID"]`; the number never appears in the answer |
| Validation | `{"question": "   "}` | HTTP **422**, rejected by Pydantic |

**Streaming** (tokens arrive progressively; `-N` turns off curl's buffering):

```bash
curl -N -X POST http://localhost:8000/ask/stream -H "Content-Type: application/json" \
     -d '{"question": "What does Article 148 say?"}'
```

**Metrics** (Prometheus format: request counts, latency histogram, tokens,
PII redactions, RAGAS faithfulness):

```bash
curl -s http://localhost:8000/metrics | grep "^rag_"
```

**When you're done:** `docker compose down`. To also delete the downloaded
models: `docker compose down -v`.

### If something goes wrong

| Symptom | Fix |
|---|---|
| `curl: (56) Connection reset by peer`, or `docker compose ps` shows `(unhealthy)` | Still starting: the models are downloading. Wait for `Application startup complete` in `docker compose logs -f api`; the status turns `(healthy)` by itself |
| `port is already allocated` (8000) | Stop whatever uses port 8000, or change `"8000:8000"` to `"8001:8000"` in `docker-compose.yml` and use port 8001 |
| Container restarts or is `Killed` | Docker needs more memory: set at least 8 GB in Docker Desktop → Settings → Resources |
| `/health` still failing after 20 min | `docker compose logs api` and look at the last lines (usually a model download that stalled; `docker compose restart api` resumes it) |
| `pull access denied` | Build locally instead. That needs DVC access to the index, so please note it in your review: it means the image isn't public yet |
| Apple Silicon Mac | The image is `linux/amd64`; Docker runs it through emulation, which works but is slower |

### What runs where (so the numbers make sense)

| | Model | Hardware | Where the results are |
|---|---|---|---|
| This Docker image | Qwen3-1.7B, transformers | CPU | what you just ran |
| Evaluation | **Qwen3-8B on vLLM** (generator and RAGAS judge) | 2× T4 GPU on Kaggle | `reports/ragas_results.json` |
| Quantization | Qwen3-8B fp16 vs **AWQ 4-bit** on vLLM | 2× T4 GPU on Kaggle | `reports/quantization.md` |

The API switches to a vLLM server when `VLLM_BASE_URL` and `GEN_MODEL` are
set (`src/egyptian_civil_code_rag/backends.py`). Without a GPU, it falls back
to the CPU model, which is why you get CPU speeds here. The GPU runs are
reproducible from the two notebooks in `notebooks/`.

---

## Part B: rubric evidence map

Each row: what the rubric asks for, where to look, and the quickest way to
check it.

| # | Rubric point | Where to look | Quick check |
|---|---|---|---|
| 1 | Code and packaging | `pyproject.toml`, `src/egyptian_civil_code_rag/` | Part C: `pip install -e .` |
| 2 | API: Pydantic, async | `src/egyptian_civil_code_rag/api.py` (`AskRequest`, `/ask`, `/ask/stream`, `/health`) | Part A: the empty question returns 422 |
| 3 | Docker, 3-command README | `Dockerfile`, `docker-compose.yml` | Part A |
| 4 | MLflow: ≥5 runs, best model promoted | `reports/mlflow_comparison.png` (5 chunking configs and the full run), `reports/mlflow_registry.png`, `scripts/gpu_eval.py` | Registered model `civil-code-rag-chunking` v1, alias `@production`; the selection rule is in `best_of()` |
| 5 | DVC: `dvc repro` reproduces | `dvc.yaml` (extract → validate → chunk → embed), `dvc.lock` | The data remote is a private Google Drive, so you can't `dvc pull`. Instead, open the latest green CI run: the **rebuild_index** job runs `dvc pull` + `dvc repro` + `dvc status` from scratch |
| 6 | CI/CD with a quality gate | `.github/workflows/ci.yml`, **Actions** tab | lint → test → rebuild_index → build_and_push_image, plus **ragas_gate** (fails if faithfulness < 0.75, or if the evaluation is stale against the corpus or params) |
| 7 | Production serving | BentoML: `src/egyptian_civil_code_rag/service.py`; vLLM; latency p50/p95 in `reports/quantization.md`; canary rollout in `deploy/canary/` | Locust at 50 users: `reports/locust_u50.html`, summary in `reports/load_test.md` (0 failures, p95 12.0 s) |
| 8 | Monitoring | `reports/grafana_dashboard.png`, `deploy/monitoring/` (alert rules in `prometheus/alerts.yml`: faithfulness < 0.80, query drift; delivered through Alertmanager to a webhook, demo in README "Monitoring"), `reports/drift.md`, `reports/langfuse_trace.png` | Part A: `/metrics` |
| 9 | Peer review | this page | |
| 10 | README and architecture | `README.md` (results, quick start, architecture diagram `docs/architecture.svg`, changelog), `docs/decisions.md` | Could you run it without asking? |

**Key results** (Qwen3-8B, 54 questions, `docs/decisions.md` has the details):

- RAGAS faithfulness **0.896**, context precision 0.908, context recall 0.854
- Load test, 50 concurrent users: 1,692 requests, 0 failures, 5.65 req/s,
  p95 12.0 s
- Query drift: questions from other jurisdictions and off-topic ones flagged
  (16 of 16 below the cut-off); new Civil Code questions not flagged
- Out-of-corpus questions declined: 4 of 6 (the main remaining risk)
- AWQ 4-bit vs fp16: faithfulness 0.888 vs 0.886 (no loss); latency p50
  1.06 s vs 2.81 s; weights 2.85 vs 7.64 GiB per GPU

**Worth reading if you have 10 minutes:** the last three sections of
`docs/decisions.md`. The first GPU evaluation scored 0.57; reading the
per-question results found a data bug (live articles flagged as repealed)
and two measurement errors. Fixing them, not tuning, is what moved it to
0.896.

### Known gaps (stated up front)

These are the things I know are missing or weaker. Please don't spend time
discovering them; spend it on what I *haven't* noticed.

- The Docker image runs a CPU model (Qwen3-1.7B), not the evaluated
  Qwen3-8B. GPU serving was measured on Kaggle, not packaged as a
  container.
- The DVC remote is a personal Google Drive, so reviewers can't `dvc pull`.
  CI proves reproducibility instead. The Drive login CI uses expires every
  7 days (the Google app is in Testing mode), so a CI run can fail at
  `dvc pull` until I renew it. That is a credentials problem, not a code
  one.
- Alerts go to a local webhook that logs them, not to email or Slack.
- 2 of 6 out-of-corpus questions were answered instead of declined.
- The generator and the RAGAS judge are the same model family (Qwen3-8B).
- Under 50 users, retrieval takes 38% of request time (2.45 s on average);
  the likely cause (local Qdrant client and embedder serializing inside the
  API process) is in `docs/decisions.md`. Not fixed yet.
- The drift check runs on hand-written query windows that simulate drift,
  not on logged production traffic.

---

## Part C: optional, development install and tests

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
ruff check . && ruff format --check .
pytest -q --ignore=tests/test_corpus_validation.py
```

You should see 110+ tests passing. `test_corpus_validation.py` checks the
real extracted corpus, which comes from DVC; in CI it runs after `dvc pull`.
Everything else uses fakes (no model downloads, no GPU, no server):

- `tests/test_api.py`: the API with a fake engine, including the 422 on an
  empty question
- `tests/test_streaming.py`: PII redaction while streaming, even when a phone
  number is split across tokens
- `tests/test_ragas_gate.py`: the CI gate, including the staleness checks
- `tests/test_gpu_eval.py`: the MLflow sweep and registry, end to end in
  dry-run mode

---

## Part D: submitting the review

Per the course instructions: post the review as a **GitHub Issue** on this
repo (at least **300 words**, all 5 sections), and send a PDF copy to the
instructor.

**Issues** → **New issue** → choose **Peer review**. The template has the
five sections:

1. **Setup:** did it run? How long did it take? What broke or was unclear?
2. **Code quality:** two specific patterns, one good and one to refactor
   (file and function names help).
3. **One genuine strength:** specific, and why it matters in production.
4. **Two specific improvements:** concrete enough to act on immediately.
5. **Extension idea:** what, why, and how, in one paragraph.

I'll reply to each section within 3 days: either implementing the
improvement or explaining why I disagree.
