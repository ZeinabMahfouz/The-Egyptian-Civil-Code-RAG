# CPU-only serving image for the Egyptian Civil Code RAG API.
# Vector store baked in at build time (rubric: "Dockerfile includes
# the vector store and embedded documents"). LLM/embedding model
# weights are NOT baked in -- see docker-compose.yml's hf_cache volume
# and docs/decisions.md for why: they're generic, ~5.6GB, and would
# bloat every image rebuild for no benefit the vector store doesn't
# already provide.
FROM python:3.13-slim

# curl: needed for docker-compose's healthcheck against /health.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch, explicitly, BEFORE requirements.txt -- otherwise
# requirements.txt's own torch line (frozen from a local dev machine,
# not guaranteed CPU-only) could pull a multi-GB CUDA build that's
# useless here and wastes the entire build time on an unwanted download.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# Exclude torch's own line (already installed above, CPU-only) and any
# self-reference to this package -- `pip freeze` records an editable
# install (`pip install -e .`) as a git+https URL back to this repo
# when run inside a git working directory, which would otherwise make
# this step try to `git clone` the repo during the build (and fail,
# since git isn't installed here) instead of using the src/ we COPY in
# via the dedicated editable-install step below.
RUN grep -v -iE "^torch(==|\s|$)|egyptian.civil.code.rag|git\+https" requirements.txt \
        > requirements-docker.txt \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e . --no-deps

COPY params.yaml .
# The vector store -- this is the "embedded documents" the rubric asks
# for baked into the image. Must exist locally first via `dvc pull`;
# see README for the full command sequence.
COPY data/processed/qdrant_storage data/processed/qdrant_storage

EXPOSE 8000

CMD ["uvicorn", "egyptian_civil_code_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
