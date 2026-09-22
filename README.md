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

Python, pdfplumber, embeddings + vector DB *(TBD -- see roadmap)*, vLLM,
RAGAS, Langfuse, BentoML, MLflow, DVC, Docker, GitHub Actions.

## Project structure

```
data/
  raw/            source PDF, DVC-tracked
  interim/        pipeline intermediates (civil_code.json, chunks.json)
  processed/      (reserved for downstream stages -- embeddings, index)
scripts/          pipeline code only (extract_corpus.py, chunk_corpus.py,
                  topic_overrides.json, manual_patches.json)
diagnostics/      one-off debugging/analysis scripts, not part of the
                  pipeline -- documents how failures were found and fixed
tests/            pytest validation suite
notebooks/        exploratory work
src/              (reserved for application/serving code)
dvc.yaml          pipeline stage definitions
params.yaml       tunable pipeline parameters (chunking thresholds, etc.)
```

## Reproducing this

```bash
git clone <repo-url>
cd The-Egyptian-Civil-Code-RAG
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt install poppler-utils   # pdftotext/pdfinfo, required by extraction

dvc pull      # fetch the DVC-tracked raw PDF and pipeline outputs
dvc repro     # rebuild everything from source, verifying it reproduces
```

Configure your own DVC remote first if you're not pulling from the
project's existing one -- see `.dvc/config`.

## Status

- [x] Data extraction: 1149/1149 articles, fully validated
- [x] Corpus validation: automated pytest gate, wired into DVC
- [x] Chunking: article-level, paragraph-split for long articles,
      repealed-range deduplication -- 1102 chunks
- [ ] Embedding model selection (Arabic/bilingual-capable)
- [ ] Vector database setup and indexing
- [ ] Retrieval + generation pipeline (vLLM)
- [ ] RAGAS evaluation harness
- [ ] MLflow experiment tracking (chunking/embedding parameter sweeps)
- [ ] BentoML serving
- [ ] Langfuse observability
- [ ] Docker + GitHub Actions CI/CD