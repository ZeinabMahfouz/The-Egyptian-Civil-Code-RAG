"""GPU evaluation: RAGAS scoring, the MLflow chunking sweep, and registry promotion.

Designed for a Kaggle T4 x2 notebook (see notebooks/kaggle_gpu_eval.ipynb),
with Qwen3-8B served by vLLM as both the generator and the RAGAS judge:

    # 1. sweep: build one index per chunking config, score each on the
    #    20-question CI subset, log every run to MLflow
    python scripts/gpu_eval.py sweep --vllm-url http://localhost:8000/v1

    # 2. full: score the best config on all 54 questions (all 4 metrics),
    #    register it in the MLflow Model Registry with alias "production"
    python scripts/gpu_eval.py full --vllm-url http://localhost:8000/v1

    # dry run anywhere (no GPU, no judge): index + answers + article hit
    # rate + MLflow logging, RAGAS skipped
    python scripts/gpu_eval.py sweep --skip-ragas --fake-generator

Every run logs: chunking params (paragraph_split_threshold, dedupe, overlap=0),
embedding model, generator, judge, the four RAGAS means, and article_hit_rate
-- a judge-free retrieval metric: the share of questions whose expected
article appears in the cited sources.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import mlflow  # noqa: E402
import yaml  # noqa: E402
from chunk_corpus import build_chunks  # noqa: E402
from documents import chunk_document, list_document_paths, load_document  # noqa: E402
from embed_and_index import build_points  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import Distance, VectorParams  # noqa: E402

from egyptian_civil_code_rag.query import RAGQueryEngine, format_context  # noqa: E402
from egyptian_civil_code_rag.refusal import is_in_corpus, is_refusal  # noqa: E402

RAGAS_METRICS = ("faithfulness", "context_precision", "context_recall", "answer_relevancy")
REGISTERED_MODEL = "civil-code-rag-chunking"


@dataclass(frozen=True)
class ChunkConfig:
    name: str
    paragraph_split_threshold: int
    dedupe_repealed: bool = True


# The sweep. "baseline" is exactly what params.yaml / dvc repro builds today.
# A threshold above the longest article (~6000 chars) means "never split".
SWEEP = [
    ChunkConfig("baseline-700", 700),
    ChunkConfig("split-400", 400),
    ChunkConfig("split-1200", 1200),
    ChunkConfig("whole-articles", 100_000),
    ChunkConfig("no-repealed-dedupe", 700, dedupe_repealed=False),
]


# --- indexing ---------------------------------------------------------------


def build_index(cfg: ChunkConfig, corpus: list, embed_model, out_dir: Path, batch_size=16):
    """Same chunking + embedding code as the DVC pipeline, one Qdrant
    collection per config. Returns (QdrantClient, collection, n_chunks)."""
    chunks = [
        asdict(c) for c in build_chunks(corpus, cfg.paragraph_split_threshold, cfg.dedupe_repealed)
    ]
    points = list(build_points(chunks, embed_model, batch_size))
    for path in list_document_paths(REPO / "data" / "documents"):
        doc = load_document(path)
        doc_chunks = chunk_document(doc, cfg.paragraph_split_threshold, cfg.dedupe_repealed)
        points += list(build_points(doc_chunks, embed_model, batch_size, doc["doc_id"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(out_dir / cfg.name))
    collection = "eval"
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config=VectorParams(
            size=embed_model.get_embedding_dimension(), distance=Distance.COSINE
        ),
    )
    client.upsert(collection, points=points)
    return client, collection, len(chunks)


# --- answering ----------------------------------------------------------------


def make_row(q: dict, hits, answer: str) -> dict:
    """One evaluation row: what RAGAS reads (user_input, retrieved_contexts,
    response, reference) plus underscore-prefixed bookkeeping it ignores."""
    # The judge sees each article exactly as the generator did (citation,
    # repeal status, text) -- see format_context.
    cited = {n for h in hits for n in h.payload["article_numbers"]}
    expected = set(q.get("expected_articles") or [])
    return {
        "user_input": q["question"],
        "retrieved_contexts": [format_context(h.payload) for h in hits],
        "response": answer,
        "reference": q["ground_truth"],
        "_id": q["id"],
        "_category": q["category"],
        "_language": q["language"],
        "_sources": [h.payload["citation"] for h in hits],
        "_expected_articles": sorted(expected),
        # None for out-of-corpus questions (nothing to hit)
        "_article_hit": (bool(expected & cited) if expected else None),
    }


NO_CONTEXT_ANSWER = "No relevant articles found."


def answer_questions(engine: RAGQueryEngine, questions: list, workers: int = 4) -> list:
    """Real retrieval + generation for every question. Threads overlap the
    vLLM calls (the server batches them); retrieval is cheap."""

    def one(q):
        hits = engine.retrieve(q["question"])
        if hits:
            answer = engine.generate_fn(engine.build_prompt(q["question"], hits)).strip()
        else:
            answer = NO_CONTEXT_ANSWER
        return make_row(q, hits, answer)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, questions))


def article_hit_rate(rows: list) -> float | None:
    hits = [r["_article_hit"] for r in rows if r["_article_hit"] is not None]
    return sum(hits) / len(hits) if hits else None


# --- RAGAS ----------------------------------------------------------------------


def _stub_vertexai():
    """ragas 0.4.3 imports langchain_community.chat_models.vertexai, which
    no longer exists -- see run_ragas_eval.py for the full story."""
    name = "langchain_community.chat_models.vertexai"
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[name] = stub


def score_with_ragas(
    rows, vllm_url, judge_model, embedding_model, metrics=RAGAS_METRICS, workers=8
):
    """Judge = the vLLM-served model through its OpenAI-compatible API, with
    Qwen3's chat template applied server-side and thinking off. This is the
    fix for the CPU failure (bare HuggingFacePipeline, no chat template,
    unparseable output) documented in docs/decisions.md."""
    _stub_vertexai()
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    from ragas.run_config import RunConfig

    judge = LangchainLLMWrapper(
        ChatOpenAI(
            base_url=vllm_url,
            api_key="EMPTY",
            model=judge_model,
            temperature=0,
            max_tokens=1024,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    )
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=embedding_model))
    available = {
        "faithfulness": Faithfulness(llm=judge),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
    }
    dataset = EvaluationDataset.from_list(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    )
    result = evaluate(
        dataset=dataset,
        metrics=[available[m] for m in metrics],
        run_config=RunConfig(timeout=600, max_workers=workers, max_retries=3),
    )
    df = result.to_pandas()
    for i, r in enumerate(rows):
        for m in metrics:
            v = df[m].iloc[i]
            r[m] = None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
    return rows


def mean_scores(rows, metrics=RAGAS_METRICS) -> dict:
    """RAGAS means over the *in-corpus* questions only. For an out-of-corpus
    question ("what does criminal law say about theft?") the right answer is
    to decline: there is no reference answer to recall and no relevant
    context to be precise about, and RAGAS scores a correct refusal ~0 on
    answer_relevancy (it reads as "noncommittal"). Averaging those in
    measured the metric's blind spot, not the system. They're scored by
    refusal_rate instead -- and false_refusal_rate keeps the system honest
    in the other direction: declining in-corpus questions is a failure too."""
    in_corpus = [r for r in rows if is_in_corpus(r)]
    out_corpus = [r for r in rows if not is_in_corpus(r)]
    out = {}
    for m in metrics:
        vals = [r[m] for r in in_corpus if r.get(m) is not None]
        out[m] = sum(vals) / len(vals) if vals else None
        out[f"{m}_n_scored"] = len(vals)  # how many the judge actually managed to score
    out["refusal_rate"] = (
        sum(is_refusal(r["response"]) for r in out_corpus) / len(out_corpus) if out_corpus else None
    )
    out["false_refusal_rate"] = (
        sum(is_refusal(r["response"]) for r in in_corpus) / len(in_corpus) if in_corpus else None
    )
    return out


# --- MLflow ---------------------------------------------------------------------


class ChunkingConfigModel(mlflow.pyfunc.PythonModel):
    """What goes in the Model Registry: the winning chunking/embedding
    configuration (a params.yaml), versioned like a model. predict() returns
    it, so `mlflow.pyfunc.load_model("models:/...@production")` gives the
    serving side the exact config that was evaluated."""

    def load_context(self, context):
        with open(context.artifacts["params"], encoding="utf-8") as f:
            self.params = yaml.safe_load(f)

    def predict(self, context, model_input, params=None):
        return [self.params]


def params_yaml_for(cfg: ChunkConfig, embedding_model: str) -> dict:
    with open(REPO / "params.yaml", encoding="utf-8") as f:
        p = yaml.safe_load(f)
    p["chunking"]["paragraph_split_threshold_chars"] = cfg.paragraph_split_threshold
    p["chunking"]["dedupe_repealed_ranges"] = cfg.dedupe_repealed
    p["embedding"]["model_name"] = embedding_model
    return p


def run_config(cfg, args, corpus, questions, embed_model, generate_fn, stage: str) -> dict:
    t0 = time.time()
    client, collection, n_chunks = build_index(
        cfg, corpus, embed_model, Path(args.work_dir) / "indexes", batch_size=args.embed_batch
    )
    engine = RAGQueryEngine(
        params_path=REPO / "params.yaml",
        generate_fn=generate_fn,
        embed_model=embed_model,
        client=client,
        collection=collection,
    )
    rows = answer_questions(engine, questions, workers=args.workers)
    if not args.skip_ragas:
        rows = score_with_ragas(
            rows, args.vllm_url, args.judge_model, args.embedding_model, workers=args.judge_workers
        )
    scores = mean_scores(rows, RAGAS_METRICS if not args.skip_ragas else ())
    scores["article_hit_rate"] = article_hit_rate(rows)
    client.close()

    out = Path(args.work_dir) / "reports" / f"ragas_{stage}_{cfg.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with mlflow.start_run(run_name=f"{stage}-{cfg.name}") as run:
        mlflow.log_params(
            {
                "config": cfg.name,
                "chunk_size": cfg.paragraph_split_threshold,  # paragraph-split threshold (chars)
                "overlap": 0,  # article-level chunks never overlap -- logged explicitly
                "dedupe_repealed": cfg.dedupe_repealed,
                "n_chunks": n_chunks,
                "embedding_model": args.embedding_model,
                "generator": args.gen_model,
                "judge": "none" if args.skip_ragas else args.judge_model,
                "n_questions": len(questions),
                "question_set": stage,
            }
        )
        mlflow.log_metrics({k: v for k, v in scores.items() if v is not None})
        mlflow.log_metric("eval_seconds", time.time() - t0)
        mlflow.log_artifact(str(out))
        pfile = Path(tempfile.mkdtemp()) / "params.yaml"
        pfile.write_text(
            yaml.safe_dump(params_yaml_for(cfg, args.embedding_model)), encoding="utf-8"
        )
        mlflow.log_artifact(str(pfile))
        run_id = run.info.run_id

    print(
        f"[{stage}] {cfg.name}: "
        + json.dumps({k: v for k, v in scores.items() if not k.endswith("_n_scored")})
    )
    return {"config": cfg, "scores": scores, "run_id": run_id, "rows": rows, "report": out}


# A 20-question subset can't resolve small differences: one answer scored
# differently moves mean faithfulness by ~0.03-0.05. A challenger has to beat
# the current production config (SWEEP[0], what params.yaml builds today) by
# more than that, or production stays as is -- re-chunking, re-indexing and
# re-deploying for noise is cost and risk with no benefit.
MIN_FAITHFULNESS_GAIN = 0.05


def best_of(results: list, min_gain: float = MIN_FAITHFULNESS_GAIN) -> dict:
    """Highest faithfulness wins (hallucination is the failure that matters
    for legal answers); context_precision, then article_hit_rate break ties.
    Without RAGAS (dry run), article_hit_rate alone decides. results[0] is the
    incumbent: it is kept unless the winner beats its faithfulness by min_gain."""

    def key(r):
        s = r["scores"]
        return tuple(
            s.get(m) if s.get(m) is not None else -1
            for m in ("faithfulness", "context_precision", "article_hit_rate")
        )

    winner = max(results, key=key)
    incumbent = results[0]
    f_win = winner["scores"].get("faithfulness")
    f_inc = incumbent["scores"].get("faithfulness")
    if f_win is not None and f_inc is not None and f_win - f_inc < min_gain:
        return incumbent
    return winner


def register_and_promote(result: dict, embedding_model: str) -> str:
    cfg = result["config"]
    pfile = Path(tempfile.mkdtemp()) / "params.yaml"
    pfile.write_text(yaml.safe_dump(params_yaml_for(cfg, embedding_model)), encoding="utf-8")
    with mlflow.start_run(run_id=result["run_id"]):
        info = mlflow.pyfunc.log_model(
            name="chunking_config",
            python_model=ChunkingConfigModel(),
            artifacts={"params": str(pfile)},
        )
    mv = mlflow.register_model(info.model_uri, REGISTERED_MODEL)
    client = mlflow.MlflowClient()
    # MLflow 3 replaced stages with aliases; "production" is the alias the
    # serving side resolves (models:/civil-code-rag-chunking@production).
    client.set_registered_model_alias(REGISTERED_MODEL, "production", mv.version)
    client.set_model_version_tag(REGISTERED_MODEL, mv.version, "config", cfg.name)
    for k, v in result["scores"].items():
        if v is not None and not k.endswith("_n_scored"):
            client.set_model_version_tag(REGISTERED_MODEL, mv.version, k, f"{v:.4f}")
    return mv.version


# --- entry point ----------------------------------------------------------------


def load_questions(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def make_generator(args):
    if args.fake_generator:

        def gen(prompt):
            return "Dry-run answer (no LLM)."

        gen.model_name = "fake"
        return gen
    from egyptian_civil_code_rag.backends import openai_backend

    return openai_backend(args.vllm_url, args.gen_model)


def main(argv=None, embed_model=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=["sweep", "full"])
    ap.add_argument("--corpus", type=Path, default=REPO / "data" / "interim" / "civil_code.json")
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1")
    ap.add_argument("--gen-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--judge-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--embedding-model", default="BAAI/bge-m3")
    ap.add_argument("--work-dir", default="eval_out")
    ap.add_argument("--mlflow-uri", default=None, help="default: sqlite:///<work-dir>/mlflow.db")
    ap.add_argument("--experiment", default="civil-code-rag-chunking")
    ap.add_argument(
        "--config", default=None, help="full mode: config name (default: best from sweep)"
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--embed-device",
        default=None,
        help="e.g. cuda:1 -- keep BGE-M3 off the GPU vLLM is fullest on",
    )
    ap.add_argument("--embed-batch", type=int, default=16)
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--skip-ragas", action="store_true", help="dry run: no judge")
    ap.add_argument("--fake-generator", action="store_true", help="dry run: no LLM")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(
        args.mlflow_uri or f"sqlite:///{Path(args.work_dir).resolve()}/mlflow.db"
    )
    mlflow.set_experiment(args.experiment)

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    from sentence_transformers import SentenceTransformer

    embed_model = embed_model or SentenceTransformer(args.embedding_model, device=args.embed_device)
    generate_fn = make_generator(args)

    if args.mode == "sweep":
        questions = load_questions(REPO / "tests" / "eval" / "eval_questions_ci_subset.json")[
            : args.limit
        ]
        results = [
            run_config(c, args, corpus, questions, embed_model, generate_fn, "sweep") for c in SWEEP
        ]
        best = best_of(results)
        summary = {
            "best": best["config"].name,
            "runs": {r["config"].name: r["scores"] for r in results},
        }
        (Path(args.work_dir) / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[sweep] best config: {best['config'].name}")
        return summary

    # full: all 54 questions on one config, then registry promotion
    if args.config:
        cfg = next(c for c in SWEEP if c.name == args.config)
    else:
        summary_path = Path(args.work_dir) / "sweep_summary.json"
        cfg = next(c for c in SWEEP if c.name == json.loads(summary_path.read_text())["best"])
    questions = load_questions(REPO / "tests" / "eval" / "eval_questions.json")[: args.limit]
    result = run_config(cfg, args, corpus, questions, embed_model, generate_fn, "full")
    version = register_and_promote(result, args.embedding_model)
    # reports/ragas_results.json: what the API's faithfulness gauge, the CI
    # gate and the README read. The meta block pins *what* was evaluated, so
    # the CI gate can tell when the evaluation no longer matches the repo.
    final = Path(args.work_dir) / "reports" / "ragas_results.json"
    report = {
        "meta": {
            "config": cfg.name,
            "params": params_yaml_for(cfg, args.embedding_model),
            "corpus_md5": hashlib.md5(args.corpus.read_bytes()).hexdigest(),
            "generator": args.gen_model,
            "judge": args.judge_model,
            "n_questions": len(questions),
            "registered": f"{REGISTERED_MODEL} v{version} @production",
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "aggregate": result["scores"],
        "rows": result["rows"],
    }
    final.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[full] {cfg.name} registered as {REGISTERED_MODEL} v{version} @production")
    return {"config": cfg.name, "scores": result["scores"], "version": version}


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
