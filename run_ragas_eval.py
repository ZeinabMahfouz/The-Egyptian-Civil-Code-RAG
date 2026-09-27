#!/usr/bin/env python3
"""
RAGAS evaluation harness: runs tests/eval/eval_questions.json through the
real RAGQueryEngine, scores faithfulness / context_precision /
context_recall / answer_relevancy, and logs everything to MLflow.

KNOWN ENVIRONMENT ISSUE (as of ragas 0.4.3, checked directly): `import
ragas` fails with
    ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
because ragas/llms/base.py unconditionally imports ChatVertexAI, and
current langchain-community releases (which is being sunset upstream)
no longer ship that submodule. We don't use Vertex AI at all -- fix is
simply:
    pip install langchain-google-vertexai
which satisfies the import chain without our code ever touching it. If
this stops being necessary in a future ragas release, this comment (and
the corresponding requirements.txt line) can be removed.

Explicitly configures RAGAS with OUR self-hosted models (Qwen3 as judge,
BGE-M3 for the embedding-based metric) -- RAGAS defaults to OpenAI
otherwise, which would silently break (or silently call out to a paid
API) for a project built entirely around self-hosted models.

Usage:
    python run_ragas_eval.py --questions tests/eval/eval_questions.json \
        --params params.yaml --mlflow-experiment civil-code-rag-eval
"""

import argparse
import json
import sys
import types
from pathlib import Path

# WORKAROUND for a real bug in ragas 0.4.3, confirmed by direct testing:
# ragas/llms/base.py unconditionally imports ChatVertexAI from
# langchain_community.chat_models.vertexai at module load time, even
# though we never use Google Vertex AI. That submodule no longer exists
# in current langchain-community releases (which is being sunset
# upstream) -- pip-installing a different package can't resurrect a
# file that's been removed from another package entirely. Since we
# genuinely never touch VertexAI, a stub module satisfies the import
# without affecting anything this script actually uses. Must run BEFORE
# any ragas import, below.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class _StubChatVertexAI:  # never instantiated -- only needs to exist
        pass

    _stub.ChatVertexAI = _StubChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
    try:
        import langchain_community.chat_models as _cm

        _cm.vertexai = _stub
    except ImportError:
        pass

import mlflow
import yaml
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig
from transformers import pipeline as hf_pipeline

sys.path.insert(0, "src")
from egyptian_civil_code_rag.backends import transformers_backend  # noqa: E402
from egyptian_civil_code_rag.query import RAGQueryEngine  # noqa: E402


def load_questions(path: Path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def build_ragas_judge(model_name: str, max_new_tokens: int = 150):
    """Wraps our own model as RAGAS's judge LLM -- NOT OpenAI, which is
    the library default if this isn't done explicitly.

    max_new_tokens is deliberately much smaller than the main pipeline's
    (400): RAGAS's internal prompts ask for short structured verdicts
    (a score, brief reasoning), not long-form answers -- letting the
    judge run to 400 tokens every call was pure wasted CPU time."""
    pipe = hf_pipeline(
        "text-generation", model=model_name, max_new_tokens=max_new_tokens, do_sample=False
    )
    langchain_llm = HuggingFacePipeline(pipeline=pipe)
    return LangchainLLMWrapper(langchain_llm)


def build_ragas_embeddings(model_name: str):
    """Same reasoning as above, for the embedding-based metric
    (answer_relevancy) -- use BGE-M3, not an OpenAI embedding model."""
    langchain_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return LangchainEmbeddingsWrapper(langchain_embeddings)


def run_pipeline_on_questions(engine: RAGQueryEngine, questions: list[dict]):
    """Runs every question through the REAL retrieval + generation
    pipeline, capturing what RAGAS needs: question, retrieved contexts,
    generated answer, ground truth."""
    rows = []
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {q['id']}: {q['question'][:60]}...", file=sys.stderr)
        context_hits = engine.retrieve(q["question"])
        contexts = [hit.payload["text"] for hit in context_hits]
        result = engine.ask(q["question"])
        rows.append(
            {
                "user_input": q["question"],
                "retrieved_contexts": contexts,
                "response": result["answer"],
                "reference": q["ground_truth"],
                # kept for our own reporting, not consumed by RAGAS itself
                "_id": q["id"],
                "_category": q["category"],
                "_language": q["language"],
                "_sources": result["sources"],
                "_expected_articles": q["expected_articles"],
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=Path("tests/eval/eval_questions.json"))
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    ap.add_argument("--judge-model", type=str, default="Qwen/Qwen3-1.7B")
    ap.add_argument("--mlflow-experiment", type=str, default="civil-code-rag-eval")
    ap.add_argument("--out", type=Path, default=Path("reports/ragas_results.json"))
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions -- use this FIRST to confirm "
        "the harness works before committing to a full, slow CPU run "
        "across all questions.",
    )
    ap.add_argument(
        "--metrics",
        type=str,
        default="faithfulness,context_precision,context_recall,answer_relevancy",
        help="Comma-separated subset of metrics to run. Use a single metric "
        "(e.g. --metrics faithfulness) for the fastest possible smoke test "
        "-- 1/4 the judge calls of a full run.",
    )
    args = ap.parse_args()

    with open(args.params, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    print(f"[info] loaded {len(questions)} evaluation questions", file=sys.stderr)

    print("[info] loading pipeline models (BGE-M3 + Qwen3, CPU dev backend)", file=sys.stderr)
    engine = RAGQueryEngine(
        params_path=args.params,
        generate_fn=transformers_backend(args.judge_model),
    )

    rows = run_pipeline_on_questions(engine, questions)
    engine.close()

    print(
        "[info] loading RAGAS judge (same model, see module docstring "
        "for the caveat about a small model judging its own answers)",
        file=sys.stderr,
    )
    judge_llm = build_ragas_judge(args.judge_model)
    judge_embeddings = build_ragas_embeddings(params["embedding"]["model_name"])

    dataset = EvaluationDataset.from_list(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    )

    requested = set(args.metrics.split(","))
    available = {
        "faithfulness": Faithfulness(llm=judge_llm),
        "context_precision": ContextPrecision(llm=judge_llm),
        "context_recall": ContextRecall(llm=judge_llm),
        "answer_relevancy": AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
    }
    metrics = [m for name, m in available.items() if name in requested]
    print(
        f"[info] running {len(metrics)} metric(s) x {len(questions)} question(s) "
        f"= {len(metrics) * len(questions)} judge calls",
        file=sys.stderr,
    )

    print(
        "[info] running RAGAS evaluation -- this scores every question "
        "with an LLM call per metric, expect this to take a while on CPU",
        file=sys.stderr,
    )
    # RAGAS's default timeout (tuned for fast API judges like OpenAI) is
    # far too short for a local CPU-hosted judge -- measured ~1094s per
    # call for Qwen3-1.7B on this hardware before this fix, causing
    # every single job to hit RAGAS's timeout and fail with
    # TimeoutError. max_workers=1 forces sequential execution: parallel
    # calls would all contend for the SAME single loaded model instance
    # on CPU, adding contention rather than real concurrency, making
    # things slower not faster.
    run_config = RunConfig(timeout=1800, max_workers=1)
    ragas_result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)
    scores_df = ragas_result.to_pandas()

    # attach our own per-question metadata back for reporting
    for col in ("_id", "_category", "_language", "_sources", "_expected_articles"):
        scores_df[col] = [r[col] for r in rows]

    aggregate = {
        "faithfulness": float(scores_df["faithfulness"].mean()),
        "context_precision": float(scores_df["context_precision"].mean()),
        "context_recall": float(scores_df["context_recall"].mean()),
        "answer_relevancy": float(scores_df["answer_relevancy"].mean()),
    }
    print(f"[info] aggregate scores: {json.dumps(aggregate, indent=2)}", file=sys.stderr)

    # A judge this small giving suspiciously uniform (all ~1.0 or all
    # ~0.0) scores across every question is a sign the judge itself
    # isn't discriminating well, not necessarily that the pipeline is
    # flawless or broken -- flag it rather than let it pass silently.
    for metric_name, val in aggregate.items():
        col_std = scores_df[metric_name].std()
        if col_std is not None and col_std < 0.02:
            print(
                f"[warn] {metric_name} has near-zero variance across all questions "
                f"(std={col_std:.4f}) -- possible sign the judge model isn't "
                f"discriminating between good and bad answers, not necessarily "
                f"that every answer is equally good/bad. Consider a stronger judge.",
                file=sys.stderr,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_json(args.out, orient="records", force_ascii=False, indent=2)
    print(f"[info] wrote per-question results to {args.out}", file=sys.stderr)

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_param("chunk_size", params["chunking"]["paragraph_split_threshold_chars"])
        mlflow.log_param("embedding_model", params["embedding"]["model_name"])
        mlflow.log_param("judge_model", args.judge_model)
        mlflow.log_param("n_questions", len(questions))
        for metric_name, val in aggregate.items():
            mlflow.log_metric(metric_name, val)
        mlflow.log_artifact(str(args.out))
        print(f"[info] logged run to MLflow experiment '{args.mlflow_experiment}'", file=sys.stderr)


if __name__ == "__main__":
    main()
