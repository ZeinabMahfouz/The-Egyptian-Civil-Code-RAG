"""AWQ 4-bit vs fp16 Qwen3-8B: answer quality (RAGAS) and latency.

Two GPU servers can't both fit on Kaggle's 2x T4, so the comparison runs in
steps, one vLLM server at a time (see notebooks/kaggle_quantization.ipynb):

    # 1. with Qwen/Qwen3-8B-AWQ served: answer all 54 questions, timed
    python scripts/quant_compare.py generate --label awq --model Qwen/Qwen3-8B-AWQ
    # 2. with Qwen/Qwen3-8B (fp16) served: the same, then judge BOTH sets
    python scripts/quant_compare.py generate --label fp16 --model Qwen/Qwen3-8B
    python scripts/quant_compare.py score --labels fp16 awq --judge-model Qwen/Qwen3-8B
    # 3. compare, log to MLflow, write reports/quantization.{json,md}
    python scripts/quant_compare.py report

The judge is the fp16 model for both answer sets. Judging the AWQ answers
with the AWQ model would change two things at once and make the
comparison meaningless.

Latency is measured two ways, on the same retrieval and the same prompts:
  - sequential: one request at a time, streamed. Time to first token and
    total latency per question (p50 / p95) -- what one user waits for.
  - concurrent: all prompts at once from N threads. Output tokens per
    second across the batch -- what the server sustains under load.

Acceptance: faithfulness may drop by less than MAX_FAITHFULNESS_DROP.
"""

import argparse
import json
import math
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import gpu_eval  # noqa: E402

from egyptian_civil_code_rag.query import RAGQueryEngine  # noqa: E402

MAX_FAITHFULNESS_DROP = 0.03
EXPERIMENT = "civil-code-rag-quantization"
LABELS = ("fp16", "awq")

# vLLM logs the weights it loaded per GPU worker, e.g.
#   "Model loading took 7.6719 GiB and 31.2 seconds"   (newer)
#   "Loading model weights took 7.6719 GB"             (older)
RE_WEIGHTS = re.compile(r"(?:Model loading|Loading model weights) took ([\d.]+) ?Gi?B")


# --- measurement helpers ------------------------------------------------------


def percentile(values: list, q: float) -> float | None:
    """Nearest-rank percentile; None for an empty list."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = max(0, math.ceil(q / 100 * len(vals)) - 1)
    return vals[k]


def weights_gib_from_log(log_path: Path | None) -> float | None:
    """Per-GPU weight memory from the vLLM server log (last match wins)."""
    if not log_path or not Path(log_path).exists():
        return None
    found = RE_WEIGHTS.findall(Path(log_path).read_text(errors="replace"))
    return float(found[-1]) if found else None


def gpu_memory_used_mib() -> list | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return [int(x) for x in out.split()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def timed_answer(generate_fn, prompt: str) -> tuple[str, float, float | None, int | None]:
    """Streams one answer. Returns (answer, total_s, ttft_s, output_tokens)."""
    t0 = time.perf_counter()
    ttft, pieces = None, []
    for piece in generate_fn.stream(prompt):
        if ttft is None and piece:
            ttft = time.perf_counter() - t0
        pieces.append(piece)
    total = time.perf_counter() - t0
    usage = getattr(generate_fn, "last_usage", None) or {}
    return "".join(pieces).strip(), total, ttft, usage.get("output")


def concurrent_throughput(generate_fn, prompts: list, concurrency: int) -> dict:
    """All prompts through N threads at once; the server batches them.

    Counts streamed chunks rather than reading generate_fn.last_usage, which
    is one attribute shared by all threads. vLLM streams about one token per
    chunk, so chunks/s approximates output tokens/s."""

    def one(p):
        n, t0 = 0, time.perf_counter()
        for _ in generate_fn.stream(p):
            n += 1
        return time.perf_counter() - t0, n

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, prompts))
    wall = time.perf_counter() - t0
    latencies = [r[0] for r in results]
    chunks = sum(r[1] for r in results)
    return {
        "concurrency": concurrency,
        "n_requests": len(prompts),
        "wall_s": wall,
        "output_chunks_per_s": chunks / wall if wall else None,
        "requests_per_s": len(prompts) / wall if wall else None,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
    }


# --- steps ----------------------------------------------------------------------


def work_path(work_dir, label: str) -> Path:
    return Path(work_dir) / "quant" / f"{label}.json"


def generate(args, embed_model=None, generate_fn=None) -> dict:
    """Answers all questions with the model currently served, timed."""
    from sentence_transformers import SentenceTransformer

    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    questions = gpu_eval.load_questions(REPO / "tests" / "eval" / "eval_questions.json")[
        : args.limit
    ]
    embed_model = embed_model or SentenceTransformer(args.embedding_model, device=args.embed_device)
    if generate_fn is None:
        from egyptian_civil_code_rag.backends import openai_backend

        generate_fn = openai_backend(args.vllm_url, args.model)

    # The production chunking config (what params.yaml builds), same for both models.
    cfg = gpu_eval.SWEEP[0]
    client, collection, _ = gpu_eval.build_index(
        cfg, corpus, embed_model, Path(args.work_dir) / "indexes", batch_size=args.embed_batch
    )
    engine = RAGQueryEngine(
        params_path=REPO / "params.yaml",
        generate_fn=generate_fn,
        embed_model=embed_model,
        client=client,
        collection=collection,
    )

    # Retrieval once per question; both latency passes reuse the same prompts.
    retrieved = [(q, engine.retrieve(q["question"])) for q in questions]
    prompts = [engine.build_prompt(q["question"], hits) for q, hits in retrieved if hits]

    if args.warmup:
        generate_fn(prompts[0])  # first request pays CUDA-graph / cache warmup

    rows = []
    for q, hits in retrieved:
        if hits:
            answer, total, ttft, n_out = timed_answer(
                generate_fn, engine.build_prompt(q["question"], hits)
            )
        else:
            answer, total, ttft, n_out = gpu_eval.NO_CONTEXT_ANSWER, None, None, None
        row = gpu_eval.make_row(q, hits, answer)
        row.update({"_latency_s": total, "_ttft_s": ttft, "_output_tokens": n_out})
        rows.append(row)
    client.close()

    load = (
        concurrent_throughput(generate_fn, prompts, args.concurrency) if args.concurrency else None
    )
    out = {
        "meta": {
            "label": args.label,
            "model": args.model,
            "config": cfg.name,
            "n_questions": len(questions),
            "weights_gib_per_gpu": weights_gib_from_log(args.vllm_log),
            "gpu_memory_used_mib": gpu_memory_used_mib(),
            "concurrent": load,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "rows": rows,
    }
    path = work_path(args.work_dir, args.label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[generate] {args.label}: {len(rows)} answers -> {path}")
    print(json.dumps(latency_summary(out), indent=2))
    return out


def score(args, score_fn=None) -> None:
    """RAGAS on each answer set, all judged by the model currently served."""
    score_fn = score_fn or gpu_eval.score_with_ragas
    for label in args.labels:
        path = work_path(args.work_dir, label)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["rows"] = score_fn(
            data["rows"],
            args.vllm_url,
            args.judge_model,
            args.embedding_model,
            workers=args.judge_workers,
        )
        data["meta"]["judge"] = args.judge_model
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        s = gpu_eval.mean_scores(data["rows"])
        print(f"[score] {label}: faithfulness={s['faithfulness']} (n={s['faithfulness_n_scored']})")


def latency_summary(data: dict) -> dict:
    rows = data["rows"]
    lat = [r.get("_latency_s") for r in rows]
    ttft = [r.get("_ttft_s") for r in rows]
    toks = [
        r["_output_tokens"] / r["_latency_s"]
        for r in rows
        if r.get("_output_tokens") and r.get("_latency_s")
    ]
    conc = data["meta"].get("concurrent") or {}
    return {
        "latency_p50_s": percentile(lat, 50),
        "latency_p95_s": percentile(lat, 95),
        "ttft_p50_s": percentile(ttft, 50),
        "ttft_p95_s": percentile(ttft, 95),
        "tokens_per_s_p50": percentile(toks, 50),
        "concurrent_requests_per_s": conc.get("requests_per_s"),
        "concurrent_tokens_per_s": conc.get("output_chunks_per_s"),
        "concurrent_latency_p95_s": conc.get("latency_p95_s"),
    }


def compare(fp16: dict, awq: dict) -> dict:
    q16, qawq = gpu_eval.mean_scores(fp16["rows"]), gpu_eval.mean_scores(awq["rows"])
    l16, lawq = latency_summary(fp16), latency_summary(awq)
    drop = None
    if q16.get("faithfulness") is not None and qawq.get("faithfulness") is not None:
        drop = q16["faithfulness"] - qawq["faithfulness"]

    def ratio(a, b):
        return a / b if a and b else None

    return {
        "quality": {"fp16": q16, "awq": qawq},
        "latency": {"fp16": l16, "awq": lawq},
        "memory": {
            label: {
                "weights_gib_per_gpu": d["meta"].get("weights_gib_per_gpu"),
                "model": d["meta"].get("model"),
            }
            for label, d in (("fp16", fp16), ("awq", awq))
        },
        "faithfulness_drop": drop,
        "max_allowed_drop": MAX_FAITHFULNESS_DROP,
        "passed": drop is not None and drop < MAX_FAITHFULNESS_DROP,
        "speedup_latency_p50": ratio(l16["latency_p50_s"], lawq["latency_p50_s"]),
        "speedup_concurrent_rps": ratio(
            lawq["concurrent_requests_per_s"], l16["concurrent_requests_per_s"]
        ),
        "judge": fp16["meta"].get("judge"),
    }


def _fmt(v, nd=3):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def to_markdown(c: dict) -> str:
    q, lat, mem = c["quality"], c["latency"], c["memory"]
    lines = [
        "# Quantization: Qwen3-8B fp16 vs AWQ 4-bit",
        "",
        f"Judge for both: {c['judge']}. Acceptance: faithfulness drop < "
        f"{c['max_allowed_drop']}. Result: **{'PASS' if c['passed'] else 'FAIL'}** "
        f"(drop {_fmt(c['faithfulness_drop'])}).",
        "",
        "## Quality (RAGAS, in-corpus questions)",
        "",
        "| Metric | fp16 | AWQ | Change |",
        "|---|---|---|---|",
    ]
    for m in (*gpu_eval.RAGAS_METRICS, "refusal_rate", "false_refusal_rate"):
        a, b = q["fp16"].get(m), q["awq"].get(m)
        d = (b - a) if a is not None and b is not None else None
        lines.append(f"| {m} | {_fmt(a)} | {_fmt(b)} | {_fmt(d)} |")
    lines += [
        "",
        "## Latency and memory",
        "",
        "| Measure | fp16 | AWQ |",
        "|---|---|---|",
    ]
    for k in (
        "latency_p50_s",
        "latency_p95_s",
        "ttft_p50_s",
        "ttft_p95_s",
        "tokens_per_s_p50",
        "concurrent_requests_per_s",
        "concurrent_tokens_per_s",
        "concurrent_latency_p95_s",
    ):
        lines.append(f"| {k} | {_fmt(lat['fp16'][k], 2)} | {_fmt(lat['awq'][k], 2)} |")
    lines.append(
        f"| weights per GPU (GiB) | {_fmt(mem['fp16']['weights_gib_per_gpu'], 2)} | "
        f"{_fmt(mem['awq']['weights_gib_per_gpu'], 2)} |"
    )
    return "\n".join(lines) + "\n"


def report(args) -> dict:
    import mlflow

    data = {
        label: json.loads(work_path(args.work_dir, label).read_text(encoding="utf-8"))
        for label in LABELS
    }
    c = compare(data["fp16"], data["awq"])

    mlflow.set_tracking_uri(
        args.mlflow_uri or f"sqlite:///{Path(args.work_dir).resolve()}/mlflow.db"
    )
    mlflow.set_experiment(EXPERIMENT)
    for label in LABELS:
        meta = data[label]["meta"]
        with mlflow.start_run(run_name=f"quant-{label}"):
            mlflow.log_params(
                {
                    "model": meta["model"],
                    "quantization": "awq-4bit" if label == "awq" else "none (fp16)",
                    "judge": c["judge"],
                    "config": meta["config"],
                    "n_questions": meta["n_questions"],
                }
            )
            metrics = {**c["quality"][label], **c["latency"][label]}
            if c["memory"][label]["weights_gib_per_gpu"] is not None:
                metrics["weights_gib_per_gpu"] = c["memory"][label]["weights_gib_per_gpu"]
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            if label == "awq" and c["faithfulness_drop"] is not None:
                mlflow.log_metric("faithfulness_drop_vs_fp16", c["faithfulness_drop"])

    out_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quantization.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
    (out_dir / "quantization.md").write_text(to_markdown(c), encoding="utf-8")
    print(to_markdown(c))
    return c


# --- entry point ----------------------------------------------------------------


def main(argv=None, embed_model=None, generate_fn=None, score_fn=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)

    def common(p):
        p.add_argument("--work-dir", default="eval_out")
        p.add_argument("--vllm-url", default="http://localhost:8000/v1")
        p.add_argument("--embedding-model", default="BAAI/bge-m3")

    g = sub.add_parser("generate")
    common(g)
    g.add_argument("--label", required=True, choices=LABELS)
    g.add_argument("--model", required=True)
    g.add_argument("--corpus", default=str(REPO / "data" / "interim" / "civil_code.json"))
    g.add_argument("--embed-device", default=None)
    g.add_argument("--embed-batch", type=int, default=16)
    g.add_argument("--vllm-log", default=None, help="vLLM server log, for weight memory")
    g.add_argument("--concurrency", type=int, default=8, help="0 = skip the load pass")
    g.add_argument("--no-warmup", dest="warmup", action="store_false")
    g.add_argument("--limit", type=int, default=None)

    s = sub.add_parser("score")
    common(s)
    s.add_argument("--labels", nargs="+", default=list(LABELS))
    s.add_argument("--judge-model", default="Qwen/Qwen3-8B")
    s.add_argument("--judge-workers", type=int, default=8)

    r = sub.add_parser("report")
    common(r)
    r.add_argument("--mlflow-uri", default=None)
    r.add_argument("--reports-dir", default=str(REPO / "reports"))

    args = ap.parse_args(argv)
    if args.step == "generate":
        return generate(args, embed_model=embed_model, generate_fn=generate_fn)
    if args.step == "score":
        return score(args, score_fn=score_fn)
    return report(args)


if __name__ == "__main__":
    main()
