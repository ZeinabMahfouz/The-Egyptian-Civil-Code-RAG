"""CI quality gate: fail the build if RAGAS faithfulness on the 20-question
CI subset is below the threshold -- or if the evaluation on record no longer
describes what the repo would build.

    python scripts/ragas_gate.py                     # threshold 0.75
    python scripts/ragas_gate.py --threshold 0.80

CI has no GPU, so it cannot re-run generation and judging on every push.
What it *can* do is refuse to pass when:
  1. there is no GPU evaluation on record (reports/ragas_results.json without
     real scores -- e.g. the old CPU run whose scores are all null),
  2. faithfulness on the CI subset is below the threshold, or
  3. the evaluation is stale: it was run against a different corpus (DVC md5
     of data/interim/civil_code.json) or different chunking/embedding params
     than params.yaml now specifies.
(3) is what makes this a gate and not a badge: change the chunker, the
embedding model or the corpus, and CI stays red until the GPU evaluation is
re-run and its new results committed.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from egyptian_civil_code_rag.refusal import OUT_OF_CORPUS, is_refusal  # noqa: E402

GATED_PARAMS = ("chunking", "embedding")


def corpus_md5_from_dvc_lock(lock: Path) -> str | None:
    data = yaml.safe_load(lock.read_text(encoding="utf-8"))
    for stage in data.get("stages", {}).values():
        for out in stage.get("outs", []):
            if out.get("path") == "data/interim/civil_code.json":
                return out.get("md5")
    return None


def check(
    report_path, subset_path, params_path, lock_path, threshold, min_refusal=0.75
) -> list[str]:
    """Returns a list of failure messages (empty = pass)."""
    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"no readable evaluation at {report_path}: {e}"]
    if not isinstance(report, dict) or "meta" not in report:
        return [
            f"{report_path} is not a GPU evaluation report (no 'meta' block) -- "
            "run scripts/gpu_eval.py full and commit its reports/ragas_results.json"
        ]

    failures = []
    subset = json.loads(Path(subset_path).read_text(encoding="utf-8"))["questions"]
    # Out-of-corpus questions are gated on declining, not on faithfulness
    # (see mean_scores in gpu_eval.py for why RAGAS can't score a refusal).
    subset_ids = {q["id"] for q in subset if q["category"] != OUT_OF_CORPUS}
    ooc_ids = {q["id"] for q in subset if q["category"] == OUT_OF_CORPUS}
    ooc_rows = [r for r in report["rows"] if r.get("_id") in ooc_ids]
    if ooc_ids:
        declined = sum(is_refusal(r.get("response", "")) for r in ooc_rows)
        print(f"out-of-corpus declined: {declined}/{len(ooc_ids)} (minimum {min_refusal:.0%})")
        if declined < len(ooc_ids) * min_refusal:
            failures.append(
                f"only {declined}/{len(ooc_ids)} out-of-corpus questions were declined -- "
                "the system answered questions the Civil Code doesn't cover"
            )
    scores = [
        r["faithfulness"]
        for r in report["rows"]
        if r.get("_id") in subset_ids and isinstance(r.get("faithfulness"), (int, float))
    ]
    if len(scores) < len(subset_ids) * 0.8:
        failures.append(
            f"only {len(scores)}/{len(subset_ids)} CI-subset questions have a faithfulness "
            "score -- the judge failed on too many to trust the mean"
        )
    if scores:
        mean = sum(scores) / len(scores)
        print(
            f"faithfulness on CI subset: {mean:.3f} over {len(scores)} questions "
            f"(threshold {threshold})"
        )
        if mean < threshold:
            failures.append(f"faithfulness {mean:.3f} < {threshold}")

    params = yaml.safe_load(Path(params_path).read_text(encoding="utf-8"))
    for section in GATED_PARAMS:
        if report["meta"]["params"].get(section) != params.get(section):
            failures.append(
                f"stale evaluation: params.yaml '{section}' is {params.get(section)}, but the "
                f"evaluation used {report['meta']['params'].get(section)} -- re-run the GPU eval"
            )
    lock_md5 = corpus_md5_from_dvc_lock(Path(lock_path))
    if lock_md5 and report["meta"].get("corpus_md5") != lock_md5:
        failures.append(
            "stale evaluation: corpus changed since it was evaluated "
            f"(dvc.lock md5 {lock_md5}, evaluated {report['meta'].get('corpus_md5')})"
        )
    return failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--report", default="reports/ragas_results.json")
    ap.add_argument("--subset", default="tests/eval/eval_questions_ci_subset.json")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--dvc-lock", default="dvc.lock")
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument(
        "--min-refusal", type=float, default=0.75, help="share of out-of-corpus questions declined"
    )
    args = ap.parse_args(argv)
    failures = check(
        args.report, args.subset, args.params, args.dvc_lock, args.threshold, args.min_refusal
    )
    for f in failures:
        print(f"[FAIL] {f}")
    if not failures:
        print("[PASS] RAGAS gate")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
