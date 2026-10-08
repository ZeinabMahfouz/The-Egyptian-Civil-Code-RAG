"""Calibrates the refusal gate (RAGQueryEngine.should_refuse) on the real
index: retrieval only, no LLM, runs on a laptop CPU in a few minutes.

    python scripts/refusal_calibration.py            # writes reports/refusal_calibration.{md,json}

Two question sets, kept apart so the threshold is not judged on the data
that chose it:

- tune:  tests/eval/eval_questions.json -- 48 in-corpus, 6 out-of-corpus
- check: tests/eval/drift_windows.json  -- 16 in-corpus (in_domain),
         32 out-of-corpus (other_jurisdiction, off_topic)

For each question: the best match's similarity, and whether it names an
article found in the index (exempt -- the gate never refuses those).

Recommended threshold: just below the lowest-scoring in-corpus question of
the tune set, so no tune question the system should answer is refused, and
as many out-of-corpus ones as possible are. Then the check set says how
that holds up on questions it never saw.
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_OF_CORPUS_WINDOWS = {"other_jurisdiction", "off_topic"}
MARGIN = 0.005  # below the lowest in-corpus score, so ties don't refuse it


def load_questions(eval_path: Path, windows_path: Path) -> list[dict]:
    rows = []
    for q in json.loads(eval_path.read_text(encoding="utf-8"))["questions"]:
        rows.append(
            {
                "id": q["id"],
                "set": "tune",
                "group": q["category"],
                "question": q["question"],
                "in_corpus": q["category"] != "out_of_corpus",
            }
        )
    windows = json.loads(windows_path.read_text(encoding="utf-8"))["windows"]
    for name, w in windows.items():
        for i, text in enumerate(w["queries"]):
            rows.append(
                {
                    "id": f"{name}_{i:02d}",
                    "set": "check",
                    "group": name,
                    "question": text,
                    "in_corpus": name not in OUT_OF_CORPUS_WINDOWS,
                }
            )
    return rows


def score_questions(engine, rows: list[dict]) -> list[dict]:
    """Adds the best match's similarity and the exemption flag to each row,
    using the same retrieve() and exemption rule as the API."""
    gate_off = engine.refusal_min_score if hasattr(engine, "refusal_min_score") else None
    engine.refusal_min_score = float("inf")  # "would refuse anything": isolates the exemption
    try:
        for r in rows:
            hits = engine.retrieve(r["question"])
            r["top_score"] = round(max(float(h.score) for h in hits), 4) if hits else None
            r["exempt"] = bool(hits) and not engine.should_refuse(r["question"], hits)
    finally:
        engine.refusal_min_score = gate_off
    return rows


def refused(r: dict, threshold: float) -> bool:
    return not r["exempt"] and r["top_score"] is not None and r["top_score"] < threshold


def outcome(rows: list[dict], threshold: float) -> dict:
    out = {}
    for s in ("tune", "check"):
        ins = [r for r in rows if r["set"] == s and r["in_corpus"]]
        oos = [r for r in rows if r["set"] == s and not r["in_corpus"]]
        out[s] = {
            "in_corpus_refused": sum(refused(r, threshold) for r in ins),
            "in_corpus": len(ins),
            "out_of_corpus_refused": sum(refused(r, threshold) for r in oos),
            "out_of_corpus": len(oos),
        }
    return out


def recommend(rows: list[dict]) -> float:
    gated = [
        r["top_score"] for r in rows if r["set"] == "tune" and r["in_corpus"] and not r["exempt"]
    ]
    return round(min(gated) - MARGIN, 3)


def report(rows: list[dict], candidates: dict[str, float]) -> tuple[str, dict]:
    def frac(a, b):
        return f"{a} of {b}"

    lines = [
        "# Refusal gate calibration",
        "",
        "Retrieval only (`scripts/refusal_calibration.py`). A question is refused when its",
        "best match among the indexed articles scores below the threshold, unless it names",
        "an article found in the index.",
        "",
        "| Threshold | | Tune: out-of-corpus refused | Tune: in-corpus refused "
        "| Check: out-of-corpus refused | Check: in-corpus refused |",
        "|---|---|---|---|---|---|",
    ]
    table = {}
    for label, t in sorted(candidates.items(), key=lambda kv: kv[1]):
        o = outcome(rows, t)
        table[label] = {"threshold": t, **o}
        cells = [
            frac(o[part][f"{kind}_refused"], o[part][kind])
            for part in ("tune", "check")
            for kind in ("out_of_corpus", "in_corpus")
        ]
        lines.append(f"| {t:.3f} | {label} | " + " | ".join(cells) + " |")

    lines += ["", "## Closest calls", ""]
    lines += [
        "Lowest-scoring in-corpus questions and highest-scoring out-of-corpus ones",
        "(exempt questions excluded):",
        "",
        "| Set | Group | Id | Best match | Question |",
        "|---|---|---|---|---|",
    ]
    gated = [r for r in rows if not r["exempt"] and r["top_score"] is not None]
    lows = sorted((r for r in gated if r["in_corpus"]), key=lambda r: r["top_score"])[:6]
    highs = sorted((r for r in gated if not r["in_corpus"]), key=lambda r: -r["top_score"])[:6]
    for r in lows + highs:
        q = r["question"].replace("|", "/")
        lines.append(f"| {r['set']} | {r['group']} | {r['id']} | {r['top_score']:.3f} | {q} |")
    return "\n".join(lines) + "\n", table


def main(argv=None, engine=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval", type=Path, default=REPO / "tests/eval/eval_questions.json")
    ap.add_argument("--windows", type=Path, default=REPO / "tests/eval/drift_windows.json")
    ap.add_argument("--reports-dir", type=Path, default=REPO / "reports")
    ap.add_argument("--drift-cutoff", type=float, default=None, help="default: reports/drift.json")
    args = ap.parse_args(argv)

    if engine is None:
        from egyptian_civil_code_rag.query import RAGQueryEngine

        engine = RAGQueryEngine(params_path=REPO / "params.yaml")
    rows = score_questions(engine, load_questions(args.eval, args.windows))

    cutoff = args.drift_cutoff
    if cutoff is None:
        drift = REPO / "reports" / "drift.json"
        if drift.exists():
            windows = json.loads(drift.read_text(encoding="utf-8"))["windows"]
            cutoff = next(iter(windows.values())).get("corpus_cutoff")
    candidates = {"recommended": recommend(rows)}
    if cutoff is not None:
        candidates["drift cut-off"] = round(cutoff, 3)
    for t in (0.55, 0.60, 0.65, 0.70):
        candidates.setdefault(f"{t:.2f}", t)

    md, table = report(rows, candidates)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "refusal_calibration.md").write_text(md, encoding="utf-8")
    result = {"recommended": candidates["recommended"], "thresholds": table, "questions": rows}
    (args.reports_dir / "refusal_calibration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(md)
    print(f"Recommended: refusal.min_score = {candidates['recommended']} in params.yaml")
    return result


if __name__ == "__main__":
    main()
