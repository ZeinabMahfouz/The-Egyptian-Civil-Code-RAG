"""Turns Locust's CSV output into the table for reports/ and the README.

    python scripts/locust_summary.py reports/locust_u1 reports/locust_u50 \\
        --out reports/load_test.md

Each argument is a --csv prefix given to locust; the script reads
<prefix>_stats.csv and reports the /ask row: requests, failures, throughput
and latency percentiles.

--metrics adds the API's own view: mean time per pipeline stage, from a
/metrics snapshot taken after the run (one snapshot per API process, so each
run's histogram covers that run only).
"""

import argparse
import csv
import re
from pathlib import Path

RE_STAGE = re.compile(
    r'^rag_request_latency_seconds_(sum|count)\{[^}]*stage="(\w+)"[^}]*\}\s+([0-9.eE+-]+)$'
)


def stage_means(metrics_text: str) -> dict[str, dict]:
    """Prometheus exposition -> {stage: {"count", "mean_s"}} for the
    rag_request_latency_seconds histogram (all services summed)."""
    acc: dict[str, dict] = {}
    for line in metrics_text.splitlines():
        m = RE_STAGE.match(line.strip())
        if m:
            kind, stage, value = m.groups()
            acc.setdefault(stage, {"sum": 0.0, "count": 0.0})[kind] += float(value)
    return {
        stage: {"count": int(v["count"]), "mean_s": v["sum"] / v["count"] if v["count"] else None}
        for stage, v in acc.items()
    }


def stages_markdown(named: list[tuple[str, dict]]) -> str:
    stages = [s for s in ("retrieve", "generate", "total") if any(s in d for _, d in named)]
    lines = ["| Run | " + " | ".join(f"{s} (mean)" for s in stages) + " |"]
    lines.append("|---|" + "---|" * len(stages))
    for name, d in named:
        cells = [
            f"{d[s]['mean_s']:.2f} s" if s in d and d[s]["mean_s"] is not None else "n/a"
            for s in stages
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def read_row(prefix: str, name: str = "/ask") -> dict | None:
    path = Path(f"{prefix}_stats.csv")
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("Name") == name:
                return row
    return None


def summarize(prefix: str, name: str = "/ask") -> dict:
    row = read_row(prefix, name)
    if row is None:
        raise SystemExit(f"no '{name}' row in {prefix}_stats.csv")

    def num(key):
        v = row.get(key, "")
        return float(v) if v not in ("", "N/A") else None

    requests = int(num("Request Count") or 0)
    failures = int(num("Failure Count") or 0)
    return {
        "run": Path(prefix).name,
        "requests": requests,
        "failures": failures,
        "failure_rate": failures / requests if requests else None,
        "rps": num("Requests/s"),
        "p50_s": (num("50%") or 0) / 1000,
        "p95_s": (num("95%") or 0) / 1000,
        "p99_s": (num("99%") or 0) / 1000,
        "max_s": (num("Max Response Time") or 0) / 1000,
    }


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "| Run | Requests | Failures | Requests/s | p50 | p95 | p99 | Max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        fr = (
            f"{r['failures']} ({r['failure_rate']:.1%})" if r["failure_rate"] is not None else "n/a"
        )
        rps = f"{r['rps']:.2f}" if r["rps"] is not None else "n/a"
        lines.append(
            f"| {r['run']} | {r['requests']} | {fr} | {rps} | {r['p50_s']:.2f} s | "
            f"{r['p95_s']:.2f} s | {r['p99_s']:.2f} s | {r['max_s']:.2f} s |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--metrics", nargs="+", default=[], help="/metrics snapshots, one per prefix, same order"
    )
    args = ap.parse_args(argv)
    md = to_markdown([summarize(p) for p in args.prefixes])
    if args.metrics:
        if len(args.metrics) != len(args.prefixes):
            raise SystemExit("--metrics needs one file per prefix")
        named = [
            (Path(p).name, stage_means(Path(m).read_text(encoding="utf-8")))
            for p, m in zip(args.prefixes, args.metrics)
        ]
        md += "\nTime per stage (API metrics):\n\n" + stages_markdown(named)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    print(md)
    return md


if __name__ == "__main__":
    main()
