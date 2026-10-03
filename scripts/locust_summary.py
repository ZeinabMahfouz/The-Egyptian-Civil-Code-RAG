"""Turns Locust's CSV output into the table for reports/ and the README.

    python scripts/locust_summary.py reports/locust_u1 reports/locust_u50 \\
        --out reports/load_test.md

Each argument is a --csv prefix given to locust; the script reads
<prefix>_stats.csv and reports the /ask row: requests, failures, throughput
and latency percentiles.
"""

import argparse
import csv
from pathlib import Path


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
    args = ap.parse_args(argv)
    md = to_markdown([summarize(p) for p in args.prefixes])
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    print(md)
    return md


if __name__ == "__main__":
    main()
