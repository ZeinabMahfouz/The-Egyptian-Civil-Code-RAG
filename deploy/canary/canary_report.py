"""Canary promotion gate: compare canary vs stable from the nginx access log.

    python deploy/canary/canary_report.py deploy/canary/logs/canary_access.log

Reads the JSON lines written by default.conf.template's `canary` log format,
groups requests by the X-App-Release header each API instance sets, and
checks the canary against the stable release. Exits 1 if any gate fails,
so it can run as a scripted step before moving to the next rollout stage.

Gates (defaults; override with flags):
  - enough canary /ask traffic to judge at all          (--min-requests 20)
  - canary 5xx rate <= stable 5xx rate + 1 percentage pt (--max-error-delta 0.01)
  - canary /ask p95 latency <= 1.2 x stable p95          (--max-p95-ratio 1.2)

These cover "is it up and is it slow". Answer quality is a separate gate
(RAGAS faithfulness on the canary image) -- see README "Canary rollout".
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReleaseStats:
    release: str
    requests: int = 0
    errors: int = 0  # 5xx, and 502/504 from nginx itself
    ask_latencies: list[float] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0

    @property
    def ask_p95(self) -> float | None:
        return percentile(self.ask_latencies, 95)


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile -- no interpolation, so p95 is always a
    latency that actually happened."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def load_stats(log_lines) -> dict[str, ReleaseStats]:
    records = [json.loads(line) for line in log_lines if line.strip()]

    # A request that failed inside the app (unhandled 500) or never got a
    # response (nginx 502/504) carries no X-App-Release header. Attributing
    # it by header alone would file the canary's crashes under "unknown"
    # and let a broken canary pass the error gate -- so learn which upstream
    # address serves which release from the successful responses, and
    # attribute header-less failures by address.
    release_of_addr = {}
    for rec in records:
        if rec.get("release") and rec.get("upstream"):
            release_of_addr[rec["upstream"]] = rec["release"]

    stats: dict[str, ReleaseStats] = {}
    for rec in records:
        release = rec.get("release") or release_of_addr.get(rec.get("upstream"), "unknown")
        s = stats.setdefault(release, ReleaseStats(release))
        s.requests += 1
        if int(rec["status"]) >= 500:
            s.errors += 1
        if rec.get("uri") == "/ask" and int(rec["status"]) < 500:
            s.ask_latencies.append(float(rec["request_time"]))
    return stats


def evaluate(
    stable: ReleaseStats,
    canary: ReleaseStats,
    min_requests: int,
    max_error_delta: float,
    max_p95_ratio: float,
) -> list[tuple[str, bool, str]]:
    """Returns (gate name, passed, detail) for every gate."""
    results = []
    n_ask = len(canary.ask_latencies)
    results.append(
        (
            "enough canary traffic",
            n_ask >= min_requests,
            f"{n_ask} successful canary /ask requests (need >= {min_requests})",
        )
    )
    results.append(
        (
            "error rate",
            canary.error_rate <= stable.error_rate + max_error_delta,
            f"canary {canary.error_rate:.2%} vs stable {stable.error_rate:.2%} "
            f"(allowed +{max_error_delta:.0%})",
        )
    )
    s_p95, c_p95 = stable.ask_p95, canary.ask_p95
    if s_p95 is None or c_p95 is None:
        results.append(("p95 latency", False, "no /ask latencies for one of the releases"))
    else:
        results.append(
            (
                "p95 latency",
                c_p95 <= s_p95 * max_p95_ratio,
                f"canary {c_p95:.2f}s vs stable {s_p95:.2f}s (allowed x{max_p95_ratio})",
            )
        )
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--stable", default=None, help="stable release name (default: auto)")
    ap.add_argument("--canary", default=None, help="canary release name (default: auto)")
    ap.add_argument("--min-requests", type=int, default=20)
    ap.add_argument("--max-error-delta", type=float, default=0.01)
    ap.add_argument("--max-p95-ratio", type=float, default=1.2)
    args = ap.parse_args(argv)

    with open(args.log, encoding="utf-8") as f:
        stats = load_stats(f)

    print(f"{'release':<28}{'requests':>9}{'share':>8}{'5xx':>7}{'/ask p95':>10}")
    total = sum(s.requests for s in stats.values()) or 1
    for s in sorted(stats.values(), key=lambda s: -s.requests):
        p95 = f"{s.ask_p95:.2f}s" if s.ask_p95 is not None else "-"
        print(
            f"{s.release:<28}{s.requests:>9}{s.requests / total:>8.1%}{s.error_rate:>7.1%}{p95:>10}"
        )

    # Auto-detect: the release names in .env.example start with stable-/canary-.
    def pick(explicit, prefix):
        if explicit:
            return stats.get(explicit)
        matches = [s for name, s in stats.items() if name.startswith(prefix)]
        return matches[0] if len(matches) == 1 else None

    stable, canary = pick(args.stable, "stable"), pick(args.canary, "canary")
    if stable is None or canary is None:
        print(
            "\n[error] couldn't identify one stable and one canary release in the log -- "
            "pass --stable/--canary explicitly",
            file=sys.stderr,
        )
        return 2

    print(f"\nGates: {canary.release} vs {stable.release}")
    results = evaluate(stable, canary, args.min_requests, args.max_error_delta, args.max_p95_ratio)
    for name, passed, detail in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    ok = all(passed for _, passed, _ in results)
    print("\nPROMOTE to next stage" if ok else "\nHOLD / ROLL BACK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
