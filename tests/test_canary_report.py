"""Canary promotion gate (deploy/canary/canary_report.py)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "deploy" / "canary"))

from canary_report import evaluate, load_stats, main, percentile  # noqa: E402

STABLE_ADDR, CANARY_ADDR = "172.18.0.2:8000", "172.18.0.3:8000"


def rec(release, addr, status=200, t=1.0, uri="/ask"):
    return json.dumps(
        {"uri": uri, "status": status, "request_time": t, "upstream": addr, "release": release}
    )


def healthy_log(n_stable=95, n_canary=25, canary_t=1.0):
    lines = [rec("stable-a", STABLE_ADDR) for _ in range(n_stable)]
    lines += [rec("canary-b", CANARY_ADDR, t=canary_t) for _ in range(n_canary)]
    return lines


def test_percentile_is_nearest_rank():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([5.0], 95) == 5.0
    assert percentile([], 95) is None


def test_healthy_canary_passes():
    stats = load_stats(healthy_log())
    results = evaluate(stats["stable-a"], stats["canary-b"], 20, 0.01, 1.2)
    assert all(passed for _, passed, _ in results)


def test_slow_canary_fails_latency_gate():
    stats = load_stats(healthy_log(canary_t=2.0))
    results = dict(
        (n, p) for n, p, _ in evaluate(stats["stable-a"], stats["canary-b"], 20, 0.01, 1.2)
    )
    assert results["p95 latency"] is False


def test_too_little_traffic_holds():
    stats = load_stats(healthy_log(n_canary=5))
    results = dict(
        (n, p) for n, p, _ in evaluate(stats["stable-a"], stats["canary-b"], 20, 0.01, 1.2)
    )
    assert results["enough canary traffic"] is False


def test_headerless_canary_failures_are_attributed_to_the_canary():
    # A crashing canary returns 500/502 with no X-App-Release header. Those
    # must count against the canary (by upstream address), not vanish.
    lines = healthy_log()
    lines += [rec("", CANARY_ADDR, status=502) for _ in range(5)]
    stats = load_stats(lines)
    assert stats["canary-b"].errors == 5
    assert "unknown" not in stats
    results = dict(
        (n, p) for n, p, _ in evaluate(stats["stable-a"], stats["canary-b"], 20, 0.01, 1.2)
    )
    assert results["error rate"] is False


def test_cli_exit_codes(tmp_path):
    good = tmp_path / "good.log"
    good.write_text("\n".join(healthy_log()))
    assert main([str(good)]) == 0

    bad = tmp_path / "bad.log"
    bad.write_text("\n".join(healthy_log(canary_t=5.0)))
    assert main([str(bad)]) == 1

    no_canary = tmp_path / "none.log"
    no_canary.write_text("\n".join(healthy_log(n_canary=0)))
    assert main([str(no_canary)]) == 2
