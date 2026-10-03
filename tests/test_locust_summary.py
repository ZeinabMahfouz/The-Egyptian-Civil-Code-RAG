"""Locust CSV -> report table (scripts/locust_summary.py), on a CSV in
Locust's own format. The load test itself runs on Kaggle."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from locust_summary import main, summarize  # noqa: E402

HEADER = (
    "Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,"
    "Min Response Time,Max Response Time,Average Content Size,Requests/s,Failures/s,"
    "50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%\n"
)


def write(tmp_path, name, ask_row):
    p = tmp_path / f"{name}_stats.csv"
    p.write_text(
        HEADER
        + "GET,/health,100,0,5,6,2,40,60,0.33,0,5,6,7,8,10,15,20,30,40,40,40\n"
        + ask_row
        + "\n,Aggregated,1100,4,900,1000,200,30000,500,3.6,0.01,900,1000,1100,1200,2000,"
        "5000,8000,12000,30000,30000,30000\n",
        encoding="utf-8",
    )
    return str(tmp_path / name)


def test_reads_the_ask_row_and_converts_ms_to_seconds(tmp_path):
    prefix = write(
        tmp_path,
        "locust_u50",
        "POST,/ask,1000,4,1100,1500,300,30000,800,3.3,0.01,"
        "1100,1300,1500,1700,2500,6000,9000,12500,30000,30000,30000",
    )
    s = summarize(prefix)
    assert s["requests"] == 1000 and s["failures"] == 4
    assert s["failure_rate"] == 0.004
    assert s["p50_s"] == 1.1 and s["p95_s"] == 6.0 and s["p99_s"] == 12.5
    assert s["rps"] == 3.3


def test_markdown_table_for_several_runs(tmp_path):
    a = write(tmp_path, "locust_u1", "POST,/ask,50,0,1000,1100,800,2500,800,0.5,0,"
              "1000,1050,1100,1150,1500,2000,2300,2500,2500,2500,2500")  # fmt: skip
    b = write(tmp_path, "locust_u50", "POST,/ask,1000,4,1100,1500,300,30000,800,3.3,0.01,"
              "1100,1300,1500,1700,2500,6000,9000,12500,30000,30000,30000")  # fmt: skip
    out = tmp_path / "load_test.md"
    md = main([a, b, "--out", str(out)])
    assert "| locust_u1 | 50 | 0 (0.0%) | 0.50 | 1.00 s | 2.00 s |" in md
    assert "| locust_u50 | 1000 | 4 (0.4%) | 3.30 | 1.10 s | 6.00 s | 12.50 s | 30.00 s |" in md
    assert out.read_text() == md
