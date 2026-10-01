"""CI faithfulness gate (scripts/ragas_gate.py)."""

import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from ragas_gate import check  # noqa: E402

SUBSET = REPO / "tests" / "eval" / "eval_questions_ci_subset.json"
PARAMS = yaml.safe_load((REPO / "params.yaml").read_text(encoding="utf-8"))


def write(tmp_path, faith=0.9, params=None, corpus_md5="abc", n=None):
    ids = [q["id"] for q in json.loads(SUBSET.read_text(encoding="utf-8"))["questions"]]
    rows = [{"_id": i, "faithfulness": faith} for i in ids[:n]]
    report = {"meta": {"params": params or PARAMS, "corpus_md5": corpus_md5}, "rows": rows}
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps(report))
    lock = tmp_path / "dvc.lock"
    lock.write_text(
        yaml.safe_dump(
            {
                "stages": {
                    "extract_corpus": {
                        "outs": [{"path": "data/interim/civil_code.json", "md5": "abc"}]
                    }
                }
            }
        )
    )
    return str(rp), str(lock)


def gate(tmp_path, threshold=0.75, **kw):
    rp, lock = write(tmp_path, **kw)
    return check(rp, SUBSET, REPO / "params.yaml", lock, threshold)


def test_passes_on_fresh_good_evaluation(tmp_path):
    assert gate(tmp_path) == []


def test_fails_below_threshold(tmp_path):
    assert any("faithfulness 0.600 < 0.75" in f for f in gate(tmp_path, faith=0.6))


def test_fails_when_chunking_params_changed(tmp_path):
    stale = json.loads(json.dumps(PARAMS))
    stale["chunking"]["paragraph_split_threshold_chars"] = 123
    assert any("stale evaluation" in f and "chunking" in f for f in gate(tmp_path, params=stale))


def test_fails_when_corpus_changed(tmp_path):
    assert any("corpus changed" in f for f in gate(tmp_path, corpus_md5="different"))


def test_fails_when_judge_scored_too_few(tmp_path):
    assert any("only 5/20" in f for f in gate(tmp_path, n=5))


def test_fails_on_old_cpu_report_format():
    # the committed CPU-era file: a bare list with null scores
    out = check(
        REPO / "reports" / "ragas_results.json",
        SUBSET,
        REPO / "params.yaml",
        REPO / "dvc.lock",
        0.75,
    )
    assert out and "not a GPU evaluation report" in out[0]
