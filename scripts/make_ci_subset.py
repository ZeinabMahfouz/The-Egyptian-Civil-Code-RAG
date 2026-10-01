"""Builds tests/eval/eval_questions_ci_subset.json: a fixed, stratified
20-question subset of the 54-question eval set, for the MLflow sweep and the
CI faithfulness gate (the course checklist's "20-question test set").

Deterministic: the first N questions of each (category, language) stratum,
in file order -- rerunning this always yields the same 20.
"""

import json
from pathlib import Path

SRC = Path("tests/eval/eval_questions.json")
OUT = Path("tests/eval/eval_questions_ci_subset.json")

PER_STRATUM = {
    ("substantive", "en"): 4,
    ("substantive", "ar"): 4,
    ("article_lookup", "en"): 2,
    ("article_lookup", "ar"): 2,
    ("repealed_status", "en"): 2,
    ("repealed_status", "ar"): 2,
    ("out_of_corpus", "en"): 2,
    ("out_of_corpus", "ar"): 2,
}


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    taken = {k: 0 for k in PER_STRATUM}
    subset = []
    for q in data["questions"]:
        key = (q["category"], q["language"])
        if taken.get(key, 0) < PER_STRATUM.get(key, 0):
            subset.append(q)
            taken[key] += 1
    assert len(subset) == sum(PER_STRATUM.values()), taken
    OUT.write_text(
        json.dumps(
            {
                "description": "Fixed 20-question stratified subset of eval_questions.json "
                "(see scripts/make_ci_subset.py). Used by the MLflow chunking sweep and "
                "the CI faithfulness gate.",
                "questions": subset,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(subset)} questions to {OUT}")


if __name__ == "__main__":
    main()
