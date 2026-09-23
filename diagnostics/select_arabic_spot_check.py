"""
Generate a 20-article Arabic-text spot-check sample and a fillable
report, per the course checklist's "Arabic text spot-checked on 20
articles" requirement.

Selection isn't random padding to hit 20 -- it deliberately includes
every category of edge case this project's debugging actually
surfaced real problems in, since that's where a spot-check earns its
keep, plus a reproducible (seeded) stratified sample across the rest
of the document for genuine coverage.

Usage:
    python select_arabic_spot_check.py
    -> writes reports/arabic_spot_check.md
"""

import json
import random
from pathlib import Path

CORPUS_PATH = Path("data/interim/civil_code.json")
OUT_PATH = Path("reports/arabic_spot_check.md")
SEED = 42  # fixed -- same sample every time this is regenerated

# Forced inclusions: every category of real, previously-diagnosed edge
# case in this corpus, so the spot-check isn't just checking "normal"
# articles that were never actually at risk.
FORCED = {
    1: "first article in the corpus (also the promulgation-law duplicate-numbering edge case)",
    1149: "last article in the corpus",
    1022: "source PDF has a genuinely empty Arabic cell for this article -- manually patched",
    238: "paragraph-split article (Arabic split by numbered clause, English kept whole)",
    658: "paragraph-split article (4 clauses, Arabic split, English kept whole)",
    60: "inside the 54-80 repealed range (deduped placeholder text)",
    400: "inside the 389-417 repealed range (headerless repeal block, no article marker in source)",
    147: "the running example used throughout this project's own schema design",
}


def main():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)
    by_number = {r["article_number"]: r for r in corpus}

    sample_numbers = set(FORCED)
    remaining_slots = 20 - len(sample_numbers)

    rng = random.Random(SEED)
    candidates = [n for n in by_number if n not in sample_numbers]
    # stratified: split the remaining number range into buckets, one
    # pick per bucket, so the sample spans the whole document instead
    # of clustering wherever randomness happens to land
    candidates.sort()
    bucket_size = max(1, len(candidates) // remaining_slots)
    for i in range(remaining_slots):
        bucket = candidates[i * bucket_size : (i + 1) * bucket_size]
        if bucket:
            sample_numbers.add(rng.choice(bucket))

    sample = sorted(sample_numbers)[:20]

    lines = [
        "# Arabic text spot-check: 20 articles",
        "",
        "Each article's `text_ar` below needs to be checked against the actual "
        "source PDF page it cites. Open `data/raw/civil_code.pdf` to the noted "
        "page, compare, and tick PASS or FAIL with a note.",
        "",
        "| # | Article | Page | Reason for inclusion | text_ar preview | Result |",
        "|---|---------|------|----------------------|------------------|--------|",
    ]
    for i, n in enumerate(sample, start=1):
        rec = by_number[n]
        reason = FORCED.get(n, "stratified random sample (seed=42) across the full article range")
        preview = rec["text_ar"][:60].replace("|", "-").replace("\n", " ")
        lines.append(
            f"| {i} | {n} | {rec['source_page']} | {reason} | {preview}... | [ ] PASS / [ ] FAIL |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "(add any discrepancies found here, with article number and what was wrong)",
        "",
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(sample)} articles to {OUT_PATH}")


if __name__ == "__main__":
    main()
