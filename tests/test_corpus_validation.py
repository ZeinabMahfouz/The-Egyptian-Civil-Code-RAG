import json
from pathlib import Path

import pytest

CORPUS_PATH = Path(__file__).parent.parent / "data" / "interim" / "civil_code.json"

EXPECTED_MIN_ARTICLE = 1
EXPECTED_MAX_ARTICLE = 1149
MAX_REASONABLE_TEXT_LENGTH = 6000  # chars -- matches extract_corpus.py's own threshold;
# a record past this almost certainly means a failed
# article-boundary split, not a genuinely long article
KNOWN_REPEALED_RANGES = [(54, 80), (389, 417)]
MIN_EXPECTED_REPEALED_COUNT = 56  # 27 (54-80) + 29 (389-417); more is fine, fewer is not
REQUIRED_FIELDS = {
    "article_number",
    "book",
    "chapter",
    "section",
    "topic",
    "text_ar",
    "text_en",
    "is_repealed",
    "source_page",
    "citation",
}


@pytest.fixture(scope="module")
def corpus():
    assert CORPUS_PATH.exists(), (
        f"Corpus not found at {CORPUS_PATH} -- run `dvc repro` to generate it first"
    )
    with open(CORPUS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and data, "Corpus JSON is empty or not a list"
    return data


@pytest.fixture(scope="module")
def by_number(corpus):
    return {rec["article_number"]: rec for rec in corpus}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_schema_fields_present(corpus):
    for rec in corpus:
        missing = REQUIRED_FIELDS - rec.keys()
        assert not missing, f"Article {rec.get('article_number')} missing field(s): {missing}"


def test_every_record_has_citation(corpus):
    for rec in corpus:
        n = rec["article_number"]
        assert rec.get("citation", "").strip(), f"Article {n} missing citation"
        assert str(n) in rec["citation"], f"Article {n} citation doesn't reference its own number"


# ---------------------------------------------------------------------------
# Numbering integrity -- the "silent gap" failure mode
# ---------------------------------------------------------------------------
def test_article_count(corpus):
    assert len(corpus) == EXPECTED_MAX_ARTICLE, (
        f"Expected {EXPECTED_MAX_ARTICLE} articles, got {len(corpus)}"
    )


def test_no_duplicate_article_numbers(corpus):
    numbers = [rec["article_number"] for rec in corpus]
    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
    assert not duplicates, f"Duplicate article number(s): {duplicates}"


def test_article_numbers_contiguous_no_unexplained_gaps(by_number):
    numbers = sorted(by_number)
    assert numbers[0] == EXPECTED_MIN_ARTICLE, f"Numbering doesn't start at {EXPECTED_MIN_ARTICLE}"
    assert numbers[-1] == EXPECTED_MAX_ARTICLE, f"Numbering doesn't end at {EXPECTED_MAX_ARTICLE}"
    expected = set(range(EXPECTED_MIN_ARTICLE, EXPECTED_MAX_ARTICLE + 1))
    missing = expected - set(numbers)
    assert not missing, f"Unexplained gap(s) in article numbering: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Text integrity -- the "empty record" and "failed split" failure modes
# ---------------------------------------------------------------------------
def test_every_record_has_nonempty_arabic_text(corpus):
    empty = [rec["article_number"] for rec in corpus if not rec.get("text_ar", "").strip()]
    assert not empty, f"Article(s) with empty text_ar: {empty}"


def test_every_record_has_nonempty_english_text(corpus):
    empty = [rec["article_number"] for rec in corpus if not rec.get("text_en", "").strip()]
    assert not empty, f"Article(s) with empty text_en: {empty}"


def test_no_oversized_records(corpus):
    oversized = [
        (rec["article_number"], len(rec["text_ar"]))
        for rec in corpus
        if len(rec.get("text_ar", "")) > MAX_REASONABLE_TEXT_LENGTH
    ]
    assert not oversized, (
        f"Article(s) exceeding {MAX_REASONABLE_TEXT_LENGTH} chars in text_ar -- "
        f"a giant record means a failed article-boundary split, not a genuinely "
        f"long article: {oversized}"
    )


# ---------------------------------------------------------------------------
# Repealed-article flagging -- the "silently wrong legal status" failure mode
# ---------------------------------------------------------------------------
def test_known_repealed_ranges_are_flagged(by_number):
    for lo, hi in KNOWN_REPEALED_RANGES:
        for n in range(lo, hi + 1):
            assert n in by_number, (
                f"Article {n} (in known repealed range {lo}-{hi}) is missing entirely"
            )
            assert by_number[n]["is_repealed"] is True, (
                f"Article {n} falls in known repealed range {lo}-{hi} but is_repealed=False -- "
                f"a downstream retriever would surface this as live law"
            )


def test_repealed_count_is_plausible(corpus):
    repealed_count = sum(1 for rec in corpus if rec.get("is_repealed"))
    assert repealed_count >= MIN_EXPECTED_REPEALED_COUNT, (
        f"Only {repealed_count} articles flagged repealed, expected at least "
        f"{MIN_EXPECTED_REPEALED_COUNT} (54-80 + 389-417 at minimum)"
    )


def test_non_repealed_articles_are_not_flagged_repealed(corpus):
    # Sanity check in the other direction: is_repealed shouldn't be a blanket
    # True or a coin flip. Every flagged article should actually contain a
    # repeal-related term in its own text.
    import re

    repeal_pattern = re.compile(r"ملغاة|ألغيت|ألغي|repealed|abolished", re.I)
    for rec in corpus:
        if rec.get("is_repealed"):
            combined = rec.get("text_ar", "") + rec.get("text_en", "")
            assert repeal_pattern.search(combined) or "manually_patched" in rec.get("flags", []), (
                f"Article {rec['article_number']} flagged is_repealed=True but its text "
                f"contains no repeal-related term -- possible false positive"
            )
