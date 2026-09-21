#!/usr/bin/env python3
"""
Extract the Egyptian Civil Code (bilingual AR/EN PDF, side-by-side
two-column layout) into structured JSON, one record per article.

Why pdfplumber and not pdftotext -layout: -layout preserves the PHYSICAL
page layout, which for this PDF means English and Arabic text from the
same row end up on the same output line, interleaved. That defeats a
line-anchored regex parser. pdfplumber gives per-word bounding boxes, so
we split each page into two columns by x-position, reconstruct each
column as its own clean line stream, and align English/Arabic by
article NUMBER rather than by physical line adjacency.

Requires: pip install pdfplumber
"""
import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pdfplumber

EXPECTED_MAX_ARTICLE = 1149

# ---------------------------------------------------------------------------
# Numeral / script helpers
# ---------------------------------------------------------------------------
AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EN_DIGITS = "0123456789"
AR2EN = str.maketrans(AR_DIGITS, EN_DIGITS)

# Core Arabic letter block. Deliberately excludes \u0660-\u0669 (Arabic-Indic
# digits), which are weak-LTR and must NOT be character-reversed -- doing so
# would silently turn ٤٣ (43) into ٣٤ (34).
RE_ARABIC_LETTER = re.compile(r"[\u0621-\u064A]")


def ar_to_int(s: str):
    digits = re.sub(r"\D", "", s.translate(AR2EN))
    return int(digits) if digits else None


def reversed_int(n: int) -> int:
    return int(str(n)[::-1])


def fix_arabic_word(token: str) -> str:
    """Reverse character order for RTL-letter tokens only; leave
    digit/punctuation-only tokens untouched (see module docstring)."""
    if RE_ARABIC_LETTER.search(token):
        return token[::-1]
    return token


# ---------------------------------------------------------------------------
# Regexes for the now-clean, single-language line streams
# ---------------------------------------------------------------------------
RE_AR_ARTICLE = re.compile(r"^\s*مادة\b[\s\)\(\.,،]*([٠-٩]+)[\s\)\(\.,،]*$")
RE_EN_ARTICLE = re.compile(r"^\s*Article\s*(\d+)\s*$")
# Fallback for the case where the header row got fused with the article's
# first body line (e.g. "Article 714 The mandate comes to an end by the").
# Only ever tried for numbers the Arabic stream already confirmed exist
# and the strict regex above didn't find -- see parse_english_stream.
RE_EN_ARTICLE_LOOSE = re.compile(r"^\s*A?rticle\s*(\d+)\b\s*(.*)$")

RE_REPEAL_RANGE = re.compile(r"المواد\s+من\s+([٠-٩]+)\s+إلى\s+([٠-٩]+)")
RE_REPEAL_WORD = re.compile(r"ملغاة|ألغيت|ألغي|repealed|abolished", re.I)
RE_BIS = re.compile(r"مكرر")

_MAXHEAD = 60
# Optional leading "ال" (definite article) -- most headings use it
# ("الباب الأول"), but the unnumbered preliminary heading is written
# without it ("باب تمهيدي", not "الباب تمهيدي") since it's grammatically
# indefinite. Without this, that heading was invisible to the parser,
# which is exactly what let the Law of Promulgation's own "مادة ١"/"مادة ٢"
# (a different, 2-article enacting decree that precedes the real code
# text) get parsed as if they were the Civil Code's real Article 1/2.
RE_PART = re.compile(r"^\s*(?:ال)?قسم\s")
RE_BOOK = re.compile(r"^\s*(?:ال)?كتاب\s")
RE_CHAPTER = re.compile(r"^\s*(?:ال)?باب\s")
RE_SECTION = re.compile(r"^\s*(?:ال)?فصل\s")


def is_heading(regex, line: str) -> bool:
    return len(line) <= _MAXHEAD and bool(regex.match(line))


def looks_like_topic(line: str) -> bool:
    """See extractor v1 for the full rationale -- unchanged limitation:
    only trusted right after a formal heading, not mid-section."""
    line = line.strip()
    if not line or len(line) > _MAXHEAD:
        return False
    if RE_AR_ARTICLE.match(line) or RE_EN_ARTICLE.match(line):
        return False
    if any(is_heading(r, line) for r in (RE_PART, RE_BOOK, RE_CHAPTER, RE_SECTION)):
        return False
    return bool(re.match(r"^[-–]?\s*[٠-٩]+\s*[-–.]", line))


# ---------------------------------------------------------------------------
# Column detection + per-page column extraction
# ---------------------------------------------------------------------------
def find_split_x(words, page_width, band=(0.15, 0.85), resolution=1.0):
    """Find the widest empty x-gap within the central band of the page --
    that's the boundary between the English and Arabic columns."""
    n_bins = int(page_width // resolution) + 1
    occupied = [False] * n_bins
    for w in words:
        a = max(0, int(w["x0"] // resolution))
        b = min(n_bins - 1, int(w["x1"] // resolution))
        for i in range(a, b + 1):
            occupied[i] = True

    lo_bound, hi_bound = band[0] * page_width, band[1] * page_width
    best_start, best_len = None, 0
    cur_start, cur_len = None, 0
    for i in range(n_bins):
        x = i * resolution
        in_band = lo_bound <= x <= hi_bound
        if in_band and not occupied[i]:
            cur_start = i if cur_start is None else cur_start
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
            cur_start, cur_len = None, 0
    if cur_len > best_len:
        best_len, best_start = cur_len, cur_start
    if best_start is None:
        return page_width / 2  # fallback -- should be rare
    return (best_start + best_len / 2) * resolution


def group_into_lines(words, tol=3.0):
    """Cluster words into rows by vertical (top) position."""
    words_sorted = sorted(words, key=lambda w: w["top"])
    lines, current, anchor = [], [], None
    for w in words_sorted:
        if anchor is None or abs(w["top"] - anchor) <= tol:
            current.append(w)
            anchor = w["top"] if anchor is None else anchor
        else:
            lines.append(current)
            current, anchor = [w], w["top"]
    if current:
        lines.append(current)
    return lines


def extract_page_columns(page, page_num: int):
    """Returns (en_lines, ar_lines): each a list of (page_num, text)."""
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    if not words:
        return [], []
    split_x = find_split_x(words, page.width)

    left = [w for w in words if w["x0"] < split_x]
    right = [w for w in words if w["x0"] >= split_x]

    en_lines = []
    for row in group_into_lines(left):
        row.sort(key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in row).strip()
        if text:
            en_lines.append((page_num, text))

    ar_lines = []
    for row in group_into_lines(right):
        row.sort(key=lambda w: -w["x0"])  # rightmost first = logical first
        text = " ".join(fix_arabic_word(w["text"]) for w in row).strip()
        if text:
            ar_lines.append((page_num, text))

    return en_lines, ar_lines


# ---------------------------------------------------------------------------
# Single-column stream parsers
# ---------------------------------------------------------------------------
@dataclass
class ArArticle:
    text: str
    book: str
    chapter: str
    section: str
    topic: str
    page: int


def parse_arabic_stream(lines, topic_overrides, errors, warnings, repeal_ranges_found, repeal_placeholders):
    """repeal_placeholders collects (lo, hi, notice_text, book, chapter,
    section, topic, page) tuples. Placeholder insertion into BOTH the
    Arabic and English article dicts happens later, once, after both
    streams are parsed -- doing it here only ever touched the Arabic
    dict, which is why 55-80 and 389-417 never reached text_en."""
    articles = {}
    cur_part = cur_book = cur_chapter = cur_section = cur_topic = ""
    mode = "seeking"
    num, page, buf = None, None, []
    # Nothing before the first real structural heading counts as a Civil
    # Code article -- see module note on the Law of Promulgation's own
    # مادة ١/٢ preceding the real code text on page 1.
    seen_heading = False

    def current_book_field():
        return f"{cur_part} - {cur_book}".strip(" -") if cur_part else cur_book

    def flush():
        nonlocal num, page, buf
        if num is None:
            return
        text = "\n".join(buf).strip()
        if num in articles:
            errors.append(f"[duplicate-ar] article {num} reappears at page {page}")
        articles[num] = ArArticle(
            text=text, book=current_book_field(), chapter=cur_chapter,
            section=cur_section, topic=topic_overrides.get(str(num), cur_topic), page=page,
        )
        m = RE_REPEAL_RANGE.search(text)
        if m and RE_REPEAL_WORD.search(text):
            lo, hi = ar_to_int(m.group(1)), ar_to_int(m.group(2))
            repeal_ranges_found.append(f"range {lo}-{hi} in body of article {num}, page {page}")
            repeal_placeholders.append((lo, hi, text, current_book_field(), cur_chapter,
                                         cur_section, cur_topic, page))
        num, page, buf = None, None, []

    for pg, t in lines:
        # Headerless repeal block, e.g. Article 54's case: the range
        # statement can share a line with the repeal keyword or not, and
        # can land mid-article (not just in "seeking" mode) -- flush
        # whatever's pending first, same as a heading does.
        m = RE_REPEAL_RANGE.search(t)
        if m and RE_REPEAL_WORD.search(t):
            if mode != "seeking":
                flush()
                mode = "seeking"
            lo, hi = ar_to_int(m.group(1)), ar_to_int(m.group(2))
            repeal_ranges_found.append(f"headerless range {lo}-{hi} at page {pg}: '{t}'")
            repeal_placeholders.append((lo, hi, t, current_book_field(), cur_chapter,
                                         cur_section, cur_topic, pg))
            continue

        am = RE_AR_ARTICLE.match(t)
        if am:
            if not seen_heading:
                # Front matter (Law of Promulgation etc.) before any real
                # structural heading -- not a Civil Code article, skip it
                # entirely rather than starting a spurious record.
                continue
            if mode != "seeking":
                flush()
            if RE_BIS.search(t):
                warnings.append(f"[bis-article] 'مكرر' on page {pg}: '{t}' -- review by hand")
            num, page, mode = ar_to_int(am.group(1)), pg, "in_article"
            buf = []
            continue

        if is_heading(RE_PART, t):
            if mode != "seeking":
                flush(); mode = "seeking"
            seen_heading = True
            cur_part, cur_book, cur_chapter, cur_section, cur_topic = t, "", "", "", ""
            continue
        if is_heading(RE_BOOK, t):
            if mode != "seeking":
                flush(); mode = "seeking"
            seen_heading = True
            cur_book, cur_chapter, cur_section, cur_topic = t, "", "", ""
            continue
        if is_heading(RE_CHAPTER, t):
            if mode != "seeking":
                flush(); mode = "seeking"
            seen_heading = True
            cur_chapter, cur_section, cur_topic = t, "", ""
            continue
        if is_heading(RE_SECTION, t):
            if mode != "seeking":
                flush(); mode = "seeking"
            seen_heading = True
            cur_section, cur_topic = t, ""
            continue
        if mode == "seeking" and looks_like_topic(t):
            cur_topic = t
            continue
        if mode == "in_article":
            buf.append(t)

    if mode != "seeking":
        flush()
    return articles


def parse_english_stream(lines, errors, needed_pages):
    """needed_pages: {article_number: page} for numbers the Arabic stream
    confirmed exist, with the page its header was found on. Strict pass
    first (avoids inline-cross-reference false positives like the
    901/993 case). Permissive fallback only for numbers still missing
    after that, AND only on the same page as the Arabic header -- an
    inline cross-reference to "Article N" can appear on any page in the
    document; a genuine header for N can only appear on the page the
    Arabic side already confirmed. This is what makes the fallback safe
    without needing the strict end-anchor that caused the false
    negatives in the first place."""
    articles = {}
    num, page, buf = None, None, []

    def flush():
        nonlocal num, page, buf
        if num is None:
            return
        if num in articles:
            errors.append(f"[duplicate-en] article {num} reappears at page {page}")
        articles[num] = "\n".join(buf).strip()
        num, page, buf = None, None, []

    for pg, t in lines:
        sm = RE_EN_ARTICLE.match(t)
        if sm:
            flush()
            num, page, buf = int(sm.group(1)), pg, []
            continue
        lm = RE_EN_ARTICLE_LOOSE.match(t)
        if lm:
            n = int(lm.group(1))
            if n in needed_pages and n not in articles and needed_pages[n] == pg:
                flush()
                num, page = n, pg
                remainder = lm.group(2).strip()
                buf = [remainder] if remainder else []
                continue
        if num is not None:
            buf.append(t)
    flush()
    return articles


# ---------------------------------------------------------------------------
@dataclass
class ArticleRecord:
    article_number: int
    book: str = ""
    chapter: str = ""
    section: str = ""
    topic: str = ""
    text_ar: str = ""
    text_en: str = ""
    is_repealed: bool = False
    source_page: int = 0
    citation: str = ""
    flags: list = field(default_factory=list)


def merge(ar_articles, en_articles, errors):
    records = {}
    for n in sorted(set(ar_articles) | set(en_articles)):
        if n not in ar_articles:
            errors.append(f"[missing-arabic] article {n} found in English stream only")
            continue
        if n not in en_articles:
            errors.append(f"[missing-english] article {n} found in Arabic stream only")
            continue
        ar, en = ar_articles[n], en_articles[n]
        repealed = bool(RE_REPEAL_WORD.search(ar.text)) or bool(RE_REPEAL_WORD.search(en))
        records[n] = ArticleRecord(
            article_number=n, book=ar.book, chapter=ar.chapter, section=ar.section,
            topic=ar.topic, text_ar=ar.text, text_en=en, is_repealed=repealed,
            source_page=ar.page, citation=f"Egyptian Civil Code, Article {n}",
        )
    return records


def validate(records, errors, warnings, repeal_ranges_found, known_gaps=None):
    known_gaps = known_gaps or set()
    nums = sorted(records)
    if not nums:
        return ["No articles extracted at all."], []
    lo, hi = nums[0], nums[-1]
    full_range = set(range(lo, hi + 1))
    missing = full_range - set(nums)
    unexplained_missing = missing - known_gaps
    if unexplained_missing:
        errors.append(f"Gap(s) in numbering: {sorted(unexplained_missing)}")
    if known_gaps & missing:
        warnings.append(f"Known/acknowledged gaps (via --known-gaps, NOT validated against "
                         f"an authoritative source by this script): {sorted(known_gaps & missing)}")
    expected_total = EXPECTED_MAX_ARTICLE - len(known_gaps & missing)
    if lo != 1 or hi != EXPECTED_MAX_ARTICLE or len(records) != expected_total:
        errors.append(f"Expected {expected_total} articles ({EXPECTED_MAX_ARTICLE} minus "
                       f"{len(known_gaps & missing)} acknowledged gap(s)) spanning "
                       f"1-{EXPECTED_MAX_ARTICLE}; got {len(records)} spanning {lo}-{hi}")
    for n, rec in records.items():
        if not rec.text_ar.strip():
            errors.append(f"Article {n}: empty text_ar")
        if not rec.text_en.strip():
            errors.append(f"Article {n}: empty text_en")
        if len(rec.text_ar) > 6000:
            warnings.append(f"Article {n}: text_ar suspiciously long ({len(rec.text_ar)} chars)")
    repealed = sum(1 for r in records.values() if r.is_repealed)
    warnings.append(f"{repealed} articles flagged repealed (expect >= 56)")
    if repeal_ranges_found:
        warnings.append(f"RE_REPEAL_RANGE matched {len(repeal_ranges_found)} time(s):")
        warnings.extend(f"  - {r}" for r in repeal_ranges_found)
    else:
        errors.append("RE_REPEAL_RANGE matched zero times -- expected >= 2. "
                       "Repeal detection is likely broken.")
    return errors, warnings


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--topic-overrides", type=Path, default=None)
    ap.add_argument("--manual-patches", type=Path, default=None,
                     help="JSON: {\"article_number\": {\"text_ar\": \"...\", \"text_en\": \"...\"}} "
                          "-- inserts or overwrites specific records after parsing, for content "
                          "confirmed real but not automatically recoverable (e.g. a genuinely "
                          "missing extracted token). Use sparingly and note the source.")
    ap.add_argument("--known-gaps", type=str, default="",
                     help="Comma-separated article numbers to treat as confirmed-absent "
                          "rather than an extraction failure (e.g. '1022'). Only pass "
                          "numbers you've verified against an authoritative source -- "
                          "this script does not verify them itself.")
    ap.add_argument("--print-outline", action="store_true")
    args = ap.parse_args()
    known_gaps = {int(x) for x in args.known_gaps.split(",") if x.strip()}

    overrides = {}
    if args.topic_overrides and args.topic_overrides.exists():
        overrides = json.loads(args.topic_overrides.read_text(encoding="utf-8"))

    en_stream, ar_stream = [], []
    with pdfplumber.open(args.pdf) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            en_lines, ar_lines = extract_page_columns(page, i)
            en_stream.extend(en_lines)
            ar_stream.extend(ar_lines)
    print(f"[info] {len(pdf.pages)} pages, {len(en_stream)} EN lines, "
          f"{len(ar_stream)} AR lines", file=sys.stderr)

    errors, warnings, repeal_ranges_found, repeal_placeholders = [], [], [], []
    ar_articles = parse_arabic_stream(ar_stream, overrides, errors, warnings,
                                       repeal_ranges_found, repeal_placeholders)
    en_articles = parse_english_stream(
        en_stream, errors,
        needed_pages={n: a.page for n, a in ar_articles.items()},
    )

    # Fill repealed-range placeholders into BOTH dicts. Reuses the Arabic
    # notice text for text_en too when no independent English notice was
    # captured -- same precedent as the original headerless-block design
    # (a cross-language placeholder is far better than a hard gap).
    for lo, hi, notice, book, chapter, section, topic, pg in repeal_placeholders:
        for k in range(lo, hi + 1):
            if k not in ar_articles:
                ar_articles[k] = ArArticle(text=notice, book=book, chapter=chapter,
                                            section=section, topic=topic, page=pg)
            if k not in en_articles:
                en_articles[k] = notice

    records = merge(ar_articles, en_articles, errors)

    if args.manual_patches and args.manual_patches.exists():
        patches = json.loads(args.manual_patches.read_text(encoding="utf-8"))
        patched_numbers = set()
        for n_str, patch in patches.items():
            n = int(n_str)
            patched_numbers.add(n)
            if n in records:
                for k, v in patch.items():
                    setattr(records[n], k, v)
                records[n].flags.append("manually_patched")
            else:
                # neighbor gives reasonable book/chapter/section defaults
                # for a record that couldn't be built from either stream
                neighbor = records.get(n - 1) or records.get(n + 1)
                records[n] = ArticleRecord(
                    article_number=n,
                    book=neighbor.book if neighbor else "",
                    chapter=neighbor.chapter if neighbor else "",
                    section=neighbor.section if neighbor else "",
                    topic=neighbor.topic if neighbor else "",
                    text_ar=patch.get("text_ar", ""),
                    text_en=patch.get("text_en", ""),
                    is_repealed=patch.get("is_repealed", False),
                    source_page=neighbor.source_page if neighbor else 0,
                    citation=f"Egyptian Civil Code, Article {n}",
                    flags=["manually_patched"],
                )
        # A patch can resolve exactly the condition an earlier
        # missing-arabic/missing-english error was reporting -- once
        # that number is in `records`, the error is stale. Downgrade
        # it to a visible, auditable warning rather than either
        # silently dropping it or leaving a false failure standing.
        if patched_numbers:
            remaining_errors, downgraded = [], []
            for e in errors:
                m = re.match(r"\[(missing-arabic|missing-english)\] article (\d+)", e)
                if m and int(m.group(2)) in patched_numbers:
                    downgraded.append(f"[resolved-by-manual-patch] {e}")
                    continue
                remaining_errors.append(e)
            errors = remaining_errors
            warnings.extend(downgraded)
    errors, warnings = validate(records, errors, warnings, repeal_ranges_found, known_gaps)

    print("--- warnings (non-blocking) ---", file=sys.stderr)
    for w in warnings:
        print(f"[warn] {w}", file=sys.stderr)
    print("--- errors (blocking) ---", file=sys.stderr)
    for e in errors:
        print(f"[error] {e}", file=sys.stderr)

    if args.print_outline:
        for n in sorted(records):
            r = records[n]
            print(f"{n:4d} | {r.book[:28]:28s} | {r.chapter[:22]:22s} | "
                  f"{r.section[:22]:22s} | {r.topic[:22]:22s} | repealed={r.is_repealed}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump([asdict(records[n]) for n in sorted(records)], f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} articles to {args.out}", file=sys.stderr)

    if errors:
        print(f"VALIDATION FAILED -- {len(errors)} blocking error(s)", file=sys.stderr)
        sys.exit(1)
    print("VALIDATION PASSED", file=sys.stderr)


if __name__ == "__main__":
    main()