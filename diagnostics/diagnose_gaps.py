import re
import sys

sys.path.insert(0, "scripts")
import extract_corpus as ec  # noqa: E402
import pdfplumber

PDF = "data/raw/civil_code.pdf"

with pdfplumber.open(PDF) as pdf:
    # --- 1: page 1 duplicate words -------------------------------------
    print("=" * 70)
    print("CHECK 1: page 1 -- duplicate word objects?")
    print("=" * 70)
    page1 = pdf.pages[0]
    words = page1.extract_words(use_text_flow=False, keep_blank_chars=False)
    print(f"total words on page 1: {len(words)}")
    seen, dupes = {}, []
    for w in words:
        key = (w["text"], round(w["x0"], 1), round(w["top"], 1))
        seen[key] = seen.get(key, 0) + 1
    for key, count in seen.items():
        if count > 1:
            dupes.append((key, count))
    print(f"exact-duplicate (text,x0,top) tuples: {len(dupes)}")
    for key, count in dupes[:20]:
        print(f"  {count}x  {key}")
    en1, ar1 = ec.extract_page_columns(page1, 1)
    print("\nreconstructed EN lines on page 1:")
    for pg, t in en1:
        print(f"  {t!r}")
    print("reconstructed AR lines on page 1:")
    for pg, t in ar1:
        print(f"  {t!r}")

    # --- 2: article 54 header split -------------------------------------
    print("\n" + "=" * 70)
    print("CHECK 2: where did 'مادة' and '٥٤' land? (scanning pages 5-9)")
    print("=" * 70)
    for i in range(5, 10):
        page = pdf.pages[i]
        _, ar_lines = ec.extract_page_columns(page, i + 1)
        for pg, t in ar_lines:
            if "مادة" in t or "٥٤" in t or "ألغيت" in t or "ملغاة" in t:
                print(f"  page {pg}: {t!r}")

    # --- 3: article 1022 ---------------------------------------------
    print("\n" + "=" * 70)
    print("CHECK 3: locating 1021/1022/1023 in both streams")
    print("=" * 70)
    en_stream, ar_stream = [], []
    for i, page in enumerate(pdf.pages, start=1):
        en_lines, ar_lines = ec.extract_page_columns(page, i)
        en_stream.extend(en_lines)
        ar_stream.extend(ar_lines)

    pat_en = re.compile(r"\b102[123]\b")
    print("EN stream matches:")
    for idx, (pg, t) in enumerate(en_stream):
        if pat_en.search(t):
            ctx = en_stream[max(0, idx - 2) : idx + 3]
            print(f"  page {pg}: {t!r}")
            for cpg, ct in ctx:
                print(f"      ctx p{cpg}: {ct!r}")

    ar_targets = ["١٠٢١", "١٠٢٢", "١٠٢٣"]
    print("AR stream matches:")
    for idx, (pg, t) in enumerate(ar_stream):
        if any(x in t for x in ar_targets):
            ctx = ar_stream[max(0, idx - 2) : idx + 3]
            print(f"  page {pg}: {t!r}")
            for cpg, ct in ctx:
                print(f"      ctx p{cpg}: {ct!r}")
