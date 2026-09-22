import re
import sys

sys.path.insert(0, "scripts")
import extract_corpus as ec  # noqa: E402
import pdfplumber

PDF = "data/raw/civil_code.pdf"
TARGETS = [277, 452, 714, 746, 898, 908, 1090]

with pdfplumber.open(PDF) as pdf:
    en_stream, ar_stream = [], []
    for i, page in enumerate(pdf.pages, start=1):
        en_lines, ar_lines = ec.extract_page_columns(page, i)
        en_stream.extend(en_lines)
        ar_stream.extend(ar_lines)

for n in TARGETS:
    print("=" * 70)
    print(f"ARTICLE {n}")
    print("=" * 70)

    # does a line CONTAINING "Article n" exist anywhere in the EN stream,
    # even if it doesn't match the strict end-anchored header regex?
    pat = re.compile(rf"\bArticle\s+{n}\b")
    hits = [(idx, pg, t) for idx, (pg, t) in enumerate(en_stream) if pat.search(t)]
    if not hits:
        print("  NOT FOUND anywhere in EN stream (not even as substring)")
    for idx, pg, t in hits:
        matches_header_regex = bool(ec.RE_EN_ARTICLE.match(t))
        print(f"  EN page {pg}: {t!r}  [matches header regex: {matches_header_regex}]")
        # show a couple lines of context
        for cidx in range(max(0, idx - 1), min(len(en_stream), idx + 2)):
            if cidx != idx:
                cpg, ct = en_stream[cidx]
                print(f"      ctx p{cpg}: {ct!r}")

    # cross-reference: confirm it DOES exist correctly on the Arabic side
    ar_num_str = str(n).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
    ar_hits = [(pg, t) for pg, t in ar_stream if ar_num_str in t and "مادة" in t]
    for pg, t in ar_hits[:2]:
        print(f"  AR page {pg}: {t!r}")
    print()
