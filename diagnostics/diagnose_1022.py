import sys

sys.path.insert(0, "scripts")
import extract_corpus as ec  # noqa: E402
import pdfplumber

PDF = "data/raw/civil_code.pdf"
PAGES = [147, 148]  # 1021 and 1023 both landed here

with pdfplumber.open(PDF) as pdf:
    for page_num in PAGES:
        print("=" * 70)
        print(f"PAGE {page_num}")
        print("=" * 70)
        page = pdf.pages[page_num - 1]
        words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
        split_x = ec.find_split_x(words, page.width)

        # anything containing digit fragments that could be a mangled 1022
        candidates = [
            w
            for w in words
            if any(c.isdigit() for c in w["text"])
            or any("\u0660" <= c <= "\u0669" for c in w["text"])
        ]
        print("\nall numeric-ish words on this page:")
        for w in candidates:
            side = "LEFT(EN)" if w["x0"] < split_x else "RIGHT(AR)"
            print(f"  x0={w['x0']:6.1f} top={w['top']:6.1f} side={side:9s} text={w['text']!r}")

        en_lines, ar_lines = ec.extract_page_columns(page, page_num)
        print("\nfull reconstructed EN lines:")
        for pg, t in en_lines:
            print(f"  {t!r}")
        print("\nfull reconstructed AR lines:")
        for pg, t in ar_lines:
            print(f"  {t!r}")
