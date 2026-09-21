import sys
sys.path.insert(0, "scripts")
import extract_corpus as ec  # noqa: E402
import pdfplumber

PDF = "data/raw/civil_code.pdf"
PAGES = {36: 277, 59: 452}  # 1-indexed page -> article number we're hunting

with pdfplumber.open(PDF) as pdf:
    for page_num, article_n in PAGES.items():
        print("=" * 70)
        print(f"PAGE {page_num} (looking for Article {article_n})")
        print("=" * 70)
        page = pdf.pages[page_num - 1]
        words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
        split_x = ec.find_split_x(words, page.width)
        print(f"page width: {page.width:.1f}, split_x: {split_x:.1f}, word count: {len(words)}")

        target_digits = str(article_n)
        hits = [w for w in words if "Article" in w["text"] or target_digits in w["text"]]
        print(f"\nwords containing 'Article' or '{target_digits}':")
        for w in hits:
            side = "LEFT(EN)" if w["x0"] < split_x else "RIGHT(AR)"
            print(f"  x0={w['x0']:6.1f} top={w['top']:6.1f} side={side:9s} text={w['text']!r}")

        # full reconstructed columns for this page, for context
        en_lines, ar_lines = ec.extract_page_columns(page, page_num)
        print(f"\nreconstructed EN lines ({len(en_lines)}):")
        for pg, t in en_lines:
            print(f"  {t!r}")