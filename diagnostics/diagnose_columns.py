import pdfplumber
import collections

PDF = "data/raw/civil_code.pdf"
PAGE = 5  # 0-indexed -> page 6, matches the sample you pasted

with pdfplumber.open(PDF) as pdf:
    page = pdf.pages[PAGE]
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    print(f"page width: {page.width:.1f}, word count: {len(words)}")

    xs = sorted(w["x0"] for w in words)
    print(f"x0 range: {xs[0]:.1f} - {xs[-1]:.1f}")

    # rough histogram over 20 buckets across the page width, to spot the
    # empty gap between the two columns
    n_buckets = 20
    bucket_w = page.width / n_buckets
    counts = collections.Counter(int(x // bucket_w) for x in xs)
    for b in range(n_buckets):
        lo = b * bucket_w
        print(f"{lo:6.1f} | {'#' * counts.get(b, 0)}")

    # show a handful of words near the middle of the page, in reading
    # order top-to-bottom, so we can see which side "Article 43" and
    # "مادة" actually land on
    print("\nsample words sorted by (top, x0):")
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"]))[:40]:
        print(f"  x0={w['x0']:6.1f}  top={w['top']:6.1f}  text={w['text']!r}")