import json
import statistics as stats
from pathlib import Path

CORPUS_PATH = Path("data/interim/civil_code.json")

with open(CORPUS_PATH, encoding="utf-8") as f:
    corpus = json.load(f)

ar_lens = [len(rec["text_ar"]) for rec in corpus]
en_lens = [len(rec["text_en"]) for rec in corpus]

def report(name, lens):
    print(f"\n--- {name} (chars) ---")
    print(f"  min:    {min(lens)}")
    print(f"  max:    {max(lens)}")
    print(f"  mean:   {stats.mean(lens):.0f}")
    print(f"  median: {stats.median(lens):.0f}")
    print(f"  stdev:  {stats.stdev(lens):.0f}")
    # rough word count proxy: chars / 5.5 for Arabic, chars / 5 for English
    for pct in [50, 75, 90, 95, 99]:
        sorted_lens = sorted(lens)
        idx = int(len(sorted_lens) * pct / 100)
        print(f"  p{pct}:    {sorted_lens[min(idx, len(sorted_lens)-1)]}")

report("text_ar", ar_lens)
report("text_en", en_lens)

# flag outliers worth looking at directly
print("\n--- shortest 10 articles (text_ar) ---")
for rec in sorted(corpus, key=lambda r: len(r["text_ar"]))[:10]:
    print(f"  Article {rec['article_number']:4d}: {len(rec['text_ar']):4d} chars  "
          f"repealed={rec['is_repealed']}  {rec['text_ar'][:50]!r}")

print("\n--- longest 10 articles (text_ar) ---")
for rec in sorted(corpus, key=lambda r: -len(r["text_ar"]))[:10]:
    print(f"  Article {rec['article_number']:4d}: {len(rec['text_ar']):4d} chars  "
          f"repealed={rec['is_repealed']}  {rec['text_ar'][:50]!r}")

# how many articles are one-liners (likely repealed placeholders or genuinely tiny)
tiny = [r for r in corpus if len(r["text_ar"]) < 30]
print(f"\narticles under 30 chars: {len(tiny)} "
      f"({sum(1 for r in tiny if r['is_repealed'])} of them repealed)")