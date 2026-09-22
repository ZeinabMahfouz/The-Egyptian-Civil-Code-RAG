import json

with open("data/interim/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

print("=" * 70)
print("PARAGRAPH-SPLIT CHUNKS (all of them)")
print("=" * 70)
split_chunks = [c for c in chunks if c["paragraph_index"] is not None]
# group by article number for readability
by_article = {}
for c in split_chunks:
    by_article.setdefault(c["article_numbers"][0], []).append(c)

for article_n, parts in sorted(by_article.items()):
    print(f"\n--- Article {article_n} ({len(parts)} paragraph chunks) ---")
    for c in parts:
        print(f"  chunk_id={c['chunk_id']}  flags={c['flags']}")
        print(f"    text_ar: {c['text_ar'][:80]!r}")
        print(f"    text_en: {c['text_en'][:80]!r}")

print("\n" + "=" * 70)
print("DEDUPED REPEALED-RANGE CHUNKS")
print("=" * 70)
dedup_chunks = [c for c in chunks if "deduped_repealed_range" in c["flags"]]
for c in dedup_chunks:
    print(f"\nchunk_id={c['chunk_id']}")
    print(
        f"  covers {len(c['article_numbers'])} articles: "
        f"{c['article_numbers'][:5]}...{c['article_numbers'][-3:]}"
    )
    print(f"  citation: {c['citation']}")
    print(f"  text_ar: {c['text_ar']!r}")

print("\n" + "=" * 70)
print("A NORMAL WHOLE-ARTICLE CHUNK, FOR COMPARISON")
print("=" * 70)
normal = next(c for c in chunks if c["article_numbers"] == [147])
print(json.dumps(normal, ensure_ascii=False, indent=2))
