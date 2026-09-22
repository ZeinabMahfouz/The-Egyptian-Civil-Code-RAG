import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

RE_AR_PARA = re.compile(r"^\s*[\)\(]\s*[٠-٩]+\s*[\)\(]")
RE_EN_PARA = re.compile(r"^\s*\(\s*\d+\s*\)")


def split_paragraphs(text, pattern):
    lines = text.split("\n")
    paras, current = [], []
    for line in lines:
        if pattern.match(line.strip()) and current:
            paras.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        paras.append("\n".join(current).strip())
    return paras


@dataclass
class Chunk:
    chunk_id: str
    article_numbers: list       # always a list -- [147] for a normal chunk,
                                  # [54..80] for a deduped repealed range
    paragraph_index: int = None  # None for whole-article chunks
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


def make_citation(numbers):
    if len(numbers) == 1:
        return f"Egyptian Civil Code, Article {numbers[0]}"
    return f"Egyptian Civil Code, Articles {numbers[0]}-{numbers[-1]}"


def build_chunks(corpus, threshold, dedupe_repealed):
    corpus_sorted = sorted(corpus, key=lambda r: r["article_number"])
    chunks = []
    i = 0
    while i < len(corpus_sorted):
        rec = corpus_sorted[i]

        # --- repealed-range dedup: collapse a run of consecutive,
        # identically-texted repealed articles into one chunk ---
        if dedupe_repealed and rec["is_repealed"]:
            run = [rec]
            j = i + 1
            while (j < len(corpus_sorted)
                   and corpus_sorted[j]["is_repealed"]
                   and corpus_sorted[j]["text_ar"] == rec["text_ar"]
                   and corpus_sorted[j]["article_number"] == run[-1]["article_number"] + 1):
                run.append(corpus_sorted[j])
                j += 1
            if len(run) > 1:
                numbers = [r["article_number"] for r in run]
                chunks.append(Chunk(
                    chunk_id=f"{numbers[0]}-{numbers[-1]}-repealed",
                    article_numbers=numbers,
                    book=rec["book"], chapter=rec["chapter"],
                    section=rec["section"], topic=rec["topic"],
                    text_ar=rec["text_ar"], text_en=rec["text_en"],
                    is_repealed=True, source_page=rec["source_page"],
                    citation=make_citation(numbers),
                    flags=["deduped_repealed_range"],
                ))
                i = j
                continue
            # a lone repealed article (no run) falls through to normal handling

        # --- long article with genuine numbered paragraphs: split ---
        n = rec["article_number"]
        if len(rec["text_ar"]) > threshold:
            ar_paras = split_paragraphs(rec["text_ar"], RE_AR_PARA)
            if len(ar_paras) > 1:
                en_paras = split_paragraphs(rec["text_en"], RE_EN_PARA)
                aligned = len(en_paras) == len(ar_paras)
                if aligned:
                    for p_idx, (ap, ep) in enumerate(zip(ar_paras, en_paras), start=1):
                        chunks.append(Chunk(
                            chunk_id=f"{n}-p{p_idx}",
                            article_numbers=[n],
                            paragraph_index=p_idx,
                            book=rec["book"], chapter=rec["chapter"],
                            section=rec["section"], topic=rec["topic"],
                            text_ar=ap, text_en=ep,
                            is_repealed=rec["is_repealed"], source_page=rec["source_page"],
                            citation=make_citation([n]),
                        ))
                else:
                    
                    for p_idx, ap in enumerate(ar_paras, start=1):
                        chunks.append(Chunk(
                            chunk_id=f"{n}-p{p_idx}",
                            article_numbers=[n],
                            paragraph_index=p_idx,
                            book=rec["book"], chapter=rec["chapter"],
                            section=rec["section"], topic=rec["topic"],
                            text_ar=ap, text_en="",
                            is_repealed=rec["is_repealed"], source_page=rec["source_page"],
                            citation=make_citation([n]),
                            flags=["ar_only_paragraph_split"],
                        ))
                    chunks.append(Chunk(
                        chunk_id=f"{n}-en-whole",
                        article_numbers=[n],
                        paragraph_index=None,
                        book=rec["book"], chapter=rec["chapter"],
                        section=rec["section"], topic=rec["topic"],
                        text_ar="", text_en=rec["text_en"],
                        is_repealed=rec["is_repealed"], source_page=rec["source_page"],
                        citation=make_citation([n]),
                        flags=["en_whole_unaligned_source"],
                    ))
                i += 1
                continue

        # --- default: whole article = one chunk ---
        chunks.append(Chunk(
            chunk_id=str(n),
            article_numbers=[n],
            book=rec["book"], chapter=rec["chapter"],
            section=rec["section"], topic=rec["topic"],
            text_ar=rec["text_ar"], text_en=rec["text_en"],
            is_repealed=rec["is_repealed"], source_page=rec["source_page"],
            citation=make_citation([n]),
        ))
        i += 1

    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    ap.add_argument("--paragraph-split-threshold", type=int, default=None,
                     help="Overrides params.yaml's chunking.paragraph_split_threshold_chars "
                          "-- for MLflow sweeps without editing the tracked params file.")
    ap.add_argument("--no-dedupe-repealed", action="store_true",
                     help="Overrides params.yaml's chunking.dedupe_repealed_ranges=false")
    args = ap.parse_args()

    threshold = 700
    dedupe = True
    if args.params.exists() and yaml is not None:
        with open(args.params, encoding="utf-8") as f:
            p = yaml.safe_load(f) or {}
        threshold = p.get("chunking", {}).get("paragraph_split_threshold_chars", threshold)
        dedupe = p.get("chunking", {}).get("dedupe_repealed_ranges", dedupe)
    if args.paragraph_split_threshold is not None:
        threshold = args.paragraph_split_threshold
    if args.no_dedupe_repealed:
        dedupe = False

    with open(args.corpus, encoding="utf-8") as f:
        corpus = json.load(f)

    chunks = build_chunks(corpus, threshold, dedupe)

    total_articles_covered = sum(len(c.article_numbers) for c in chunks)
    split_chunks = sum(1 for c in chunks if c.paragraph_index is not None)
    dedup_chunks = sum(1 for c in chunks if "deduped_repealed_range" in c.flags)
    en_whole_chunks = sum(1 for c in chunks if "en_whole_unaligned_source" in c.flags)
    print(f"[info] {len(corpus)} articles -> {len(chunks)} chunks "
          f"(threshold={threshold} chars, dedupe_repealed={dedupe})", file=sys.stderr)
    print(f"[info] {split_chunks} paragraph-split chunks, {dedup_chunks} deduped-range chunks, "
          f"{en_whole_chunks} separate whole-English chunks (for articles where AR/EN "
          f"paragraph counts didn't align)", file=sys.stderr)
    print(f"[info] sum(len(article_numbers)) across all chunks = {total_articles_covered} "
          f"(expected to exceed {len(corpus)} when articles are paragraph-split -- "
          f"the real check is set-based, below)", file=sys.stderr)

    covered = set()
    for c in chunks:
        covered.update(c.article_numbers)
    original = {rec["article_number"] for rec in corpus}
    missing = original - covered
    unexpected = covered - original
    assert not missing and not unexpected, (
        f"Chunk article coverage mismatch. "
        f"Missing from chunks: {sorted(missing) or 'none'}. "
        f"Unexpected in chunks (not in source corpus): {sorted(unexpected) or 'none'}."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(chunks)} chunks to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()