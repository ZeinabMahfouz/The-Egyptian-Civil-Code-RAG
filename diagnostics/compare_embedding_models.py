import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/interim/chunks.json")

CANDIDATES = {
    "BAAI/bge-m3": {"query_prefix": "", "passage_prefix": ""},
    "intfloat/multilingual-e5-large": {"query_prefix": "query: ", "passage_prefix": "passage: "},
}

TEST_QUERIES = [
    {
        "query": "ماذا يحدث إذا جعلت ظروف استثنائية غير متوقعة تنفيذ العقد مرهقاً للمدين؟",
        "expected_article": 147,
        "lang": "ar",
    },
    {
        "query": "What happens if unforeseen exceptional circumstances make contract "
        "performance excessively burdensome for the debtor?",
        "expected_article": 147,
        "lang": "en",
    },
    {
        "query": "إذا لم يوجد نص تشريعي ينطبق على مسألة معينة، فبماذا يحكم القاضي؟",
        "expected_article": 1,
        "lang": "ar",
    },
    {
        "query": "In the absence of an applicable legal provision, what does the judge "
        "decide according to?",
        "expected_article": 1,
        "lang": "en",
    },
]


def cosine_sim(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T


def main():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    chunk_texts, chunk_meta = [], []
    for c in chunks:
        text = c["text_ar"] or c["text_en"]
        if text.strip():
            chunk_texts.append(text)
            chunk_meta.append(c)

    print(f"[info] {len(chunk_texts)} chunks to embed\n")

    for model_name, prefixes in CANDIDATES.items():
        print("=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)

        t0 = time.time()
        model = SentenceTransformer(model_name)
        load_time = time.time() - t0

        passage_texts = [prefixes["passage_prefix"] + t for t in chunk_texts]
        t0 = time.time()
        chunk_embeddings = model.encode(
            passage_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=False
        )
        embed_time = time.time() - t0

        print(
            f"\n  load_time={load_time:.1f}s  embed_time={embed_time:.1f}s  "
            f"dim={chunk_embeddings.shape[1]}  n_chunks={len(chunk_texts)}\n"
        )

        hits_at_3 = 0
        for tq in TEST_QUERIES:
            q_text = prefixes["query_prefix"] + tq["query"]
            q_emb = model.encode([q_text], normalize_embeddings=False)
            sims = cosine_sim(q_emb, chunk_embeddings)[0]
            top_idx = np.argsort(-sims)[:3]
            top_articles = [chunk_meta[i]["article_numbers"] for i in top_idx]
            found = any(tq["expected_article"] in nums for nums in top_articles)
            hits_at_3 += found
            status = "HIT " if found else "MISS"
            print(
                f"  [{status}] ({tq['lang']}) expected article {tq['expected_article']}: "
                f"{tq['query'][:60]}..."
            )
            print(f"          top-3 article_numbers: {top_articles}")

        print(f"\n  recall@3 = {hits_at_3}/{len(TEST_QUERIES)}\n")


if __name__ == "__main__":
    main()
