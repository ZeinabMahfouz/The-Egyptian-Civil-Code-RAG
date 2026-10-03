import json
import re
from pathlib import Path
from typing import Callable

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
from sentence_transformers import SentenceTransformer

DEFAULT_TOP_K_RAW = 8  # raw Qdrant hits fetched before dedup
DEFAULT_TOP_K_DISTINCT = 3  # distinct articles kept as context after dedup

# Every indexed point carries a doc_id payload. The Civil Code's is fixed;
# additional laws added via scripts/reindex_batch.py get their own.
CIVIL_CODE_DOC_ID = "egyptian_civil_code"

AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EN_DIGITS = "0123456789"
AR2EN = str.maketrans(AR_DIGITS, EN_DIGITS)
RE_ARTICLE_REF = re.compile(r"(?:Article|article|مادة|المادة)\s*[:#]?\s*([0-9٠-٩]+)")


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


RE_ARABIC = re.compile(r"[\u0600-\u06FF]")


def question_language(question: str) -> str:
    """'ar' if the question contains Arabic letters, else 'en'."""
    return "ar" if RE_ARABIC.search(question) else "en"


def format_context(payload: dict, question: str = "") -> str:
    """One retrieved article exactly as the generator sees it: citation,
    repeal status, text. The RAGAS judge gets the same string -- if it only
    saw the bare text, every correct "according to Article 44" would count
    as an unsupported claim (the label lives outside the text).

    Repealed ranges are indexed as one chunk ("Articles 389-417"). Asked
    "Is Article 400 still in force?", the small CPU model was shown that
    chunk and still answered "no information" -- it didn't work out that
    400 lies between 389 and 417. So when the question names an article
    inside a repealed range, the context states it outright."""
    status = " (REPEALED -- no longer in force)" if payload.get("is_repealed") else ""
    note = ""
    numbers = payload.get("article_numbers") or []
    if payload.get("is_repealed") and len(numbers) > 1 and question:
        inside = sorted(set(extract_referenced_article_numbers(question)) & set(numbers))
        if inside:
            listed = ", ".join(str(n) for n in inside)
            note = (
                f"\nNote: Article {listed} is within this range, so Article {listed} is REPEALED."
            )
    return f"[{payload['citation']}]{status}\n{payload['text']}{note}"


def extract_referenced_article_numbers(question: str) -> list[int]:
    numbers = []
    for m in RE_ARTICLE_REF.finditer(question):
        digits = m.group(1).translate(AR2EN)
        if digits.isdigit():
            numbers.append(int(digits))
    return numbers


class RAGQueryEngine:
    def __init__(
        self,
        params_path: Path = Path("params.yaml"),
        generate_fn: Callable[[str], str] = None,
        embed_model=None,
        client=None,
        collection: str | None = None,
    ):
        """embed_model / client / collection override what params.yaml says --
        used by the GPU evaluation sweep, which builds several indexes and
        reuses one loaded embedding model across them."""
        with open(params_path, encoding="utf-8") as f:
            params = yaml.safe_load(f)
        self.embed_model = embed_model or SentenceTransformer(params["embedding"]["model_name"])
        self.client = client or QdrantClient(path=params["qdrant"]["storage_path"])
        self.collection = collection or params["qdrant"]["collection_name"]
        self.generate_fn = generate_fn
        self._check_index_has_doc_ids()

    def _check_index_has_doc_ids(self):
        """Exact article lookup filters on doc_id == Civil Code. An index
        built before doc_id existed would make that filter match nothing,
        silently regressing "What does Article 147 say?" back to pure
        semantic search -- so refuse to start instead."""
        civil_code_points = self.client.count(
            collection_name=self.collection,
            count_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=CIVIL_CODE_DOC_ID))]
            ),
        ).count
        if civil_code_points == 0:
            raise RuntimeError(
                f"No points with doc_id={CIVIL_CODE_DOC_ID!r} in collection "
                f"'{self.collection}'. The index predates the doc_id payload field -- "
                "rebuild it with `dvc repro` (and `dvc push` so CI/Docker get it too)."
            )

    def retrieve(
        self,
        question: str,
        top_k_raw: int = DEFAULT_TOP_K_RAW,
        top_k_distinct: int = DEFAULT_TOP_K_DISTINCT,
    ):
        hits = self._retrieve_distinct(question, top_k_raw, top_k_distinct)
        return self._prefer_language(hits, question_language(question))

    def _prefer_language(self, hits, lang: str):
        """Every chunk is indexed twice (Arabic and English text). Dedup keeps
        whichever copy ranked first, so an English question could be answered
        from Arabic text (and vice versa) -- the model then translates on the
        fly, and the judge has to verify an English claim against Arabic.
        Swap each hit for its same-language twin when one exists; rank and
        score stay those of the original hit."""
        out = []
        for hit in hits:
            p = hit.payload
            if p.get("lang") == lang or "chunk_id" not in p or "doc_id" not in p:
                out.append(hit)
                continue
            twins, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="chunk_id", match=MatchValue(value=p["chunk_id"])),
                        FieldCondition(key="doc_id", match=MatchValue(value=p["doc_id"])),
                        FieldCondition(key="lang", match=MatchValue(value=lang)),
                    ]
                ),
                limit=1,
                with_payload=True,
            )
            if twins:
                hit.payload = twins[0].payload
            out.append(hit)
        return out

    def _retrieve_distinct(self, question: str, top_k_raw: int, top_k_distinct: int):
        query_vec = self.embed_model.encode([question], normalize_embeddings=False)[0].tolist()
        seen_groups = set()
        distinct = []

        referenced = extract_referenced_article_numbers(question)
        if referenced:
            exact_hits = self.client.query_points(
                collection_name=self.collection,
                query=query_vec,
                # Scoped to the Civil Code: "Article 1" must not also match
                # Article 1 of every other indexed law.
                query_filter=Filter(
                    must=[
                        FieldCondition(key="article_numbers", match=MatchAny(any=referenced)),
                        FieldCondition(key="doc_id", match=MatchValue(value=CIVIL_CODE_DOC_ID)),
                    ]
                ),
                limit=top_k_raw,
            ).points
            for hit in exact_hits:
                group_key = tuple(hit.payload["article_numbers"])
                if group_key in seen_groups:
                    continue
                seen_groups.add(group_key)
                distinct.append(hit)
                if len(distinct) >= top_k_distinct:
                    return distinct

        hits = self.client.query_points(
            collection_name=self.collection,
            query=query_vec,
            limit=top_k_raw,
        ).points
        for hit in hits:
            group_key = tuple(hit.payload["article_numbers"])
            if group_key in seen_groups:
                continue
            seen_groups.add(group_key)
            distinct.append(hit)
            if len(distinct) >= top_k_distinct:
                break
        return distinct

    def build_prompt(self, question: str, context_hits):
        context = "\n\n".join(format_context(hit.payload, question) for hit in context_hits)

        return f"""You are a legal assistant answering questions about the Egyptian Civil Code.
Answer ONLY using the articles provided below. Every claim in your answer must be
attributable to one of these articles. Cite the article number for every claim.
If an article is marked REPEALED, say so explicitly rather than treating it as current law.
If the provided articles don't contain enough information to answer, say so -- do not guess.

Articles:
{context}

Question: {question}

Answer (in the same language as the question, citing article numbers):"""

    def ask(self, question: str) -> dict:
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        context_hits = self.retrieve(question)
        if not context_hits:
            return {"answer": "No relevant articles found.", "sources": []}

        prompt = self.build_prompt(question, context_hits)
        raw_answer = self.generate_fn(prompt)

        sources = [hit.payload["citation"] for hit in context_hits]
        return {"answer": raw_answer.strip(), "sources": sources}

    def close(self):
        """Explicit cleanup rather than relying on __del__ -- avoids the
        'Python is likely shutting down' warning from Qdrant's local
        client destructor racing interpreter teardown, and this is the
        method a FastAPI shutdown handler will call later."""
        self.client.close()


if __name__ == "__main__":
    from egyptian_civil_code_rag.backends import transformers_backend

    MODEL_NAME = "Qwen/Qwen3-1.7B"
    print(f"[info] loading {MODEL_NAME} (CPU dev backend, not production)")
    engine = RAGQueryEngine(generate_fn=transformers_backend(MODEL_NAME))

    test_questions = [
        "ماذا يحدث إذا جعلت ظروف استثنائية غير متوقعة تنفيذ العقد مرهقاً للمدين؟",
        "Is the law about associations (Articles 54-80) still in force?",
    ]
    for q in test_questions:
        print(f"\n{'=' * 70}\nQ: {q}\n{'=' * 70}")
        result = engine.ask(q)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    engine.close()
