import json
import re
from pathlib import Path
from typing import Callable

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny
from sentence_transformers import SentenceTransformer

DEFAULT_TOP_K_RAW = 8  # raw Qdrant hits fetched before dedup
DEFAULT_TOP_K_DISTINCT = 3  # distinct articles kept as context after dedup

AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EN_DIGITS = "0123456789"
AR2EN = str.maketrans(AR_DIGITS, EN_DIGITS)
RE_ARTICLE_REF = re.compile(r"(?:Article|article|مادة|المادة)\s*[:#]?\s*([0-9٠-٩]+)")


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def extract_referenced_article_numbers(question: str) -> list[int]:
    numbers = []
    for m in RE_ARTICLE_REF.finditer(question):
        digits = m.group(1).translate(AR2EN)
        if digits.isdigit():
            numbers.append(int(digits))
    return numbers


class RAGQueryEngine:
    def __init__(
        self, params_path: Path = Path("params.yaml"), generate_fn: Callable[[str], str] = None
    ):
        with open(params_path, encoding="utf-8") as f:
            params = yaml.safe_load(f)
        self.embed_model = SentenceTransformer(params["embedding"]["model_name"])
        self.client = QdrantClient(path=params["qdrant"]["storage_path"])
        self.collection = params["qdrant"]["collection_name"]
        self.generate_fn = generate_fn

    def retrieve(
        self,
        question: str,
        top_k_raw: int = DEFAULT_TOP_K_RAW,
        top_k_distinct: int = DEFAULT_TOP_K_DISTINCT,
    ):
        query_vec = self.embed_model.encode([question], normalize_embeddings=False)[0].tolist()
        seen_groups = set()
        distinct = []

        referenced = extract_referenced_article_numbers(question)
        if referenced:
            exact_hits = self.client.query_points(
                collection_name=self.collection,
                query=query_vec,
                query_filter=Filter(
                    must=[FieldCondition(key="article_numbers", match=MatchAny(any=referenced))]
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
        context_blocks = []
        for hit in context_hits:
            p = hit.payload
            status = " (REPEALED -- no longer in force)" if p["is_repealed"] else ""
            context_blocks.append(f"[{p['citation']}]{status}\n{p['text']}")
        context = "\n\n".join(context_blocks)

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

    engine.close()
