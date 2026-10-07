"""Out-of-corpus handling for evaluation: which questions the system should
decline, and how a declined answer looks.

Shared by scripts/gpu_eval.py (scores), scripts/ragas_gate.py (CI gate) and
metrics.py (the faithfulness gauge). Standard library only, so the CI gate
can import it without the ML stack.
"""

import re

OUT_OF_CORPUS = "out_of_corpus"

# How the generator declines ("the provided articles do not contain ...").
RE_REFUSAL = re.compile(
    r"no relevant articles|do(?:es)? not contain|not covered|cannot answer|"
    r"not (?:enough|sufficient) information|insufficient information|"
    r"لا تحتوي|لا تتضمن|لا توجد معلومات|لا يتناول|لا تتناول|لا يعالج|لا يُعالج|غير كافية",
    re.I,
)


# What the API answers when the refusal gate declines a question before the
# LLM sees it (query.RAGQueryEngine.should_refuse). Worded so is_refusal()
# recognises it, in the question's language.
GATE_REFUSAL = {
    "en": (
        "The Egyptian Civil Code articles available to me do not cover this question, "
        "so I cannot answer it."
    ),
    "ar": "لا تتناول مواد القانون المدني المصري المتاحة لي هذا السؤال، لذلك لا يمكنني الإجابة عنه.",
}


def gate_refusal(lang: str) -> str:
    return GATE_REFUSAL.get(lang, GATE_REFUSAL["en"])


def is_refusal(answer: str) -> bool:
    return bool(RE_REFUSAL.search(answer or ""))


def is_in_corpus(row: dict) -> bool:
    """Evaluation rows carry the question's category as _category."""
    return row.get("_category") != OUT_OF_CORPUS
