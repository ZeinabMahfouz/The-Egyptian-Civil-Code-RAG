"""Refusal gate: decline before generation when nothing indexed is close
to the question (RAGQueryEngine.should_refuse), wired through the shared
pipeline, and its calibration script -- with fake hits, no model or index."""

import sys
from pathlib import Path

import pytest
from prometheus_client import REGISTRY

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import refusal_calibration as cal  # noqa: E402
from test_api import FakeEngine  # noqa: E402

from egyptian_civil_code_rag.pii import PIIGuard  # noqa: E402
from egyptian_civil_code_rag.pipeline import TracedPipeline, make_langfuse  # noqa: E402
from egyptian_civil_code_rag.query import (  # noqa: E402
    RAGQueryEngine,
    extract_referenced_article_numbers,
)
from egyptian_civil_code_rag.refusal import GATE_REFUSAL, is_refusal  # noqa: E402


class Hit:
    def __init__(self, score, articles=(147,), doc_id="egyptian_civil_code"):
        self.score = score
        self.payload = {
            "citation": f"Egyptian Civil Code, Article {articles[0]}",
            "article_numbers": list(articles),
            "doc_id": doc_id,
            "lang": "en",
        }


def gate(min_score):
    e = RAGQueryEngine.__new__(RAGQueryEngine)  # no model or index
    e.refusal_min_score = min_score
    return e


def test_off_by_default_never_refuses():
    assert not gate(None).should_refuse("weather tomorrow?", [Hit(0.1)])
    assert not RAGQueryEngine.__new__(RAGQueryEngine).should_refuse("q", [Hit(0.1)])


def test_refuses_only_below_the_threshold():
    e = gate(0.6)
    assert e.should_refuse("What is the capital of France?", [Hit(0.45), Hit(0.40)])
    assert not e.should_refuse("Can a contract be rescinded?", [Hit(0.62), Hit(0.3)])
    assert not e.should_refuse("anything", [])  # no hits: the no-context path handles it


def test_named_article_in_the_index_is_never_refused():
    e = gate(0.6)
    assert not e.should_refuse("What does Article 147 say?", [Hit(0.41, (147,))])
    assert not e.should_refuse("ما نص المادة ١٤٧؟", [Hit(0.41, (147,))])  # Arabic digits
    # names an article the hits don't contain -> judged on similarity
    assert e.should_refuse("What does Article 5000 say?", [Hit(0.41, (147,))])
    # Article 1 of another law doesn't exempt a Civil Code question
    assert e.should_refuse("What does Article 1 say?", [Hit(0.41, (1,), doc_id="other_law")])


def test_gate_message_is_recognised_as_a_refusal_in_both_languages():
    assert all(is_refusal(m) for m in GATE_REFUSAL.values())


class GatedEngine(FakeEngine):
    def __init__(self, score):
        self.score = score
        self.prompts = []
        self.refusal_min_score = 0.6

    def retrieve(self, question):
        return [Hit(self.score)]

    should_refuse = RAGQueryEngine.should_refuse

    def generate_fn(self, prompt):
        raise AssertionError("a refused question must not reach the LLM")


def run(engine, question, svc):
    p = TracedPipeline(engine, PIIGuard(use_guardrails=False), make_langfuse(), svc)
    chunks = []
    return p.ask(question, on_chunk=chunks.append), chunks


@pytest.mark.parametrize(
    "question,lang", [("What is the capital of France?", "en"), ("ما عاصمة فرنسا؟", "ar")]
)
def test_pipeline_refuses_without_calling_the_llm(question, lang):
    svc = f"t-refuse-{lang}"
    result, chunks = run(GatedEngine(0.3), question, svc)
    assert result.refused and result.sources == []
    assert result.answer == GATE_REFUSAL[lang] and chunks == [result.answer]
    assert (
        REGISTRY.get_sample_value("rag_requests_total", {"service": svc, "status": "refused"}) == 1
    )
    assert (
        REGISTRY.get_sample_value(
            "rag_request_latency_seconds_count", {"service": svc, "stage": "generate"}
        )
        is None
    )


def test_pipeline_answers_above_the_threshold():
    class Answering(GatedEngine):
        def generate_fn(self, prompt):
            return "Article 147 says ..."

    result, _ = run(Answering(0.8), "Can a contract be rescinded?", "t-pass")
    assert not result.refused and result.sources == ["Egyptian Civil Code, Article 147"]


class ScoredEngine:
    """retrieve() returns one hit whose score is looked up by question."""

    def __init__(self, scores):
        self.scores = scores
        self.refusal_min_score = None

    should_refuse = RAGQueryEngine.should_refuse

    def retrieve(self, question):
        # in-corpus lookups find the article they name, as the real index does
        named = extract_referenced_article_numbers(question)
        return [Hit(self.scores[question], tuple(named) or (147,))]


def test_calibration_recommends_below_the_lowest_in_corpus_question(tmp_path):
    rows = cal.load_questions(
        REPO / "tests/eval/eval_questions.json", REPO / "tests/eval/drift_windows.json"
    )
    # in-corpus questions score 0.62-0.9, out-of-corpus 0.35-0.66
    scores = {}
    for i, r in enumerate(rows):
        scores[r["question"]] = (
            (0.62 + (i % 10) * 0.03) if r["in_corpus"] else 0.35 + (i % 10) * 0.035
        )
    engine = ScoredEngine(scores)
    result = cal.main(["--reports-dir", str(tmp_path), "--drift-cutoff", "0.646"], engine=engine)

    in_tune = [
        r["top_score"]
        for r in result["questions"]
        if r["set"] == "tune" and r["in_corpus"] and not r["exempt"]
    ]
    assert result["recommended"] == round(min(in_tune) - cal.MARGIN, 3)
    rec = result["thresholds"]["recommended"]
    assert rec["tune"]["in_corpus_refused"] == 0
    assert rec["tune"]["out_of_corpus"] == 6 and rec["check"]["out_of_corpus"] == 32
    assert rec["check"]["in_corpus"] == 16
    # article-lookup questions are exempt however low they score
    assert any(r["exempt"] for r in result["questions"] if r["group"] == "article_lookup")
    assert engine.refusal_min_score is None  # calibration leaves the engine as it found it
    md = (tmp_path / "refusal_calibration.md").read_text(encoding="utf-8")
    assert "| recommended |" in md and "drift cut-off" in md
    assert (tmp_path / "refusal_calibration.json").exists()


def test_params_ship_with_the_gate_set_or_explicitly_off():
    import yaml

    params = yaml.safe_load((REPO / "params.yaml").read_text(encoding="utf-8"))
    min_score = params["refusal"]["min_score"]
    assert min_score is None or 0 < min_score < 1


def test_article_5000_refusals_are_recognised():
    # The two answers that were miscounted as "answered" in the GPU evaluation
    assert is_refusal(
        "The provided articles do not include Article 5000. Therefore, it is not "
        "possible to answer what Article 5000 says."
    )
    assert is_refusal("المادة 5000 ليست موجودة في القائمة المقدمة من مواد القانون المدني المصري.")
    # an ordinary answer still isn't one
    assert not is_refusal("Article 147: the contract makes the law of the parties.")
