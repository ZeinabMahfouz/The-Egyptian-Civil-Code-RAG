import pytest

from egyptian_civil_code_rag.pii import GUARDRAILS_AVAILABLE, PIIGuard, detect, redact

needs_guardrails = pytest.mark.skipif(
    not GUARDRAILS_AVAILABLE, reason="guardrails-ai not installed (optional)"
)


@pytest.mark.parametrize(
    "text, entity",
    [
        ("الرقم القومي 29801011234567", "EG_NATIONAL_ID"),
        ("الرقم القومي ٢٩٨٠١٠١١٢٣٤٥٦٧", "EG_NATIONAL_ID"),  # Arabic-Indic digits
        ("ID 30512311234567", "EG_NATIONAL_ID"),  # born 2005
        ("موبايل 01012345678", "EG_PHONE"),
        ("موبايل ٠١١٢٣٤٥٦٧٨٩", "EG_PHONE"),
        ("call +20 100 123 4567", "EG_PHONE"),
        ("call 0020-1512345678", "EG_PHONE"),
        ("mail zeinab.test@example.com", "EMAIL"),
        ("card 4111 1111 1111 1111", "CREDIT_CARD"),
        ("iban EG380019000500000000263180002", "IBAN"),
    ],
)
def test_detects(text, entity):
    assert [m.entity for m in detect(text)] == [entity]


@pytest.mark.parametrize(
    "text",
    [
        "What does Article 147 say?",
        "المادة ١٤٧ من القانون رقم ١٨١ لسنة ٢٠١٨",
        "Articles 54-80 were repealed",
        "Articles 389-417, page 34, 1149 articles in total",
        "Law No. 131 of 1948, published 16 July 1948",
        "مبلغ 1500000 جنيه",  # a large amount is not a phone number
        "card-like but invalid Luhn 4111 1111 1111 1112",
    ],
)
def test_no_false_positives_on_legal_text(text):
    assert detect(text) == []


def test_redact_replaces_every_entity_and_keeps_the_rest():
    text = "أنا 29801011234567 وتليفوني 01012345678 وإيميلي a@b.co، ما حكم المادة 147؟"
    out, entities = redact(text)
    assert out == "أنا [EG_NATIONAL_ID] وتليفوني [EG_PHONE] وإيميلي [EMAIL]، ما حكم المادة 147؟"
    assert entities == ["EG_NATIONAL_ID", "EG_PHONE", "EMAIL"]


def test_guard_without_guardrails_library():
    guard = PIIGuard(use_guardrails=False)
    assert guard("tel 01012345678") == ("tel [EG_PHONE]", ["EG_PHONE"])
    assert guard("Article 147") == ("Article 147", [])


@needs_guardrails
def test_guardrails_guard_redacts_via_fix_action():
    guard = PIIGuard(use_guardrails=True)
    out, entities = guard("tel 01012345678")
    assert out == "tel [EG_PHONE]"
    assert entities == ["EG_PHONE"]
    assert guard("Article 147")[0] == "Article 147"


def test_national_id_requires_a_valid_birth_date():
    assert "EG_NATIONAL_ID" not in {m.entity for m in detect("29813011234567")}


@needs_guardrails
def test_guard_sends_no_telemetry(monkeypatch):
    import socket
    import time

    lookups = []
    real = socket.getaddrinfo
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda host, *a, **k: lookups.append(host) or real(host, *a, **k)
    )
    guard = PIIGuard(use_guardrails=True)
    guard("tel 01012345678")
    time.sleep(6)  # span exporter batches in the background
    assert lookups == []
