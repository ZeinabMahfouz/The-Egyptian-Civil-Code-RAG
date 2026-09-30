import re
from dataclasses import dataclass

AR2EN = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    # EG IBAN: EG + 27 digits, optionally grouped in 4s
    "IBAN": re.compile(r"\bEG\d{2}(?:[ ]?\d){25}\b", re.IGNORECASE),
    # 13-19 digits, optional space/dash groups; Luhn-checked below
    "CREDIT_CARD": re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)"),
    # Egyptian national ID: century (2=1900s, 3=2000s) + YYMMDD + 7 digits
    "EG_NATIONAL_ID": re.compile(
        r"(?<!\d)[23]\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{7}(?!\d)"
    ),
    # Egyptian mobile: 010/011/012/015 + 8 digits, optional +20 / 0020
    "EG_PHONE": re.compile(r"(?<!\d)(?:(?:\+|00)20[ -]?)?0?1[0125](?:[ -]?\d){8}(?!\d)"),
}


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass(frozen=True)
class PIIMatch:
    entity: str
    start: int
    end: int


def detect(text: str) -> list[PIIMatch]:
    norm = text.translate(AR2EN)
    found: list[PIIMatch] = []
    taken = [False] * len(norm)
    for entity, pattern in PATTERNS.items():
        for m in pattern.finditer(norm):
            if entity == "CREDIT_CARD" and not _luhn_ok(re.sub(r"\D", "", m.group())):
                continue
            if any(taken[m.start() : m.end()]):
                continue
            found.append(PIIMatch(entity, m.start(), m.end()))
            for i in range(m.start(), m.end()):
                taken[i] = True
    return sorted(found, key=lambda x: x.start)


def redact(text: str) -> tuple[str, list[str]]:
    """Returns (redacted text, sorted entity types found)."""
    matches = detect(text)
    out, last = [], 0
    for m in matches:
        out.append(text[last : m.start])
        out.append(f"[{m.entity}]")
        last = m.end
    out.append(text[last:])
    return "".join(out), sorted({m.entity for m in matches})


class StreamingRedactor:
    """Redacts PII in text that arrives in pieces (streamed LLM tokens).

    A phone number can arrive as "010" + "1234" + "5678": redacting each
    piece on its own would miss it, and emitting "010" before the rest
    arrives would leak it. So the last HOLD_BACK characters are always held
    back, and a PII match is never cut in half -- if one straddles the
    boundary, the boundary moves to before it. Everything longer than any
    realistic PII value (the longest, a spaced EG IBAN, is ~35 chars) is
    therefore seen whole before it's released.

    Cost: the client sees text ~HOLD_BACK characters behind the model.
    """

    HOLD_BACK = 64

    def __init__(self):
        self._buffer = ""
        self.entities: set[str] = set()

    def _release(self, upto: int) -> str:
        head, self._buffer = self._buffer[:upto], self._buffer[upto:]
        redacted, found = redact(head)
        self.entities.update(found)
        return redacted

    def feed(self, piece: str) -> str:
        """Add a piece; return whatever is now safe to show (may be "")."""
        self._buffer += piece
        cut = len(self._buffer) - self.HOLD_BACK
        if cut <= 0:
            return ""
        for m in detect(self._buffer):
            if m.start < cut < m.end:
                cut = m.start
                break
        return self._release(cut) if cut > 0 else ""

    def flush(self) -> str:
        """End of stream: release (and redact) everything left."""
        return self._release(len(self._buffer))


# --- Guardrails enforcement layer ----------------------------------------

try:
    from guardrails import Guard, OnFailAction
    from guardrails.settings import settings
    from guardrails.validator_base import (
        ErrorSpan,
        FailResult,
        PassResult,
        Validator,
        register_validator,
    )

    @register_validator(name="egyptian-civil-code-rag/egyptian_pii", data_type="string")
    class EgyptianPII(Validator):
        def _validate(self, value: str, metadata: dict):
            matches = detect(value)
            if not matches:
                return PassResult()
            redacted, entities = redact(value)
            return FailResult(
                error_message=f"PII detected: {', '.join(entities)}",
                fix_value=redacted,
                error_spans=[ErrorSpan(start=m.start, end=m.end, reason=m.entity) for m in matches],
            )

    GUARDRAILS_AVAILABLE = True
except ImportError:  # pragma: no cover -- keeps the module usable without guardrails-ai
    GUARDRAILS_AVAILABLE = False


class PIIGuard:
    def __init__(self, use_guardrails: bool | None = None):
        self.uses_guardrails = GUARDRAILS_AVAILABLE if use_guardrails is None else use_guardrails
        if self.uses_guardrails and not GUARDRAILS_AVAILABLE:
            raise RuntimeError("use_guardrails=True but guardrails-ai is not installed")
        self._guard = None
        if self.uses_guardrails:
            guard = Guard()
            guard.configure(allow_metrics_collection=False)
            settings.rc.enable_metrics = False
            self._guard = guard.use(EgyptianPII(on_fail=OnFailAction.FIX))

    def __call__(self, text: str) -> tuple[str, list[str]]:
        redacted, entities = redact(text)
        if self._guard is not None:
            redacted = self._guard.validate(text).validated_output
        return redacted, entities
