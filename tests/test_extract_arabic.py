"""Unit test for the RTL word fix in scripts/extract_corpus.py -- runs without
the PDF (the corpus-level regression gate is in test_corpus_validation.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from extract_corpus import fix_arabic_word  # noqa: E402


def word(*glyphs):
    """A pdfplumber word whose glyphs are given in *visual* (left-to-right)
    order, the way the PDF lays out RTL text."""
    return {"text": "".join(glyphs), "chars": [{"text": g} for g in glyphs]}


def test_ligature_glyph_keeps_its_internal_order():
    # "إلا" on the page, left to right: [لا-ligature][إ]. The ligature glyph's
    # text is already logical ("لا"); only the glyph *order* must be reversed.
    assert fix_arabic_word(word("لا", "إ")) == "إلا"
    # The old approach -- reversing the text string -- produced the broken form:
    assert word("لا", "إ")["text"][::-1] == "إال"


def test_genuine_alef_lam_is_untouched():
    # "المال": alef and lam are separate glyphs here (no ligature) and must
    # stay alef-then-lam.
    assert fix_arabic_word(word("ل", "ا", "م", "ل", "ا")) == "المال"


def test_ligature_mid_word():
    # "خلال" = خ + لا(ligature) + ل ; visual order: ل, لا, خ
    assert fix_arabic_word(word("ل", "لا", "خ")) == "خلال"


def test_digit_tokens_unchanged():
    assert fix_arabic_word(word("١", "٤", "٧")) == "١٤٧"


def test_falls_back_to_string_reversal_without_glyphs():
    assert fix_arabic_word({"text": "مدا"}) == "ادم"


def test_presentation_form_words_are_treated_as_arabic():
    # Some PDFs emit presentation forms (U+FEFB = lam-alef ligature) instead of
    # base letters; such a word must still be reversed.
    assert fix_arabic_word(word("ﻻ", "إ")) == "إﻻ"
