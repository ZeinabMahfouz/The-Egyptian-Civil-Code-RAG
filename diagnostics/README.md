# Extraction debugging trail

One-off scripts used to diagnose specific failure modes during
development of `scripts/extract_corpus.py`. Not part of the
reproducible pipeline (no dvc.yaml stage depends on these) --
kept as documentation of the debugging process.

- diagnose_columns.py   -- found the AR/EN column-split boundary via pdfplumber word positions
- diagnose_gaps.py      -- page-1 duplicate articles, Article 54's headerless repeal block, Article 1022 location
- diagnose_missing_en.py -- fused header/body rows (714, 746, 898, 908, 1090)
- diagnose_277_452.py   -- column-split false alarm (277) vs. dropped-character glyph bug (452)
- diagnose_1022.py      -- confirmed Article 1022's Arabic cell is genuinely empty in the source PDF
