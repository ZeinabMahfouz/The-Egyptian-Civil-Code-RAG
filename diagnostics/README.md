# Debugging and analysis trail

One-off scripts used during development of the extraction and chunking
pipeline (`scripts/extract_corpus.py`, `scripts/chunk_corpus.py`). Not
part of the reproducible pipeline (no dvc.yaml stage depends on these)
-- kept as documentation of the debugging and design process.

## Extraction debugging (data/interim/civil_code.json)
- diagnose_columns.py    -- found the AR/EN column-split boundary via pdfplumber word positions
- diagnose_gaps.py       -- page-1 duplicate articles, Article 54's headerless repeal block, Article 1022 location
- diagnose_missing_en.py -- fused header/body rows (714, 746, 898, 908, 1090)
- diagnose_277_452.py    -- column-split false alarm (277) vs. dropped-character glyph bug (452)
- diagnose_1022.py       -- confirmed Article 1022's Arabic cell is genuinely empty in the source PDF

## Chunking analysis (data/interim/chunks.json)
- corpus_length_stats.py -- article length distribution (min/max/percentiles) that informed the
                             paragraph_split_threshold_chars default in params.yaml, and flagged
                             the 29 near-duplicate repealed-placeholder articles that led to the
                             repealed-range dedup design
- spot_check_chunks.py   -- manual verification of paragraph-split chunks (238, 658), the two
                             deduped repealed-range chunks, and a normal whole-article chunk,
                             confirming chunk_corpus.py's output matches its design on real data