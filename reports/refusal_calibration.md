# Refusal gate calibration

Retrieval only (`scripts/refusal_calibration.py`). A question is refused when its
best match among the indexed articles scores below the threshold, unless it names
an article found in the index.

| Threshold | | Tune: out-of-corpus refused | Tune: in-corpus refused | Check: out-of-corpus refused | Check: in-corpus refused |
|---|---|---|---|---|---|
| 0.472 | recommended | 2 of 6 | 0 of 48 | 15 of 32 | 0 of 16 |
| 0.550 | 0.55 | 3 of 6 | 4 of 48 | 28 of 32 | 0 of 16 |
| 0.600 | 0.60 | 5 of 6 | 6 of 48 | 30 of 32 | 1 of 16 |
| 0.646 | drift cut-off | 6 of 6 | 7 of 48 | 32 of 32 | 2 of 16 |
| 0.650 | 0.65 | 6 of 6 | 7 of 48 | 32 of 32 | 2 of 16 |
| 0.700 | 0.70 | 6 of 6 | 17 of 48 | 32 of 32 | 11 of 16 |

## Closest calls

Lowest-scoring in-corpus questions and highest-scoring out-of-corpus ones
(exempt questions excluded):

| Set | Group | Id | Best match | Question |
|---|---|---|---|---|
| tune | repealed_status | r01_en | 0.477 | Is the law about associations still in force? |
| tune | repealed_status | r05_en | 0.496 | What happened to the articles on proof of obligations (389-417)? |
| tune | repealed_status | r10_ar | 0.498 | هل يمكن الاستناد إلى نصوص المواد من 54 إلى 80 اليوم؟ |
| tune | repealed_status | r02_ar | 0.505 | هل ما زالت أحكام الجمعيات سارية؟ |
| tune | substantive | s11_ar | 0.550 | ما هو تعريف الالتزام التخييري؟ |
| tune | repealed_status | r06_ar | 0.564 | هل تم إلغاء نصوص إثبات الالتزام؟ |
| tune | out_of_corpus | o01_en | 0.621 | What does Egyptian criminal law say about theft? |
| check | other_jurisdiction | other_jurisdiction_05 | 0.619 | What are the income tax brackets for individuals in Egypt? |
| check | other_jurisdiction | other_jurisdiction_02 | 0.600 | What is the penalty for drug possession under the Egyptian Penal Code? |
| tune | out_of_corpus | o03_en | 0.582 | What is the minimum wage in Egypt according to this code? |
| check | other_jurisdiction | other_jurisdiction_03 | 0.575 | How long can the public prosecution detain a suspect before trial in Egypt? |
| tune | out_of_corpus | o06_en | 0.574 | What is the capital of Egypt? |
