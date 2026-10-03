# Query drift

Embeddings: BAAI/bge-m3. Baseline: the 54 evaluation questions.

## Off-corpus share (alerting signal)

Cut-off: best-match similarity to the indexed articles below 0.6460 (the 5th percentile of the substantive evaluation questions). Flagged when the count reaches the binomial limit (false-alarm rate 1%).

| Window | Queries | Mean best-match similarity | Below cut-off | Limit | Off-corpus? |
|---|---|---|---|---|---|
| in_domain | 16 | 0.6927 | 2 | 4 | no |
| other_jurisdiction | 16 | 0.4981 | 16 | 4 | **YES** |
| off_topic | 16 | 0.4487 | 16 | 4 | **YES** |

## Centroid drift (reported, not alerted on)

Threshold: 99th percentile of drift between random evaluation samples of the same size.

| Window | Queries | Centroid cosine | Drift | Threshold | Above threshold? | Mean nearest-baseline similarity |
|---|---|---|---|---|---|---|
| in_domain | 16 | 0.8800 | 0.1200 | 0.0898 | **YES** | 0.613 |
| other_jurisdiction | 16 | 0.7688 | 0.2312 | 0.0898 | **YES** | 0.571 |
| off_topic | 16 | 0.7017 | 0.2983 | 0.0898 | **YES** | 0.475 |
