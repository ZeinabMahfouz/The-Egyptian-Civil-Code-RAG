# Quantization: Qwen3-8B fp16 vs AWQ 4-bit

Judge for both: Qwen/Qwen3-8B. Acceptance: faithfulness drop < 0.03. Result: **PASS** (drop -0.002).

## Quality (RAGAS, in-corpus questions)

| Metric | fp16 | AWQ | Change |
|---|---|---|---|
| faithfulness | 0.886 | 0.888 | 0.002 |
| context_precision | 0.908 | 0.917 | 0.009 |
| context_recall | 0.854 | 0.854 | 0.000 |
| answer_relevancy | 0.701 | 0.706 | 0.005 |
| refusal_rate | 0.667 | 0.667 | 0.000 |
| false_refusal_rate | 0.021 | 0.021 | 0.000 |

## Latency and memory

| Measure | fp16 | AWQ |
|---|---|---|
| latency_p50_s | 2.81 | 1.06 |
| latency_p95_s | 9.42 | 2.51 |
| ttft_p50_s | 0.24 | 0.17 |
| ttft_p95_s | 0.35 | 0.26 |
| tokens_per_s_p50 | 18.28 | 55.02 |
| concurrent_requests_per_s | 2.24 | 4.21 |
| concurrent_tokens_per_s | 149.21 | 285.02 |
| concurrent_latency_p95_s | 8.40 | 3.92 |
| weights per GPU (GiB) | 7.64 | 2.85 |
