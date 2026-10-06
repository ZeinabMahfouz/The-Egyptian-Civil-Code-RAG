| Run | Requests | Failures | Requests/s | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|---|---|
| locust_u50 | 1692 | 0 (0.0%) | 5.65 | 7.00 s | 12.00 s | 15.00 s | 18.11 s |
| locust_local_u50 | 2040 | 0 (0.0%) | 6.84 | 5.00 s | 12.00 s | 13.00 s | 20.02 s |
| locust_server_u50 | 2186 | 0 (0.0%) | 7.30 | 4.60 s | 11.00 s | 13.00 s | 14.03 s |

Time per stage (API metrics):

| Run | retrieve (mean) | generate (mean) | total (mean) |
|---|---|---|---|
| locust_u50 | 2.45 s | 4.02 s | 6.47 s |
| locust_local_u50 | 0.40 s | 5.13 s | 5.54 s |
| locust_server_u50 | 0.13 s | 5.07 s | 5.20 s |
