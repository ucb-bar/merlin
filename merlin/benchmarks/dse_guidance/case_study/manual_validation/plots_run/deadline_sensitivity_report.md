# Deadline-sensitivity of the requirement envelope (all)

> Required **compute rate** (= configured-K replan MACs / deadline) as the replan deadline tightens. A REQUIREMENT, not a hardware prediction — a future accelerator must *exceed* this to meet the deadline. K is configured/reference; deadlines are a sweep.

| workload | K | 50ms | 100ms | 200ms | 500ms |
|---|---|---|---|---|---|
| bitvla | 7 | 0.20 GMAC/s | 0.10 GMAC/s | 0.05 GMAC/s | 0.02 GMAC/s |
| groot_n1d7 | 4 | 1631.47 GMAC/s | 815.73 GMAC/s | 407.87 GMAC/s | 163.15 GMAC/s |
| molmoact | 8 | 151.50 GMAC/s | 75.75 GMAC/s | 37.87 GMAC/s | 15.15 GMAC/s |
| openvla | 7 | 0.11 GMAC/s | 0.06 GMAC/s | 0.03 GMAC/s | 0.01 GMAC/s |
| pi05 | 10 | 3138.62 GMAC/s | 1569.31 GMAC/s | 784.66 GMAC/s | 313.86 GMAC/s |
| rdt | 5 | 3946.60 GMAC/s | 1973.30 GMAC/s | 986.65 GMAC/s | 394.66 GMAC/s |
| rdt2 | 5 | 99.19 GMAC/s | 49.59 GMAC/s | 24.80 GMAC/s | 9.92 GMAC/s |
| smolvla | 10 | 1018.88 GMAC/s | 509.44 GMAC/s | 254.72 GMAC/s | 101.89 GMAC/s |
| tiny_llama | 7 | 21.51 GMAC/s | 10.75 GMAC/s | 5.38 GMAC/s | 2.15 GMAC/s |
| xr0 | 5 | 111.48 GMAC/s | 55.74 GMAC/s | 27.87 GMAC/s | 11.15 GMAC/s |

Full K×deadline grid (compute / weight-bandwidth resident-vs-nonresident / command rate): `timing_requirement_envelope.csv`.

