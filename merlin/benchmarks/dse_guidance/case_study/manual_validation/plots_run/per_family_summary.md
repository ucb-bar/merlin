# Per-family decomposition (all)

> Corpus split by family so averages don't hide family behavior. Linear-GEMM vs attention MAC mass and op mix per family (structural; recovered from IR).

| family | workloads | linear MACs | attention MACs | visible_linear_frac | attn ops | softmax | norm |
|---|---|---|---|---|---|---|---|
| iterative_denoise | 6 | 2318598675968 | 567247468032 | 0.803 | 387 | 504 | 1950 |
| token_decode | 4 | 9397534720 | 0 | 1.000 | 0 | 60 | 0 |
