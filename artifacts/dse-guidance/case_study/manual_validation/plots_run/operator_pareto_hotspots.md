# Operator Pareto hotspots (all)

> How many top ops are needed to reach a MAC threshold — whether DSE should size for a few giant ops or many even ones. Structural (MAC/byte share), no performance.

| workload | n_ops | k@50%MAC | k@80%MAC | k@90%MAC | k@95%MAC | top-op MAC share |
|---|---|---|---|---|---|---|
| bitvla | 30 | 5 | 10 | 12 | 14 | 11% |
| groot_n1d7 | 116 | 27 | 56 | 77 | 87 | 2% |
| molmoact | 34 | 4 | 10 | 16 | 20 | 13% |
| openvla | 30 | 6 | 11 | 13 | 16 | 9% |
| pi05 | 777 | 34 | 66 | 193 | 341 | 2% |
| rdt | 21 | 1 | 1 | 3 | 9 | 87% |
| rdt2 | 26 | 6 | 14 | 17 | 19 | 12% |
| smolvla | 302 | 23 | 76 | 116 | 163 | 2% |
| tiny_llama | 30 | 6 | 10 | 14 | 18 | 10% |
| xr0 | 19 | 5 | 7 | 8 | 10 | 12% |
