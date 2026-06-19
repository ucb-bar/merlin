# Workload influence / magnitude stability (all)

> Per cross-workload MAC-fraction metric: macro (equal-weight) vs micro (MAC-weighted), the most influential workload (largest leave-one-out micro swing), and a flag for metrics whose **winner is stable but whose magnitude is not** — a metric can keep its ranking yet move sharply when one workload is dropped.

| metric | macro | micro | gap | most influential | max LOO micro Δ | winner-stable/magnitude-unstable |
|---|---|---|---|---|---|---|
| dense_gemm_mac_fraction | 0.140 | 0.040 | 0.100 | pi05 | 0.468 | **yes** |
| gemv_skinny_mac_fraction | 0.800 | 0.643 | 0.157 | pi05 | 0.317 | **yes** |
| true_gemv_mac_fraction | 0.051 | 0.001 | 0.050 | pi05 | 0.007 | **no** |
| skinny_gemm_mac_fraction | 0.749 | 0.642 | 0.107 | pi05 | 0.324 | **yes** |

Residency-pressure ranking (avoidable weight reload, bytes; ranking is the robust signal, absolute bytes are random-init): molmoact > pi05 > groot_n1d7 > tiny_llama > smolvla > rdt > rdt2 > xr0 > bitvla > openvla.
Per-(metric, workload) leave-one-out micro: `leave_one_out_delta_table.csv`.

