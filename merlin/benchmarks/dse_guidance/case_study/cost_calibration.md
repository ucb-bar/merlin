# Cost-model calibration — predicted vs measured cycles

- substrate: **firesim_fased_scalar**  ·  source: docs/results.md (FireSim FASED scalar sweep)
- predictor: `cycles ~ (cycles/MAC) * total_matmul_MACs` (MACs from real capture IR)
- fitted: **99.3 cycles/MAC** (median over 4 consistent models)
- MAPE (consistent set): **32%**

| model | dtype | MACs | measured cycles | cycles/MAC | predicted | rel err | outlier |
|-------|-------|------|-----------------|------------|-----------|---------|---------|
| small_llama | int8 | 3.42e+06 | 1.80e+08 | 52.7 | 3.40e+08 | 88% |  |
| tiny_llama | int8 | 1.23e+09 | 1.33e+11 | 108.3 | 1.22e+11 | 8% |  |
| openvla | int8 | 7.95e+07 | 9.83e+09 | 123.6 | 7.90e+09 | 20% |  |
| bitvla | int8 | n/a | 8.17e+09 | n/a | n/a | n/a |  |
| rdt2 | int8 | 9.41e+08 | 8.50e+10 | 90.3 | 9.35e+10 | 10% |  |
| xr0 | fp32 | 1.31e+06 | 1.46e+11 | 111541.7 | 1.30e+08 | 100% | YES |

**Verdict:** Fitted 99.3 cycles/MAC (median over 4 consistent models). MAPE on the consistent set = 32%. 1 outlier(s) the matmul-only predictor CANNOT explain: xr0 (1123x median) — its capture MAC count is inconsistent with its measured cycles (a partial capture, or a run dominated by non-matmul / repeated-body work). 1 model(s) excluded (capture did not parse). Conclusion: a single cycles/MAC constant is crude but usable as analytical ordering; matmul MACs alone do not capture whole-model scalar cycles. Quantitative gap_closure stays gated on per-op-family calibration or direct measurement.

## Per-component calibration attempt (multi-feature, leave-one-out CV)

Fit measured cycles against feature sets over 4 consistent points; leave-one-out CV is the honest test of whether extra features *generalize*.

| feature set | LOO-CV MAPE | condition number |
|-------------|-------------|------------------|
| macs_only | 133% | 1128267218 |
| macs+act_bytes | 1277% | 1262561812 |
| macs+matmuls | 5613% | 3596599347 |

**Best CV:** `macs_only` (LOO-MAPE 133%). 
**Finding:** with only a handful of whole-model totals, the features are collinear (high condition number) and multi-feature fits do **not** reliably beat the single cycles/MAC term under cross-validation — per-component coefficients are **not identifiable** from whole-model data. Cycle-exact per-component calibration needs isolated microbenchmarks measured on RTL/spike: a compute-bound matmul, a memory/repeated-RHS matmul, a dispatch-heavy tiny-kernel sequence, `matmul_bias_requant_relu`, and `no_reuse_matmul`. Until then per-axis gap_closure stays gated.

_Status: the cycle-exact microbenchmark path needs the chipyard/spike (or FireSim) toolchain; where that is unavailable this is the precise scoped remaining measurement — not a fabricated coefficient._
