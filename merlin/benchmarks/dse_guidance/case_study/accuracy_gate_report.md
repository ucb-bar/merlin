# Quantization accuracy gate (measurable-now)

> Accuracy depends on the numerics, not the future hardware, so it is measured now to decide whether a low-bit candidate is legal. W8A8 (int8) vs fp32 golden, host interpreter (`docs/results.md`). Multi-tier gate: T1 cos>0.999 vs W8A8 ref, T2 cos>0.99 vs fp32 + top-1 argmax. No speedup is claimed.

| model | dtype | cos vs fp32 | rel | status |
|-------|-------|-------------|-----|--------|
| small_llama | int8 | 0.99993 | 0.013 | pass |
| tiny_llama | int8 | 0.99842 | 0.064 | pass |
| openvla | int8 | 0.99813 | 0.084 | pass |
| rdt2 | int8 | 0.99979 | 0.025 | pass |
| bitvla | int8 | 0.99995 | 0.010 | pass |

**Finding:** 5/5 measured int8 variants pass the W8A8 accuracy band — so the int8 low-bit residency/compute candidates are accuracy-legal. fp8/int4/fp4/fp6 are **unavailable** (not yet measured) and stay gated, not assumed.
