# Fusion / epilogue / numerical-contract report

> Numerical structure beyond dtype capacity: the epilogue ops that follow each matmul, the accumulator/dequant/requant placement, and the fused abstractions structurally suggested. Patterns are `recovered_from_ir`; erased structure (dequant, requant, scale, packed layout, sparsity) is `unavailable`. **No speedup or low-bit performance is claimed.**

## Detected epilogue patterns (from IR)

| pattern | matmuls |
|---|---|
| matmul->bias | 663 |
| matmul | 369 |
| matmul->bias->scale->activation | 13 |
| matmul->bias->scale | 4 |
| matmul->scale | 2 |

## Directly-fused vs reshape-separated

- Of the 369 matmuls with **no directly-fused epilogue**, 366 are `reshape_separated_epilogue` — their output is reshaped (collapse/expand/transpose) before any elementwise op, so downstream ops (residual add, rotary, SiLU gating) are **not directly fused**. This is the bias-free LLaMA-style projection pattern. The reshape-distant ops are deliberately **not** labelled bias/scale: a reshape-distant addf/mulf is ambiguous (residual / rotary / gating / norm), so claiming it would over-state the fusion — it is reported as a layout-separated boundary instead.

## Where numerical structure is preserved vs erased

- **Preserved (`recovered_from_ir`):** the matmul→bias epilogue (addmm / addf) and the activation/clamp ops (erf / exp / maximumf) that follow it — the epilogue slot is real.
- **Erased (`unavailable` / lost):** dequant-before-matmul, requant, scale/zero-point metadata, and the packed low-bit weight layout — the capture is dequantized f32, so native low-bit compute and its scales are gone (consistent with the cross-zoo numerical-fidelity finding).

## Fused abstractions structurally suggested vs blocked

- **Suggested (IR-supported):** `fused_requant_epilogue`, `accumulator_object`, `accumulator_commit`, `activation_clamp_unit` — the epilogue slot, accumulator, and activation/clamp unit are backed by detected patterns.
- **Blocked (missing low-bit / scale / sparsity in the capture):** `fused_dequant_matmul`, `scale_object`, `packed_lowbit_tensor`, `resident_packed_weight_object`, `structured_sparsity_skip` — these need a low-bit capture (packed weights + scales) or accuracy measurement before a DSE can use them.

## How this changes future DSE search-space knobs

- adds an **epilogue-op-set** knob (which ops fuse onto the matmul output: bias / requant / activation / clamp) — the slot is proven present;
- adds an **accumulator-dtype** knob (i32 for int8 storage, derived);
- gates the **weight-format / dequant-in-load / scale-granularity** knobs behind a low-bit capture + accuracy measurement (int8 measured for the gated workloads; fp8/int4 unavailable).

**Caveat:** these are structural placement candidates with accuracy/performance measurements named per certificate. **No speedup and no low-bit performance is claimed.**
