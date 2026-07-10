# Corpus coverage audit — do we have all the benchmarks?

Answering "double-check we have all the inputs/benchmarks to compare." The golden reference is the
bareMetalC Gemmini test suite (**55 .c tests**); we want "more or less all of them" + distinctive
model-derived kernels. Below: what we have, what each approach can run, and the extension plan.

## Have now (27 kernels, 4 op categories — EXTENDED)
- **matmul (20):** 9 golden-set shapes (single-tile, multi-tile sq, rect, deep-K, wide-N, tall-M,
  acc_scale→i8, relu→i8, 128³) + 8 model-harvested (smolvla/openvla/tiny_llama: MLP, attention-proj,
  lm_head) + 3 attention QK^T/PV (`semantic attn_qk/attn_pv`, the Gemmini-relevant transformer op).
- **conv2d (4):** std 3×3, 16ch 3×3, 1×1 pointwise, strided 3×3 (golden=conv.c family; NOTE spike ISS
  skips conv → conv golden is verilator-only).
- **movement (3):** mvin→mvout identity, single-tile/multi-tile/wide (golden=mvin_mvout.c).
Generators: `gen_golden_kernels.py` + `harvest_model_kernels.py` + `gen_category_kernels.py`. All 27
schema-validate + have deterministic goldens (capsule_golden supports matmul/conv2d/movement/attn).
Runnable: baseline/merlin/native via capsule_runner for ALL; golden(C-lib) for matmul+attention now;
golden conv/movement C templates = the remaining run_perf_bench wiring. IREE arm path in IREE_ARM_STATUS.md.

## bareMetalC categories (55 tests) vs our coverage
| category | #tests | covered? | which approaches CAN run it |
|---|---|---|---|
| matmul (ws/os/spad, tiled, At/Bt, rect, full_C, low_D) | ~14 | ✅ (shapes) | all 4 + IREE |
| **conv2d** (basic, dw, rect, stride, dilation, pool-fused, layout-transpose) | 20 | ❌ MISSING | golden(conv.c), baseline/merlin (im2col, per capsule_bench B3/B4), IREE (--iree-gemmini-enable-conv2d) |
| **movement** (mvin_mvout + stride/acc/block/scale variants) | 13 | ❌ MISSING | golden(mvin_mvout.c — already in baremetalc_corroborate), baseline/merlin (movement op, capsule_bench A1) |
| matmul+fusion (softmax/igelu/layernorm) | 3 | ❌ | golden only (others lack the fused epilogue) |
| elementwise/residual (matrix_add, resadd) | 3 | ❌ | golden; baseline/merlin maybe (VECTOR_MAP add) |
| pooling (global_average, conv_with_pool) | 2 | ❌ | golden; others partial |
| transpose (transpose, transpose_scale) | 2 | ❌ | golden; movement-class |
| utility (aligned/padded/raw_hazard/counter) | ~4 | n/a (not perf kernels) | — |

## Extension plan (priority by cross-approach competability)
1. **conv2d (highest value)** — all of golden/baseline/merlin/IREE can do it → a true multi-way perf
   comparison. Add ~4 shapes: standard 3×3 (the capsule_bench B3 shape), depthwise, rect, strided.
   Golden = `conv.c`/`conv_perf.c` parametrized; baseline/merlin via the capsule conv2d→im2col path
   (capsule_runner already supports it); IREE via `--iree-gemmini-enable-conv2d`.
2. **movement (mvin/mvout)** — golden = `mvin_mvout.c` (already built by baremetalc_corroborate);
   baseline/merlin via the movement op (capsule_bench A1). 2–3 shapes (identity, strided, acc).
3. **residual-add / matrix_add** — golden + (baseline/merlin if VECTOR_MAP add supported). 1–2.
4. **fusions/pooling/transpose** — golden-only (or golden + partial) → report as golden-reference
   coverage where other approaches lack the op (honest "n/a" cells).
5. **More model-derived** — already have 8; can broaden to more models (rdt2, bitvla, pi05) for more
   distinctive shapes if wanted.

Net: the matmul comparison (17 kernels × 4–5 approaches) is the first complete deliverable; conv +
movement are the next corpus extensions to reach "all the golden roccm tests," using the SAME runner
(capsule_runner already handles conv2d + movement) + golden C templates (conv.c / mvin_mvout.c).

## Model-derived candidates — the op reality (model2MLIR)
The supported models (tiny_llama, smolvla, openvla, …) are **transformer-heavy ViT/LLM, NOT CNNs**.
Their op families (by `prov.op`): `matmul`, **`batch_matmul`** (attention QK^T / PV), `softmax`,
`layer_norm`, `add`/`mul` (elementwise), `reduce_mean`, `dequantize` — and **almost no conv2d** (patch
embeds are rare; vision encoders here are attention-based). So the **Gemmini-relevant** model ops are
**matmul + attention-matmul** (capsule_bench models attention QK/PV as `op: matmul` with
`semantic: attn_qk/attn_pv`). Our 8 harvested matmul kernels already span MLP (wide-K) + attention-
projection shapes; we should ALSO add explicit **QK^T / PV attention** shapes harvested from the models'
`batch_matmul` ops (flattened to 2D matmul). Conv/movement/pooling/residual/transpose are therefore
**golden(bareMetalC)-driven**, not model-driven — which is expected and correct for these workloads.

**capsule_golden / capsule_runner op support** (what runs through golden+baseline+merlin+native, i.e.
multi-way): `matmul`, `conv2d` (im2col), `attention_qk`, `attention_pv`, `movement`. Categories WITHOUT
capsule support (residual/pooling/transpose/fusions softmax-gelu-layernorm) are **golden-C-reference
only** → reported with honest `n/a` for the MLIR arms that lack the op.

## Extension execution split
- **Multi-way (golden+baseline+merlin+native[+IREE]):** conv2d (golden `conv.c`), movement (golden
  `mvin_mvout.c`, already in baremetalc_corroborate), attention QK/PV (model-harvested). Author capsules
  (op supported) + golden C templates; runner already handles them.
- **Golden-reference-only:** residual-add (`resadd.c`), pooling (`global_average.c`), transpose, fusions
  (softmax/igelu/layernorm) — golden C gives the reference cycles/util; MLIR arms = n/a (op unsupported).
