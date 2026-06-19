# RVV kernel-mining → compiler-improvement: results

A reproducible pipeline that **abstracts why expert RVV kernels are fast into general compiler
capabilities**, then measures every change against the frozen baseline and the expert libraries
(OpenBLAS, XNNPACK) on real SpacemiT K1 silicon. This document consolidates the measured results.

> **One-line honesty contract.** Every number below is measured and correctness-gated (bit-exact on
> spike / cosine on the board). Hand-written kernels (`ours-intrinsic`) are **ceiling references**, never
> presented as compiler output. The baseline RVV compiler (`hand_v0`) is frozen; every improvement is a
> **default-off** feature that leaves the baseline byte-identical.

---

## 1. The pipeline (the research contribution)

Not a one-off tuning session — a deterministic, auditable chain. Each `merlin-rvv-mine` run mints a
versioned, immutable folder `mined_knowledge/rvv/mining_rvv_v{V}_{ts}/` with five YAML artifacts:

| artifact | content |
|---|---|
| `manifest.yaml` | provenance: corpora, policies, baseline run, git lineage, counts |
| `expert_cca.yaml` | **abstraction of what experts do**, lifted from their asm (no regex): `accumulator_resident`, `nr_is_vsetvlmax`, `contraction_form=fused_fma`, `lmul`, `vl_strategy`, … |
| `our_cca.yaml` | the same lift for our codegen — gaps visible field-by-field |
| `divergences.yaml` | every expert-vs-ours mismatch per CCA axis, **with evidence kernels** |
| `actions.yaml` | each divergence → a typed `CompilerAction` (PASS/HEURISTIC/KNOB/CODEGEN) with `target_seam`, `forkable_now`, `expected_effect` |

The abstraction (`kernels/cca.py`) is **target-agnostic** — `accumulator_resident` is a shared
compute-facet property across RVV / spatial (Gemmini) / NPU. Structured decoders (`kernels/decode/`)
read SEW/LMUL/vtype/loops/residency off the asm via a vtype state machine, **no regex**.

---

## 2. Compiler features delivered (all default-off, baseline frozen)

| feature | class | what it does | status |
|---|---|---|---|
| `fused_vfmacc_tiled` | PASS | tile→vectorize→outerproduct→**vfmacc** (replaces degenerate vfmul+vfadd) | **9.18× bitvla, 3.61× openvla e2e** |
| `accumulator_resident_ntail` | PASS | clamps batch_matmul `NR=min(NR,N)` → llama-style N=8 attention vectorizes instead of scalar fallback | whole-model-safe attention |
| `accumulator_resident_mtail` | HEURISTIC | clamps matmul `MR=min(MR,M)` → M=1 token-decode matmuls vectorize | fixes decode matmuls |
| `accumulator_resident_wholemodel` | PASS | both tail clamps inherent in one schedule + composition guard | **rdt2 2.35× e2e** |
| `vectorized_transcendental_activation` | PASS | `erf`/`exp`/`tanh` → compiler-emitted minimax polynomial (vfmacc Horner) instead of scalar libm | GELU 6.2× / sigmoid 3.2× vs our scalar (spike) |
| `intrinsic_microkernel` | CODEGEN | *ceiling reference only* — see §4 | honest: not compiler-emitted |

A **composition guard** prevents two full-schedule-replacement features from silently clobbering each
other (raises `CompositionError`); additive edits still layer.

---

## 3. Measured comparisons (three metric families — kept separate)

Figures: `output/kernels/ceiling/paper_e2e.png`, `paper_crossover.png`, `op_coverage.png`,
`all_comparisons.png`.

### 3a. Whole-model e2e on real K1 silicon (wall seconds, cos-gated)
| model | baseline | ours (compiler vfmacc) | speedup |
|---|---|---|---|
| bitvla | 2.517 s | 0.274 s | **9.18×** |
| openvla | 5.848 s | 1.619 s | **3.61×** |
| rdt2 | 73.71 s | 31.41 s | **2.35×** |

Swapping XNNPACK's hand RVV GEMM into bitvla's matmuls (same graph/weights/runtime) → 0.184 s
(**13.65×**), isolating **~1.49× of remaining matmul-codegen headroom**; the rest of the win is shared
runtime. smolVLA e2e is an honest `not_run` (SpacemiT clang backend crash, hits the frozen baseline too).

### 3b. Single-GEMM ceiling on K1 (rdtime ticks, inner-compute) — the crossover
`ours-intrinsic` (hand ceiling) beats **both** experts 32³→384³, then **OpenBLAS retakes the lead at
512³** (cache-blocking amortizes). The shipped compiler path (`ours-tiled`) trails the experts ~10×;
the baseline ~100× (never forms vfmacc). On spike the experts re-rank (instruction-count proxy can't
see VLEN/lane utilization) — which is exactly why the board numbers are the authority.

### 3c. Beyond GEMM (op coverage vs XNNPACK; OpenBLAS is BLAS = GEMM-only)
| op | result |
|---|---|
| conv2d (im2col→GEMM) | rides the GEMM win (vfmacc 2.4× over baseline) |
| GELU / sigmoid (f32) | **the surfaced gap**: baseline lowers `erf`/`exp` to scalar libm; now vectorized (§2) — silicon head-to-head vs XNNPACK in §3d |
| int8 GEMM (W8A8) | correct (cos>0.99); the vwmacc win is e2e-amortized, ~200× behind in an isolated tight loop |
| depthwise conv | `not_run` — no depthwise primitive (regular conv only via im2col) |
| attention bmm (N=8) | ours-vs-ours (no library primitive); vfmacc regresses 4.8× at tiny N (tail dominates) |

### 3d. Activation on silicon — partial win isolated, regression whole-model (corrected)
The `vectorized_transcendental_activation` feature, re-measured on K1 (commit 284291c):
- **Isolated GELU/sigmoid (the op it targets):** the genuine polynomial (feature ON) **vectorizes and is
  2.3–3.7× faster than our scalar**, but still **~3.6–4.6× behind XNNPACK's hand polynomial** (GELU
  983 vs 270; sigmoid 559 vs 139) — narrows the gap from a previously-misreported ~11–18× (that column
  was scalar mislabeled as vectorized), not closed. Accuracy within the approximation band (errors=0 @2e-3).
- **Residual cause:** our polynomial lowers to separate `vfmul.vv`+`vfadd.vv` (fmuladd=0 — **no fused
  vfmacc**), more ops/element than XNNPACK's tuned rational kernels. (Fusing to vfmacc is the next lever.)
- **Whole-model (bitvla/openvla):** the schedule edit tiles *every* `linalg.generic[16]`, which breaks
  on non-activation generics → **scalar fallback → 0.76× / 0.68× regression**; and it does not compose
  with the matmul feature (both `schedule_replace` → `CompositionError`). So whole-model it is **not a
  win** and stays default-off. A fix for the whole-model scalar-fallback is in progress.
Full detail: `output/rvv_bench/k1_e2e_activation.md`.

---

## 4. What we honestly cannot (yet) do

- **Accumulator-resident GEMM micro-kernel via the transform-dialect compiler.** The schedule path is
  ~19× off the hand ceiling on the spike proxy (the K-loop still spills the accumulator to stack). So
  `intrinsic_microkernel` is a **labeled hand-written ceiling reference**, and the gap is recorded as a
  `forkable_now=False` CODEGEN work-item ("needs a dedicated micro-kernel codegen pass"). On real silicon
  whole-model this gap is only ~1.49× (matmul is one component) — see §3a.
- **smolVLA whole-model on the board** — blocked by a SpacemiT clang backend crash (not our feature).

---

## 5. Reproduce

```
merlin-rvv-mine  --target rvv --op matmul --mined <prior-run>   # mint the 5-YAML mining run
merlin-rvv-report --mined <run> --out report.md                 # auditable evidence report
.venv/bin/python scripts/plot_paper_style.py                     # paper figures
.venv/bin/python scripts/plot_rvv_comparisons.py                 # full comparison + op-coverage
```

Board runs: `scripts/k1_e2e_*.py`, `scripts/k1_cross_framework*.py` (SpacemiT clang, riscv64, rdtime).
Baseline immutability guarded by `tests/test_impr_features.py` (byte-identical when features=∅).
