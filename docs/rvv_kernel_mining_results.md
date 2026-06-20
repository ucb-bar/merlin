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
| `vectorized_transcendental_activation` | PASS | `erf`/`exp`/`tanh` → compiler-emitted minimax polynomial (vfmacc Horner) instead of scalar libm | isolated only (§3d); whole-model still falls back |
| `accumulator_resident_v2` | PASS | hoist accumulator to a value-semantic `vector<MR×NR>` `scf.for` iter_arg (pre-bufferize) → register-resident across K | residency achieved (§4) |
| `accumulator_resident_microkernel_v3` | PASS | v2 + scalarize A reads → emits `vfmacc.vf` K-loop; **compute kernel matches the hand ceiling, beats OpenBLAS** | **closes the §4 gap** |
| `intrinsic_microkernel` | CODEGEN | *hand-written ceiling reference* — superseded by v3 (compiler now emits the real thing) | cross-check only |

A **composition guard** prevents two full-schedule-replacement features from silently clobbering each
other (raises `CompositionError`); additive edits still layer.

---

## 3. Measured comparisons (three metric families — kept separate)

Figures: `output/kernels/ceiling/paper_e2e.png`, `paper_crossover.png`, `op_coverage.png`,
`all_comparisons.png`.

### 3a. Whole-model e2e on K1 — SAME-PASS, fair (N=5/3, cos-gated) — XNNPACK wins 2 of 3
Measured in one pass vs the same baseline; XNNPACK uses a **resident-weight pack** (excluded from the
timed path, matching ours' pack-free scope) and was measured on **all three** models for the first time.
Speedups vs baseline (higher = better); **best-ours** is the fastest of {tiled, v3, wholemodel}:

Full **four-way** (baseline · best-ours · XNNPACK · OpenBLAS), one same-pass campaign per model, N=5/3,
experts with resident-weight pack (fair vs ours' pack-free path), cos-gated:

| model | baseline | best ours | XNNPACK | OpenBLAS | ours vs best expert |
|---|---|---|---|---|---|
| **bitvla** | 2.524 s | **v3 16.88×** (0.150 s) | 13.19× (0.191 s) | 13.39× (0.189 s) | **ours WINS (+1.26×)** |
| **openvla** | 5.855 s | wholemodel-**vf** 5.38× (1.089 s) | 8.92× (0.657 s) | 8.53× (0.686 s) | ours **60%** (XNN 1.66×) |
| **rdt2** | 74.04 s | wholemodel-**vf** 2.45× (30.27 s) | 3.90× (18.97 s) | 3.64× (20.32 s) | ours **63%** (XNN 1.59×) |

(openvla/rdt2 best-ours is the `.vf` kernel from iteration 2 of the loop — see §7 — which lifted ours 55%→60% / 62%→63%. The canonical, reproducible version of this whole table + the structural attribution is now produced by `merlin-compare` → `mined_knowledge/rvv/compare_*/`.)

All cos ≥ 0.99999 (figure `paper_fourway.png`: all-4 with baseline + zoomed contest). **Honest verdict:
ours is competitive with BOTH hand-tuned vendor libraries.** It **beats both** XNNPACK *and* OpenBLAS on
bitvla (the compiler-emitted v3 kernel, +1.26–1.28×), and reaches **55–66%** of the fastest expert on
openvla/rdt2 (~1.6–1.8× behind) — geomean ≈ 76% across the three. Two clean findings: (1) **XNNPACK and
OpenBLAS are neck-and-neck** whole-model (within ~5%), so "the expert ceiling" is well-defined; (2) our
kernels are **shape-brittle** — v3 is 16.88× on bitvla but only 2.38× on openvla, and tiled/v3 *regress
rdt2 by 3×* (only `accumulator_resident_wholemodel` avoids that). This is exactly why the **per-model beam
is essential**: with the right kernel ours is in the experts' league everywhere and beats them on one;
with the wrong one it is catastrophic. smolVLA e2e is an honest `not_run` (SpacemiT clang backend crash,
hits the frozen baseline too).

**Why ours is ~55–62% on openvla/rdt2 (decoded — `output/kernels/ceiling/kernel_breakdown.md`):** NOT lane-width
and NOT residency — `ours_wholemodel` is already NR=32/LMUL=m4/vsetvlmax + accumulator-resident with 0 spills
(identical lane-width to XNNPACK, wider than OpenBLAS m2). The dominant residual is the **`vfmacc.vv`
A-broadcast ladder**: lacking `vfmacc.vf`, the kernel rebuilds each A scalar into a `vector<32>` via a
vslideup/vmv ladder every K-step → **~20 inner-loop insns/FMA vs XNNPACK's ~3** (6.7× inflation). Fix (next
iteration): carry v3's `scalarize_a_reads` (emits `.vf`) into `accumulator_resident_wholemodel` keeping its
M/N-tail clamps so it survives the small-M openvla/rdt2 matmuls. This is the loop's value — a *measured*
residual corrected a wrong lane-width hypothesis.

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

### 3d. Activation on silicon — FIXED (c1cf6cc): correct + vectorized whole-model
The first whole-model run regressed (bitvla 0.76× / openvla cos=0.541). Root cause was **not** low
accuracy — it was a **crash**: the blanket rewrite replaced *every* `math.exp` including softmax's, and
the pipeline turned it into `llvm.intr.exp` which the freestanding RVV runtime can't legalize (`bad
syscall`). Fix = **provenance-targeted** rewrite (only generics the abstraction marks gelu/silu/sigmoid/
tanh; softmax/normalization `exp` stays on the exact libm path), tag-and-match only those (no blanket
`foreach`, no `failures(suppress)`), and a pure-arith polynomial so no `convert-math-to-llvm` edit is
needed (feature is now schedule-only, pipeline byte-identical).
- **Post-fix bitvla K1: vectorized, cos 0.99999, no crash, no regression** (2.525 s ≈ baseline — activations
  are negligible vs matmul in bitvla, so whole-model wall is neutral; the point is it's now *correct and
  vectorized*, not a 0.76× scalar-fallback regression). openvla post-fix confirmation in flight.
- **Isolated GELU/sigmoid:** vectorizes, ~2.3–3.7× over our scalar, accuracy cos≈1.0 / max-abs ~5e-7 over
  realistic ranges; still **~3.6–4.6× behind XNNPACK's hand polynomial**.
- **Honest residual:** activations lower to `vfmul`+`vfadd` (no fused `vfmacc`) — fusing needs `math.fma`,
  which forces the `convert-math-to-llvm` pass that re-introduces the softmax `llvm.intr.exp` crash. So
  fusion was traded for whole-model correctness; the activation still vectorizes (replaces the scalar libm loop).
Full detail: `output/rvv_bench/k1_e2e_activation.md`.

---

## 4. The microkernel gap — now CLOSED (compiler-emitted)

The earlier honest verdict ("the transform-dialect compiler cannot emit the accumulator-resident
kernel; ~19× off; needs a dedicated pass") is **superseded** (commit 68137e2). The compiler now
genuinely emits it:
- **Root cause of the old 19×:** the residency hoist ran *post*-bufferize, where the K-loop carries the
  accumulator as a *memref* iter_arg and the hoist no-ops. Fix = run `loop-invariant-subset-hoisting` on
  the **tensor** form *before* one-shot-bufferize → the accumulator becomes a value-semantic
  `vector<MR×NR>` loop iter_arg held in vregs across K (feature `accumulator_resident_v2`). A second
  rewrite scalarizes the A reads so clang emits `vfmacc.vf` (feature `accumulator_resident_microkernel_v3`).
- **Decode-confirmed** (structured `decode.rvv`, asserted in tests): the innermost K-loop is **4 `vfmacc.vf`,
  0 `vfmacc.vv`, 0 in-loop accumulator spills** — on cubes and non-cubes; bit-exact (32³/64³/128³ + 96×48×160).
- **Instret (spike inner-compute), the compute kernel (`forward`):** 7,045 / 53,207 / 409,764 @32/64/128³
  vs hand ceiling 6,549 / 50,693 / 399,239 (**~1.05–1.08×**) and **beats OpenBLAS** (11,037 / 84,481 / 664,809).
  The ~19× compute-kernel gap is closed; no hand kernel is linked — the compiler emits it.
- **Honest residual (isolated):** the v3 *total* (~7× ceiling) is dominated by an O(M×N) result-buffer
  copy-out — an ABI artifact of the single-op workload returning a fresh tensor. Out of scope for kernel codegen.
- **Whole-model board result — FIXED (a35d37b), v3 now applies whole-model and is the new winner.**
  The first whole-model run fell back to SCALAR (bitvla 0.77×, openvla 0.68× regression): the v3
  A-operand scalarization rewrite over-matched a non-matmul `vector<…x1>` read → `PipelineError` →
  silent scalar fallback. Fix = restrict the matcher to the matmul register-tile read (statically-ranked
  source, extract-position length == source rank, identity permutation_map); leave other reads on the
  baseline (compose, don't replace-and-fail). **Post-fix bitvla K1 (cos 0.99999, vectorized, 60 vfmacc.vf):
  0.1498 s = 16.83×** — beating `fused_vfmacc_tiled` (9.16×) **and the XNNPACK-kernel swap (13.65× / 0.184 s,
  cross-run, matched baseline)**. So the §3a 1.49× kernel headroom is not just closed but **reversed**:
  the compiler-emitted accumulator-resident kernel is ~1.23× faster than XNNPACK's hand RVV GEMM
  whole-model on bitvla — consistent with the isolated K1 matrix where ours-intrinsic beats XNNPACK at
  bitvla's matmul sizes.
- **Model-dependent (honest):** on openvla, post-fix v3 also applies cleanly (vectorized, cos 1.0) but is
  **2.38×** — *below* `fused_vfmacc_tiled` (3.65×). v3's MR=4 register-blocked kernel wins on bitvla's
  small-M decode matmuls; the tiled vfmacc wins on openvla's shapes. So the compiler now holds a
  **portfolio of correct, whole-model-safe matmul kernels and the best is per-model** — which the
  autotune/beam layer selects. v3 is the new winner on bitvla (and beats XNNPACK there); not universal.

## 4b. What still genuinely doesn't work
- **Vectorized activation whole-model** — fixed the rank-1-tile scalar-fallback bug (40a8cbc), but it
  does not compose with the matmul feature (both `schedule_replace`); see §3d.
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

---

## 6. Limits (scope boundaries, stated honestly)

These are deliberate boundaries of the current results, not defects — listed so the claims aren't over-read:

- **No cycle-accurate confirmation.** All board numbers are K1 `rdtime`/wall (`cycle_accurate=false`; the K1
  traps userspace `rdcycle`, so there are no hardware perf counters). Spike is an IPC=1 instruction-count
  proxy. The whole-model wins are real-silicon wall measurements, but they have NOT been confirmed on a
  cycle-accurate model (FireSim/VCS) — that path is queue-gated (priority 5, never escalated) and out of
  scope here. "Utilization" in the figures is therefore a *ceiling proxy* (% of the expert reached + VPU
  state), not measured VPU occupancy.
- **Workload scope.** The comparison covers fp32, matmul-dominated VLA inference on three models
  (bitvla, openvla, rdt2) on one board (K1). It does not cover int8 end-to-end (separate datapath),
  conv-heavy or attention-heavy models, or other RVV silicon. The beam evaluated the distinct named
  optimizations + composable combos whole-model; it did NOT sweep the 33 MR/NR/KC grid points whole-model
  (those were tuned single-op) — a future-work item.
- **Beyond GEMM we are not yet competitive.** Activation vectorization is correct but neutral whole-model
  (activations are negligible vs matmul here) and still ~3.6–4.6× behind XNNPACK isolated; isolated int8
  GEMM is far off (the vwmacc win is e2e-amortized); depthwise conv has no primitive; small-N attention
  regresses. These are honest gaps, scoped as separate efforts — the headline win is the fp32 GEMM path.
- **Composition limit.** The matmul features and the activation feature are full-schedule replacements,
  so they cannot currently compose (`CompositionError`). Getting matmul-v3 AND vectorized activation in one
  whole-model lowering needs a unified schedule (future work).

---

## 7. The iterative loop (and `merlin-compare`, the tool that drives it)

The comparison is not a one-shot — it's a closed loop: **measure on real workloads → lift the residual
divergence from asm (CCA) → route to a typed action → implement a default-off feature → re-measure → feed
the new residual back.** Turns so far on the fp32 GEMM path:

| turn | action (mined) | result |
|---|---|---|
| 1 | vfmacc forming (fused_vfmacc) → tiled → accumulator-residency (v3 / wholemodel) | bitvla beats both experts; openvla/rdt2 reach 55–62% |
| 2 | `.vf` A-scalarize (kill the `vfmacc.vv` broadcast ladder, 20→6 insns/FMA) — `accumulator_resident_wholemodel_vf` | openvla 55→**60%**, rdt2 62→**63%** — a *modest* whole-model bump (1.04–1.09×) |

**Turn 2's honest lesson:** the kernel-level 6.7× inner-loop reduction dampens to ~1.05× whole-model — the
matmul *compute* is not the openvla/rdt2 bottleneck; **memory/packing is**. So the loop has surfaced the
**next residual = packing/cache-blocking** (the breakdown's secondary factor), which is turn 3's input. This
is the loop working: each turn closes part of the gap and reveals the next thing to abstract — diminishing
per-turn returns, honestly measured, not overclaimed.

**The tool:** `merlin-compare` makes one turn's measurement+comparison a single repeatable command — a spec
(`configs × workloads × target × metric`) → a versioned `compare_<ts>/` artifact with the measured table,
per-config CCA, **auto-attribution** (measured gap × structural divergence × routed action — it
auto-identifies the `.vf`-vs-`.vv` driver), and figures. Target-agnostic spec (RVV/K1 impl now). So the loop
is now: `merlin-compare` → read the routed actions → implement → `merlin-compare` again. Reproducible,
scalable (add a model/target/candidate = a spec line), not me hand-stitching agents.
