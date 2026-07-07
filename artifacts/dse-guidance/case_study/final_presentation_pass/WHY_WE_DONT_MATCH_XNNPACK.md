# Why we don't match XNNPACK despite mining their kernels — and why bitVLA still won

All numbers are K1 silicon, measured, cos-gated. Matmul buckets are timed on BOTH arms via `rdtime`
brackets (the `ours_board` backend), so this is no longer attribution-by-decode — it's measured.
Sources: `output/rvv_bench/{dispatch_breakdown_measured.json, k1_kernel_speedup_*.json, k1_e2e_fair_*.json}`.

## The one-paragraph answer
We mine the experts' kernels (disassemble XNNPACK/OpenBLAS RVV GEMM → a CCA → divergences → compiler
actions), yet our emitted matmul is **3.18× / 7.93× / 13.57× slower** (bitvla / openvla / rdt2) than
XNNPACK's. Two reasons, both measured: **(1) we never mined the #1 GEMM lever — register blocking (MR)**
— it came back `null` from every CCA, so we hand-chose MR=4 while the experts use MR=7/16 (now fixed,
below); and **(2) even at matched MR=7 our compiler-EMITTED code is worse than their hand-tuned asm** —
the accumulator doesn't stay in registers and the instruction density is lower, leaving a 2.23×–9.06×
residual that is *codegen quality*, not a missing decision. On the small-M VLAs (openvla M=17, rdt2 M=1)
the lever can't even lower. So "mining" got us the *decisions*; it did not get us their *codegen*.

---

## Part A — the fidelity-loss chain

### A.1 The pipeline: extract → divergence → action → schedule → emitted
- **Extract** (`merlin/python/merlin/kernels/cca.py:188` `lift_asm`): from the expert RVV asm we read
  `contraction_form` (vfmacc count), `accumulator_resident` (no spill-store in the K-loop),
  `register_block` MR (`_infer_register_block`: distinct accumulator vregs fed by `vfmacc.vf`), `lmul`,
  `widening`. **What the asm does NOT contain:** the instruction schedule / latency-hiding chain count,
  the operand *packing layout*, the loop-nest order / K-tiling depth, and the microarch tuning (VLEN /
  FU-latency constants). Those are lost at decode — a structural limit of CCA-from-asm.
- **Divergence + action** (`mine.py` → `cca_compare.py` → `action_catalog.py`): each captured facet that
  differs from ours becomes a typed `CompilerAction` (FLAG/KNOB/HEURISTIC/PASS/CODEGEN).
- **Schedule + emit** (`impr_features.py` transform schedules → `pipeline.py build_rvv_pipeline`): the
  action drives a transform-dialect schedule (tile + vectorize + form vfmacc), then LLVM-23 lowers it.

### A.2 The gap, layered and measured

**(1) Register blocking (MR) — ~5× of the gap, and it was a MINING BLIND SPOT.**
Every CCA had `register_block: null`. The kernel index *already had* MR=7 (`f32-gemm-7x4v-rvv.c → MR=7`),
but `mine.expert_cca_from_policies` built the expert CCA from a *policy table* that dropped MR, and
`cca_compare` omitted `register_block` from its keys — so the compiler never saw the experts block
registers. We hand-chose MR=4; experts use MR=7 (XNNPACK) / MR=16 (OpenBLAS). The measured MR sweep
(`k1_kernel_speedup_*.json`) shows MR is the dominant lever, ~5× consistently:

| model | XNN matmul | ours MR=1 | MR=4 | MR=7 |
|---|---|---|---|---|
| bitvla | 10.3 ms | 116 ms (11.2×) | 33 ms (3.2×) | **23 ms (2.2×)** |
| openvla | 21.0 ms | 507 ms (24.2×) | 144 ms (6.9×) | **100 ms (4.8×)** |
| rdt2 | 246.5 ms | 11096 ms (45.0×) | 3337 ms (13.5×) | **2233 ms (9.1×)** |

**FIXED this session:** `mine._expert_register_block` reads MR from the index, `cca_compare` emits an
MR-aware `compute.register_block` divergence (None == MR 1), `action_catalog` routes it to a raise-MR
KNOB. The compiler now *learns* register blocking from the experts.

**(2) The residual at matched MR=7 = CODEGEN QUALITY (2.23×–9.06×) — the real surprise.**
At MR=7 both kernels are structurally identical: 7 accumulators, LMUL=4, 7× `vfmacc.vf` per K-step, and
(per `output/kernels/ceiling/packing_residual.md`) the **same 2.0 loads/useful-FMA, unit-stride, zero
broadcast ladder**. Yet ours stays 2.23× slower (bitVLA cube) up to 9.06× (rdt2). Causes:
- (a) the schedule's `scoped-vectorize [MR,NR,1]` keeps **KC=1** — the K reduction is a scalar loop, not
  blocked in vector registers;
- (b) **the accumulator does NOT stay register-resident.** The post-bufferize hoist can't lift `memref`
  iter_args under RVV's fixed VLEN, so the C tile **round-trips through the stack every K-tile**
  (objdump-confirmed: `vl4re8.v`/`vs4r.v` of the accumulator per step) — `lift_asm` reads our
  `accumulator_resident=FALSE`, ~19× off the hand-intrinsic ceiling at 64³;
- (c) lower instruction density: ours ≈0.028 `vfmacc`/insn vs XNNPACK ≈0.04 (their inner loop is ~25
  insns; ours ~141 at MR=4).

This is the gap between *knowing* MR=7 and *emitting* an MR=7 kernel as good as theirs. The genuine closer
is a dedicated register-resident-accumulator **CODEGEN pass** (hold MR accumulators in vregs across K);
the transform-schedule features (`accumulator_resident_*` in `impr_features.py`) are *honest that they
can't express it* — see their "HONEST MEASURED STATUS" notes.

**(3) Small-M structural (openvla M=16–20, rdt2 M=1): MR>1 can't even lower.**
M not divisible by MR → LLVM-23 masked `vector.transfer_write` PipelineError → scalar fallback. Measured:
`vf_mr4` on openvla regressed to 2442 ms; M-padding to a multiple of 4 regressed to 8723 ms (the pad copy
dominates). So the experts' A-reuse is unreachable at the *kernel* level on the VLAs — it needs
*dispatch-level large-M batching* (group the many small-M projections into one large-M GEMM), a graph
rewrite, not a schedule.

**(4) B-packing is NOT the gap — disproven.** MR=7 with the resident B-pack (23.6 ms) ≈ without (23.0 ms)
on bitVLA; and `packing_residual.md` shows identical data-movement (2.0 loads/FMA, unit-stride) at every
openvla/rdt2 shape when both exclude the resident pack. Don't blame packing.

**(5) Not mineable from asm.** The instruction schedule / latency-hiding, the exact packing layout, and
loop fusion are structural facts the asm doesn't carry — a real limit of mining a CCA from disassembly.

### A.3 Gap attribution (one line)
`mining blind spot (MR, ~5×, now fixed) × codegen residual (2.2–9×, accumulator-not-resident + low
insn density) × small-M (lever can't lower)`. We learned the *decisions*; we don't yet emit the *codegen*.

---

## Part B — why bitVLA won (consolidated, ranked, measured)

Headline (`k1_e2e_fair_bitvla.json`, fair/cos-gated): ours **148.3 ms** vs XNNPACK-7x4v **167.3 ms (1.13×)**
/ OpenBLAS-16x8 **180.5 ms (1.22×)**.

1. **Whole-model integration (inlined-vs-routed) — the primary measured reason.** Ours inlined = 147 ms;
   the SAME kernel routed through a `func.call` shim = 177 ms → a ~30 ms **routing tax** the library MUST
   pay at all 15 call sites. Ours *routed* at MR=7 ties XNNPACK routed (169 ≈ 167 ms). **The win is
   integration — a library is a call you can't fuse across — NOT a better kernel.**
2. **M=32 is cleanly divisible** → clean register blocking + aligned vectorization, no masking/padding
   waste — unlike openvla M=17 / rdt2 M=1 where every schedule lever regressed (~8700 ms, measured).
3. **The `vf` schedule helps bitVLA but HURTS openvla** (~+400 ms non-matmul there) — it's bitVLA-specific.
4. **Honest caveat:** our GEMM kernel is **3.18× slower even on bitVLA** (matmul bucket 32.4 vs 10.2 ms).
   The win is whole-model scheduling + no routing tax + a divisible M — not kernel quality.

**Why it does NOT carry to openvla/rdt2:** small-M structural (the same levers regress), our matmul is
7.93×/13.57× slower, and they're dispatch-bound. The only ungeneralized lever — dispatch-level small-M
batching — is unmeasured/future.

### Claims to retract (do not repeat in talks)
- "Our GEMM kernel is competitive" — **false**, 3–14× slower everywhere (measured).
- "The bitVLA win is the non-matmul vf schedule" — **wrong mechanism**; it's inlined-vs-routed.
- "Spike instret proves our kernel is faster" — **mirage**; rank kernels on K1 silicon only.

---

## What would close the gap (actionable follow-ons; not done here)
1. **A register-resident-accumulator CODEGEN pass** (hold MR accumulators in vregs across K) — closes the
   biggest codegen residual; the transform-only features provably can't.
2. **Route the packing / loop-order divergences to actions** (currently mined-but-unrouted).
3. **Dispatch-level small-M batching** for the VLAs — the only path to the experts' A-reuse on M=17/M=1
   (and the only remaining lever for an openvla/rdt2 whole-model win, still unmeasured).
