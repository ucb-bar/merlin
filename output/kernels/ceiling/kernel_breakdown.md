# Why the whole-model-safe RVV GEMM reaches only ~55–62% of XNNPACK/OpenBLAS on openvla/rdt2 — a static-asm decode attribution

**Method.** Static decode of the emitted RVV asm — no perf counters. The K1 has no userspace
performance counters (it traps `rdcycle`; only a 24 MHz `rdtime` wall is delegated), so this is a
**static-asm + instruction-mix attribution**, NOT a cycle profile. Where a per-instruction cycle cost
would be needed to fully separate two effects, that is stated. The gap *magnitude* anchors are the
already-measured whole-model walls (`output/rvv_bench/k1_h2h_fair_openvla.md`: ours-wholemodel /
XNNPACK = **0.5544×** on openvla, same-pass, resident-weight pack excluded for both); this note
explains the *mechanism* of that ~half-gap by decoding the kernels.

Decoder reused as-is (no rebuild): `merlin/python/merlin/kernels/decode/{objdump,rvv}.py`
(RawInsn + vtype state machine, loop back-edges, vfmacc histogram) and
`merlin/python/merlin/kernels/cca.py::lift_asm` (accumulator_resident, register_block MR, NR/LMUL,
nr_is_vsetvlmax). Harness: `scripts/decode_kernel_breakdown.py` (throwaway). Raw decode JSON:
`output/kernels/ceiling/kernel_breakdown_decode.json`.

**Kernels decoded.** ours `accumulator_resident_wholemodel` (the kernel behind the 0.5544× openvla
number), `fused_vfmacc_tiled`, `accumulator_resident_microkernel_v3` — each lowered through the exact
runner path (transform schedule + feature) to `model.o`, K-loop decoded. Experts: XNNPACK
`xnn_f32_gemm_ukernel_1x4v__rvv` and OpenBLAS `openblas_sgemm_kernel` (= `sgemm_kernel_8x8_zvl128b`),
decoded at the ukernel symbol of the bare-metal ELF the ceiling drivers build.

**Shapes.** A clean cube (64³) plus REAL throughput-bound matmul dims pulled from the lowered
models (`output/{openvla,rdt2}_fp32_consistent/model.mlir`, `linalg.matmul` ins/outs):
- openvla `17×192×576` (attn/MLP proj, small-M=17), `20×128×512` (action-head MLP up, M=20)
- rdt2 `28×1024×1024` (workhorse attn proj), `28×1024×2816` (MLP up) — M=28, large K/N.
NR/LMUL are reported as **lanes @ VLEN=256** (the K1 board; `NR = VLEN/SEW × LMUL`). The spike harness
is VLEN=128, so an `e32,m4` block is 16 lanes on spike but **scales to 32 on the K1 with no recompile** —
the kernels are VL-agnostic, so the lane count is set by LMUL, not a baked constant.

---

## Per-kernel decode table

| kernel | shape | SEW | LMUL | NR (lanes@VLEN256) | nr_is_vsetvlmax | MR | acc-resident | vfmacc form | acc spills (K-loop) | packed | K-loop insns / useful-fma |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **XNNPACK** 1x4v | ukernel | 32 | 4 | **32** | **yes** (vsetvli) | 1 | **yes** | **vfmacc.vf** | 0 | **yes** (goi pre-pack) | **~3** |
| **OpenBLAS** 8x8 | ukernel | 32 | 2 | 16 | no (vsetivli) | **16** | **yes** | **vfmacc.vf** | 0 | **yes** (ncopy/tcopy) | ~7 (60 fma / 8x8 block) |
| **ours_wholemodel** | cube_64 | 32 | 4 | **32** | **yes** | 1 | **yes** | **vfmacc.vv** | 0 | no (streams model layout) | **~20** |
| **ours_wholemodel** | openvla 17×192×576 | 32 | 4 | **32** | yes | 1 | yes | vfmacc.vv | 0 | no | ~21 |
| **ours_wholemodel** | openvla 20×128×512 | 32 | 4 | **32** | yes | 1 | yes | vfmacc.vv | 0 | no | ~21 |
| **ours_wholemodel** | rdt2 28×1024×1024 | 32 | 4 | **32** | yes | 1 | yes | vfmacc.vv | 0 | no | ~20 |
| **ours_wholemodel** | rdt2 28×1024×2816 | 32 | 4 | **32** | yes | 1 | yes | vfmacc.vv | 0 | no | ~20 |
| ours_v3 | cube_64 | 32 | 4 | 32 | no | 4 | yes | vfmacc.vf | 0 | no | (4-fma block) |
| ours_v3 | rdt2 28×1024×1024 | 32 | 4 | 32 | no | 4 | yes | vfmacc.vf | 0 | no | (4-fma block) |
| ours_v3 | rdt2 28×1024×2816 | 32 | 4 | 32 | no | 4 | yes | vfmacc.vf | 0 | no | (4-fma block) |
| ours_v3 | **openvla 17×192×576** | 32 | 1 | **8** | yes | 5 | **NO** | vfmacc.vv | **118** | no | degraded (M-tail) |
| ours_vfmacc_tiled | cube_64 / rdt2 | 32 | 4 | 32 | no | — | (fully unrolled) | — | 0 | no | 64-fma unrolled body |

(`fused_vfmacc_tiled` is a fully-unrolled constant-64-fma body — `_fma_loop` finds the enclosing
N/M back-edge, not a K back-edge, so its MR/residency read as N/A; it is an older fork, not the
whole-model headline. `ours_v3` is the isolated micro-kernel that hits the LLVM-23 M-tail
PipelineError on M=17 → degraded NR=8 / non-resident / 118 spills; this M-tail failure is exactly
what `accumulator_resident_wholemodel`'s inherent `MR_mm=1` clamp fixes — note wholemodel decodes
clean at 17×192×576.)

---

## Attribution of the openvla/rdt2 ~half-gap (vs the headline kernel, `accumulator_resident_wholemodel`)

The decode **rules out the two effects that the brief hypothesised as dominant**, for the
whole-model kernel specifically:

- **Lane-width (NR): NOT the gap.** ours_wholemodel emits `e32, m4` with a `vsetvli` VL-loop →
  `nr_is_vsetvlmax = True`, **NR = 32 lanes @ VLEN=256** — *identical* to XNNPACK (also `e32,m4`,
  NR=32). It is in fact **wider** than OpenBLAS (NR=16, `e32,m2`). The `nr_is_vsetvlmax` divergence
  the brief expected does not exist here: the whole-model kernel already tracks vsetvlmax. So
  lane-width contributes **~0×** of the gap vs XNNPACK (and is a *net positive* vs OpenBLAS).
- **Accumulator residency: NOT the gap.** The K-loop holds C in `v8` across the whole K reduction
  and stores it once (`vse32.v`) after the loop. `accumulator_resident = True`, **0 in-loop acc
  spills** (`vsNr`/`vlNre`) at every openvla/rdt2 shape. This matches both experts. Residency
  contributes **~0×**.

**The whole gap is the A-operand broadcast form + packing/blocking:**

- **Dominant factor — `vfmacc.vv` + a per-K broadcast ladder (the `.vf` deficit).** XNNPACK reads
  the A scalar straight from an FP register and broadcasts it for free inside `vfmacc.vf`, so its
  K-loop is **~3 instructions per useful FMA** (`flw` + `vle32.v` + `vfmacc.vf`). ours_wholemodel
  cannot emit `.vf`: it materialises the A element as a full `vector<NR>` with a **vslideup/vmv
  ladder** every K step, then does `vfmacc.vv`. Decoded K-loop body (rdt2 28×1024×1024, identical
  shape-to-shape): **1× vfmacc.vv, 4× vslideup.vi, 4× vmv{1,2,4}r.v, 2× vle32.v, 6× vsetivli** =
  **~20 instructions per useful FMA**. That is a **~20/3 ≈ 6.7× instruction-count inflation in the
  inner loop vs XNNPACK**, and it is constant across all five openvla/rdt2 shapes (20–21 ops/fma) —
  a purely structural defect, not shape overfit. On the functional spike (IPC=1) this maps ~linearly
  to the gap; on the K1 the slide/move ladder ops are cheaper than a vfmacc but still issue, so the
  realised gap is smaller than 6.7× — consistent with the measured **1/0.5544 ≈ 1.8×** whole-model
  wall once the *non*-matmul work (norms, elementwise, the O(MN) result copy) dilutes the matmul
  fraction. **This `.vf`→`.vv` broadcast ladder is by far the largest attributable component of the
  openvla/rdt2 gap.**
- **Secondary — packing / register-block reuse.** Both experts run on **pre-packed** panels
  (XNNPACK goi weight pack; OpenBLAS ncopy/tcopy) hoisted out of the timed region, so their inner
  load is a single contiguous `vle32.v` with unit stride. ours_wholemodel **streams from the model's
  native layout**: the K-loop carries strided address arithmetic and the slide ladder partly exists
  *because* the A column is not pre-gathered into a broadcast-friendly panel. OpenBLAS additionally
  uses an **MR=16 register block** (60 vfmacc per 8×8 tile → high A/B reuse, ~7 insns/fma amortised);
  ours_wholemodel is **MR=1** (no A-reuse across output rows). MR=1 is *not* a gap vs XNNPACK (also
  MR=1), but it is a gap vs OpenBLAS at large K.
- **Spills: zero.** No in-loop accumulator spill at any openvla/rdt2 shape (the `vfmacc_packed`
  fork's NR=32-overflow spills are absent here). Spills contribute **0×** to this kernel's gap.

**Quantified split of the ~1.8× openvla/rdt2 whole-model gap (static-mix attribution, K1):**

| component | vs XNNPACK | vs OpenBLAS | evidence |
|---|---|---|---|
| lane-width (NR) | 0× (tie, NR=32) | favourable (32 vs 16) | both `e32,m4`, vsetvlmax |
| accumulator residency | 0× (tie) | 0× (tie) | acc-resident, 0 spills, all shapes |
| **A-broadcast `.vf`→`.vv` ladder** | **dominant (~6.7× inner-loop insn inflation)** | dominant | 20 vs 3 ops/useful-fma |
| packing / MR register block | minor (pack only) | moderate (pack + MR=16 reuse) | strided stream vs pre-packed panel; MR 1 vs 16 |

> **Honest limit of separation.** Without a per-instruction cycle counter on the K1 (none in
> userspace), I cannot split the residual **packing** cost from the **`.vv`-ladder** cost *in cycles* —
> the slide ladder exists partly *because* A isn't pre-packed into a broadcast panel, so the two are
> entangled at the asm level. What the static decode establishes unambiguously is the **ranking**: the
> `.vf`→`.vv` broadcast ladder is the leading term (6.7× inner-loop instruction inflation, shape-
> invariant), packing/MR is second, and lane-width + residency are non-factors for this kernel. A
> Saturn-RTL/FireSim run would be needed to turn the 6.7× instruction ratio into an exact cycle ratio.

---

## Implied fix

**Emit `vfmacc.vf` (scalar-from-FP-register broadcast) instead of `vfmacc.vv` + the vslideup/vmv
ladder in the whole-model-safe kernel.** This is the single change that collapses the dominant
component: it removes ~17 of the ~20 inner-loop instructions per FMA (4 vslideup + 4 vmv + the extra
vsetivli/vle to build the broadcast), taking the K-loop from ~20 to ~3 ops/fma — the XNNPACK shape.
The mechanism is already proven in-tree: `accumulator_resident_microkernel_v3`'s `scalarize_a_reads`
rewrite emits exactly `vfmacc.vf` (the v3 cube/rdt2 rows above: vfmacc.vf, MR=4, 0 spills, at the
hand ceiling). The whole-model blocker is only that v3's MR=4 M-tiling trips the LLVM-23 masked-
transfer_write PipelineError on small-M (M=17/20/28) → the degraded NR=8/non-resident/118-spill path
seen in the `ours_v3 @ openvla 17×192×576` row. So the concrete fix is:

> **Carry v3's `scalarize_a_reads` (`.vf`) into `accumulator_resident_wholemodel`, keeping
> wholemodel's inherent `MR_mm=1` / `NR_bmm=8` tail clamps** so the `.vf` micro-kernel survives the
> small-M openvla/rdt2 matmuls without the M-tail fallback. That converts the whole-model kernel's
> K-loop from `vfmacc.vv`(~20 ops/fma) to `vfmacc.vf`(~3 ops/fma) while preserving the NR=32 +
> accumulator-residency it already has — closing the dominant share of the openvla/rdt2 gap and
> putting the kernel on XNNPACK's inner-loop instruction footing. (Lane-width and residency need no
> change; they are already at the expert bar. Matching OpenBLAS additionally at large K would want an
> MR>1 register block for A-reuse, a separate, smaller follow-up.)

Constraints honoured: baseline (`RVV_TRANSFORM_SCHEDULE` / `hand_v0` / `impr_features` core) read
only; decode harness is a throwaway under `scripts/`; spike-toolchain + host only, no K1 board.
