# The packing / memory-traffic residual — what the experts do for DATA MOVEMENT that ours does not (iteration 3, Part A)

**Method.** Static decode of the emitted RVV asm, EXTENDED with a new **memory-traffic facet**
(`merlin/python/merlin/kernels/decode/memory.py::analyze_memory`) that the CCA `vector`/`compute`
facets were blind to. The CCA captures NR/LMUL/accumulator-residency/vfmacc-form (the *compute*
mechanics); it did NOT capture how the OPERANDS reach the FMA. The new facet, scoped to
`cca._fma_loop` (the K-reduction loop), classifies every load STRUCTURALLY by mnemonic — the RVV
ISA encodes the addressing mode in the opcode, so this is a mnemonic-prefix classification, no regex
over operand text (same robustness contract as `decode/rvv`):

- unit-stride vector load (`vle`/`vlNre`) — the expert pre-packed-panel shape
- strided vector load (`vlse`) — a 2-D model-layout gather
- indexed gather (`vlux`/`vlox`)
- scalar FP load (`flw`/`fld`) — the `.vf` A scalar straight into an FP reg
- A-broadcast ladder (`vslideup`/`vmv`/`vrgather`) — rebuilding the A vector each K step (the `.vv` cost)

and reports **loads / useful-FMA**, **A-broadcast / FMA**, and **unit_stride_only**. Harness:
`scripts/decode_packing_residual.py` (throwaway). Raw JSON: `packing_residual_decode.json`.
HOST + spike-toolchain only, no K1 board.

**Shapes.** The cube baseline plus the REAL openvla/rdt2 matmul dims, and the actual M-spread of the
two models (extracted from `output/{openvla,rdt2}_fp32_consistent/model.mlir`).

---

## Memory-traffic decode table (K-loop, one trip)

| kernel | shape | MR | vfmacc.vf | vfmacc.vv | unit loads | strided | scalar (A) | broadcast ladder | **loads / FMA** | unit-stride only |
|---|---|---|---|---|---|---|---|---|---|---|
| **XNNPACK** 1x4v | ukernel | 1 | 1 | 0 | 1 | 0 | 1 | 0 | **2.00** | yes |
| **OpenBLAS** 8x8 | ukernel | 16 | 60 (8×8 tile) | 0 | (pre-packed) | 0 | — | 0 | **~1.06 amortized** (MR=16 reuse) | yes |
| ours `wholemodel` (iter-1, `.vv`) | every openvla/rdt2 shape | 1 | 0 | 1 | 2 | 0 | 0 | **8** | 2.00 | yes |
| **ours `wholemodel_vf` (iter-2, `.vf`)** | **every openvla/rdt2 shape** | **1** | **1** | **0** | **1** | **0** | **1** | **0** | **2.00** | **yes** |
| ours `wholemodel_vf_mr4` (iter-3) | large-M, M%4==0 (cube 64/128, M=20) | 4 | 4 | 0 | 1 | 0 | 4 | 0 | **1.25** | yes |

(K-loop body of `wholemodel_vf` at rdt2 28×1024×1024, decoded verbatim: `vle32.v v12,(a3)` +
`flw fa5,0(s1)` + `c.addi s1,4` + `vfmacc.vf v8,fa5,v12` + `c.add a3,a1` + `bne` = **6 insns, 1
unit-stride B load + 1 scalar A load, identical to XNNPACK's inner loop**.)

---

## Finding 1 — vs XNNPACK, there is NO per-FMA memory residual left. Iteration 2 already closed it.

At **every** openvla/rdt2 shape the iteration-2 `accumulator_resident_wholemodel_vf` kernel decodes
**identically to XNNPACK's data movement**: MR=1, **1 unit-stride B-row load + 1 scalar A load =
2.00 loads / useful-FMA**, `unit_stride_only = True`, **0 broadcast-ladder ops**. The `.vv` A-broadcast
ladder (8 `vslideup`/`vmv` ops in the iteration-1 `wholemodel` kernel) is **gone**. So the brief's
hypothesised "strided model-layout stream" residual **does not exist** for the `.vf` kernel — the A
scalar is read by `flw` (not a strided/lane gather), the B row by a single contiguous `vle32.v`, and
neither operand is strided (`vec_strided_loads = 0` at all five shapes). The iteration-2 `.vf` fix
that collapsed the broadcast ladder ALSO put the operand loads on XNNPACK's unit-stride footing.

## Finding 2 — the one remaining data-movement lever is OpenBLAS's MR>1 A-reuse register block.

Both ours-`vf` and XNNPACK are **MR=1**: each B row is loaded once per *output row* (no A-reuse
across rows). **OpenBLAS holds MR=16 output rows in 16 accumulator vreg-groups**, so ONE B-row load
is shared across 16 FMAs (and each A element across NR columns) → ~1.06 amortized loads/FMA vs our
2.00. That register block is the only lever left. The iteration-3 feature reproduces it on the `.vf`
path: **MR=4 register block → loads/FMA = 1.25** (1 B-load + 4 A-scalars over 4 FMAs), MEASURED on
large-M cube/M=20, 0 spills, accumulator-resident — the OpenBLAS A-reuse shape, bit-exact.

## Finding 3 (the honest blocker) — MR>1 cannot help openvla/rdt2: they are ALL small-M.

The A-reuse register block needs **M ≥ MR with a clean M-tile**. The openvla/rdt2 matmuls have **no
large-M matmul** — their leading dim is the token/batch count, which is structurally small for VLAs:

| model | `linalg.matmul` M dims (LHS rows) | count |
|---|---|---|
| openvla | M ∈ {20 (×11), 17 (×8), 16 (×3)} | small-M prefill |
| rdt2 | M ∈ {28 (×17), 1 (×5)} | small-M / M=1 decode |

On those shapes MR=4 (MEASURED, decode):
- **M=17, M=1** (not divisible by 4): the M-tail trips the LLVM-23 masked-`transfer_write`
  multi-op `vector.mask` PipelineError → degrades to **NR=8 / non-resident / 118 spills / vfmacc.vv**
  (the documented broken path) — a hard regression.
- **M=16, M=28** (divisible by 4): the v3 `.vf` path silently **scalar-falls-back** (0 vfmacc
  emitted) — also a whole-model regression.
- **M=20** (=5×4): the only small-M shape where MR=4 lowers cleanly (loads/FMA 1.25, bit-exact).

So **MR>1 is not whole-model-safe for the VLA decode/prefill matmuls** and would regress them. The
A-reuse the VLAs leave on the table (loads/FMA 2.0 instead of OpenBLAS's ~1.06) is a **STRUCTURAL
property of their small token dim, not a matmul-kernel defect**. Closing it cannot be done by the
matmul kernel — it needs a **dispatch-level layout/batching pass** that groups multiple small-M
matmuls (e.g. the 11 separate M=20 projections, or per-head attention) into one large-M GEMM so an
MR>1 register block has rows to reuse A across. That is out of scope for a matmul-kernel feature and
is the precise, bounded blocker.

## Finding 4 (honest) — on the spike cycle proxy the MR=4 A-reuse does NOT win.

`measure_ours` (bare-metal, inner-compute, bit-exact-verified) on spike:

| shape | `vf` (MR=1) cycles | `vf_mr4` (MR=4) cycles |
|---|---|---|
| 64³ | 98,791 | 370,466 |
| 128³ | 787,325 | 1,678,317 |
| 96×48×160 (non-cube) | 277,544 | 499,837 |

All **bit-exact (VERIFY PASS)**, but MR=4 is *slower* on the proxy. spike's functional model is
IPC=1 with no memory hierarchy, so a load costs the same as an FMA and **reducing loads/FMA buys
nothing**; the MR=4 setup (4 accumulator groups, wider tile) just adds instructions. The A-reuse
register block is a **real-hardware memory-bandwidth** win (fewer DRAM/cache touches per FMA), which
only shows on the K1 board / FireSim — NOT on the spike cycle proxy. Reporting this honestly: the
structural loads/FMA metric improves (2.0 → 1.25, decode-confirmed) but the spike-measurable cycle
count does not, and the board re-measure (parent-chained) is what would show whether the reuse pays
off on silicon.

---

## Conclusion (Part A)

1. **vs XNNPACK: closed.** The iteration-2 `.vf` kernel already ties XNNPACK's inner-loop data
   movement at every openvla/rdt2 shape (2.0 loads/FMA, unit-stride, 0 ladder, MR=1). No per-FMA
   memory residual remains vs XNNPACK.
2. **vs OpenBLAS: the residual is MR>1 A-reuse**, reproduced as `accumulator_resident_wholemodel_vf_mr4`
   (loads/FMA 2.0 → 1.25, decode-confirmed, bit-exact on large-M).
3. **The openvla/rdt2 gap is NOT matmul-kernel-closable**: those models have only small-M matmuls
   (token dim 1–28), where MR>1 has no clean tile and regresses. The remaining A-reuse needs a
   dispatch-level large-M batching/layout pass — the precise, honest blocker.
