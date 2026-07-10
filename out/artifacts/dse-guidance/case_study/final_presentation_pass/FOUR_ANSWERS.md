# The four questions — answered with K1-measured evidence

All numbers are real SpacemiT K1 (VLEN=256), cos-gated ≥0.9999, fair (experts at their BEST kernels:
XNNPACK `7x4v` MR=7, OpenBLAS `16x8_zvl256b` MR=16). The matmul bucket is measured on BOTH arms via an
`rdtime` bracket (the `ours_board` shim is a *measurement instrument* only — our product is the MLIR
compiler, never a hand kernel). Sources: `output/rvv_bench/k1_kernel_speedup_{bitvla,openvla,rdt2}.json`,
`dispatch_breakdown_measured.json`, `k1_e2e_fair_*.json`.

## Q1 — Why do we beat bitVLA whole-model if our kernel is slower?
**It is whole-model integration (no per-matmul call boundary), NOT a better kernel.** The expert arms
must reach their ukernel through a `func.call` shim at every routable matmul (15 sites on bitVLA); our
compiler *inlines* the kernel into the whole-model binary. Measured, same MR=4 kernel:
- ours **inlined** (`accumulator_resident_microkernel_v3`): **147 ms** (the fair bitVLA winner).
- ours **routed** through the shim (MR=4): **177 ms** → the `func.call` + descriptor-unpack + per-call
  A-pack **routing tax ≈ 30 ms**.
- XNNPACK 7x4v **routed**: **167 ms**; ours routed at MR=7 ties it (169 ms).

So our kernel is *slower* (matmul bucket 33 ms vs XNNPACK 10 ms, 3.18×), but bitVLA's M=32 is cleanly
register-tileable so our inlined MR=4 runs well enough that **avoiding the routing tax a library cannot
avoid** nets the 1.13×/1.22× win. "A library is a call you can't fuse across." This does NOT hold on
openvla/rdt2 (Q2/Q4).

(Correction: earlier docs said the bitVLA winner was `accumulator_resident_wholemodel_vf` via a
"non-matmul schedule." Wrong on both counts — the winner is `microkernel_v3`, and the v3 schedule
touches only matmul/batch_matmul. The mechanism is inlined-vs-routed, now measured.)

## Q2 — What do the experts do that we don't, and why is our path slow?
The four-way is a **kernel-swap on our runtime** — the experts' *own* runtime is never tested; what we
expose is (a) their GEMM ukernel structure and (b) our non-matmul path.
1. **Kernel: register blocking.** XNNPACK reuses each loaded B-row across MR=7 broadcast-FMA
   accumulators (1+1/MR loads/useful-FMA, MR independent chains to hide vfmacc latency); OpenBLAS MR=16.
   Our baseline is unblocked (MR=1, 2.0 loads/FMA) and our v3 is MR=4.
2. **Non-matmul dominates the loss.** On openvla/rdt2 the matmul is only ~17–21 % of our wall; the gap is
   the dispatch/runtime path: scalar transcendental activations (a vectorized fix EXISTS but is OFF —
   `act_poly.py`, ~3.5× on the op), no cross-op fusion (~4,484 `tensor.empty`→`alloc` materializations,
   DRAM round-trips), scalar softmax, per-op vsetvl/setup. See `RUNTIME_INVESTIGATION.md`.

## Q3 — Why is our GEMM slower + lower-utilization, when we literally disassemble their kernels?
**A register-blocking blind spot in the mining — now fixed.** The mined kernel index already extracted
MR per expert kernel (`f32-gemm-7x4v-rvv.c → MR=7`), but:
- `mine.expert_cca_from_policies` built the expert CCA from a **policy table** that had no MR axis →
  `register_block: null`; and
- `cca_compare`/`compare.py` did not surface a register-block divergence.

So our compiler disassembled XNNPACK yet the #1 GEMM decision (MR) was invisible to it; we hand-chose
MR=4. **Fix (this work):** `expert_cca_from_policies` now reads the experts' MR from the index
(`_expert_register_block` → MR=7), `cca_compare` emits an MR-aware `compute.register_block` divergence
(None == MR 1), and the action catalog routes it to the register-block KNOB. Verified: the divergence
`expert=(7,…) vs ours=None` now appears and routes to an action.

**Utilization (measured matmul buckets vs XNNPACK 7x4v):** our compute throughput is a fraction of
theirs, and the fraction is set by register blocking and M-structure:

| model | M | ours MR=1 | MR=4 | MR=7 | ↳ what's left at MR=7 |
|---|---|---|---|---|---|
| bitVLA | 32 (clean) | 11.2× slower | 3.16× | **2.23×** | codegen / packing residual |
| openvla | 16–20 | 24× | 6.9× | 4.8× | small-M padding waste |
| rdt2 | 1, 28 | 45× | 13.5× | 9.1× | M=1 GEMV — blocking useless |

Register blocking (MR 1→7) is the dominant lever (~5× on the matmul bucket). B-packing is **not** the
residual (tested: MR=7 packed 23.6 ms ≈ unpacked 23.0 ms). The residual is codegen + the structural
small-M cap.

## "First try the kernel speedup" — honest result
- **Register blocking is the lever, and the compiler must learn + emit it (not us).** We fixed the mining
  so the compiler *learns* MR from the experts (Q3). The compiler already emits MR=4 register-blocked
  `vfmacc.vf` (the bitVLA winner).
- **But naive higher MR via the existing grid knobs REGRESSES the compiler's whole-model output**:
  inlined MR=8 → 178 ms, NR=32 → 165 ms, vs MR=4/NR=16 → 147 ms. The transform-schedule codegen does not
  realize the diagnostic's MR=7 headroom — so closing the residual needs a register-block **codegen**
  improvement (a PASS), not a knob bump. This is exactly the compiler-vs-hand-kernel distinction: the
  knob that speeds an idealized C microkernel regresses in whole-model lowering.
- **Small-M / GEMV is structurally capped** (padding waste) — the fix is dispatch-level batching, not the
  kernel (Q4).

## Q4 — What can we (a compiler) exploit that a frozen library cannot?
Discovered by **analyzing the workload** (scalable, not hand-picked), routed per model
(`opportunity_discovery`, Phase 2):
1. **Small-M matmul batching** — group the VLAs' independent same-(N,K) token-dim matmuls into one
   large-M GEMM so MR>1 register blocking finally applies. A per-op GEMM library sees each small-M call
   separately and *cannot* batch across dispatches. Directly attacks the openvla/rdt2 compute cap.
2. **Attention QK·softmax·V + epilogue (matmul+bias+act+residual) fusion** — one tiled pass, no DRAM
   round-trips a per-op library must pay.
3. **Vectorized transcendental activations** — built (`act_poly.py`), OFF by default; the cheapest
   demonstrable win (turn on whole-model, re-measure).
4. **Native low-bit** — bitVLA W1.58 ternary / W8A8 int8 the libraries have *no* kernel for: categorical.

The throughline: we **learn** the experts' decisions (register blocking, residency, widening, …) into the
**compiler**; where the library is frozen (cross-op fusion, cross-dispatch batching, native low-bit) is
where the compiler wins — and the workload analysis is what selects the right lever per model.
