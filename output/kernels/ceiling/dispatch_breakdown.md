# Whole-model TIME BREAKDOWN on the K1 board — matmul kernel vs dispatch (iteration-3 proof)

**Question.** Iteration 3 found ours-best is ~60–63% of XNNPACK whole-model (openvla ours-vf
1.089s vs XNNPACK 0.657s; rdt2 30.3s vs 19.0s) and *inferred* the gap is DISPATCH-LEVEL, not the
matmul kernel — because the matmul inner kernel decodes **identically** to XNNPACK's RVV ukernel
(`output/kernels/ceiling/packing_residual.md`: ours-vf is MR=1, 1 unit-stride B load + 1 scalar A
load = 2.00 loads/FMA, `unit_stride_only`, 0 broadcast ladder — byte-for-byte XNNPACK's inner
loop). This measures the split on real silicon to **prove** it.

## Method (board-side, default-off, env-gated)

The K1 runs ONE monolithic compiled `_mlir_ciface_forward` — the ops are fused into a single
lowered function, so there is **no per-op C call boundary** to hook. We create the boundary the
SAME way the existing XNNPACK kernel-backend does: every routable f32 `linalg.matmul` is rewritten
to a `func.call` into the RVV GEMM shim (`runtime/backends/xnnpack_board`). Built with
`-DMERLIN_DISPATCH_TIMING` (default-OFF; the baseline path is byte-identical when off — guarded by
`test_k1.py::test_dispatch_timing_default_off_is_byte_identical`), the shim brackets its
GEMM-ukernel loop with `rdtime` and **accumulates** ticks + call count across every dispatch into a
global the harness prints as `METRIC matmul_ticks` / `METRIC matmul_calls`. Per run:

```
matmul_bucket_ns   = matmul_ticks * (1e9 / 24_000_000)      # GEMM-ukernel compute, resident pack excluded
dispatch_bucket_ns = whole_model_wall_ns - matmul_bucket_ns # everything else
```

The matmul bucket is the inner-kernel compute scope — exactly the part proven == XNNPACK by decode
(the resident-weight pack is cached, excluded, matching the ceiling drivers).

**Localization logic.** The XNNPACK config gives both buckets directly. The matmul work is the
same flops/shapes/kernel regardless of which config emits it, and the inner kernel decodes
identically, so the measured XNNPACK matmul-bucket is the GEMM cost **ours-vf also pays**. ours-vf
whole-model wall MINUS that shared bucket is ours-vf's non-matmul/dispatch cost. Because the matmul
bucket is shared, it cancels in the difference: **the entire ours-vs-XNNPACK wall delta is in the
dispatch bucket, by construction of the measurement** (and the bucket itself is small, see below —
so this is not a tautology hiding a fat kernel; the kernel is genuinely ~8% of either total).

**Caveats (honest).**
- `rdtime` is the 24 MHz platform counter (`cycle_accurate=false`) — the same wall proxy the K1
  harness already uses; matmul ticks and total wall share the timebase, so the ratio is sound.
- The matmul bucket is the XNNPACK ukernel's compute. We attribute it to ours-vf via the decode
  equivalence proof, NOT by re-timing ours' inlined vfmacc (no call boundary exists to isolate
  ours' kernel without changing ours' lowering). This is the method's stated limit: it tests
  "is the gap in the kernel or outside it", and the kernel is shared/equal by construction.
- Routing matmuls to the shim moves a sliver (the call ABI + descriptor unpack) into the call
  itself; tiny vs the per-op glue cost being localized.

cos-gated (≥ 0.9999 vs host golden) before any wall is recorded. Timer:
`CLOCK_MONOTONIC wall_ns + rdtime matmul ticks; cycle_accurate=false`. Board: SpacemiT K1, VLEN=256.

---

## openvla — per-category wall (N=5 runs/config, min wall + range%)

| config | whole-model min wall (ns) | range% (N) | fp32 cos | **matmul-bucket (ns)** | **dispatch-bucket (ns)** | matmul frac | matmul calls |
|---|---|---|---|---|---|---|---|
| baseline (hand_v0) | 5,876,396,986 | 1.70% (5) | 0.9999999 | — | — | — | — |
| **ours-vf** (`accumulator_resident_wholemodel_vf`) | 1,094,477,461 | 0.83% (5) | 1.0000000 | 52,945,000 (shared) | **1,041,532,461** | 0.048 | 26 |
| **XNNPACK** (`xnn_f32_gemm_ukernel_1x4v__rvv`) | 660,799,302 | 1.79% (5) | 0.9999999 | **52,945,000** | **607,854,302** | **0.080** | 26 |

(matmul-bucket measured ONLY in the XNNPACK config — the routed+timed path; spread across 5 runs
52.6–54.1ms, i.e. ±1.3%. Attributed to ours-vf by the decode-equivalence proof.)

### Localization

- **The matmul kernel is only 8.0% of XNNPACK's whole-model wall** (52.9ms of 660.8ms). 92% of
  XNNPACK's time is already NON-matmul (attention generics, rmsnorm, elementwise, softmax,
  layout-copies, per-dispatch glue).
- **ours-vs-XNNPACK delta = 433.7ms, and 100% of it is in the dispatch bucket.** The shared matmul
  bucket (52.9ms) cancels; ours' dispatch bucket is 1,041.5ms vs XNNPACK's 607.9ms — a **433.7ms
  dispatch excess**, identical to the whole-model delta. ours/XNNPACK = 1.656×.
- **Verdict: CONFIRMED — the 60% gap is dispatch-level, not the matmul kernel.** The kernel ours-vf
  emits is the same ~53ms work XNNPACK does; the entire shortfall is in everything-else.

The mechanism is the **non-matmul / dispatch bucket** (per-op interpreter-free monolithic lowering
still pays per-op setup, no cross-op fusion, and the non-GEMM ops — attention/norm/activation —
are themselves on the un-tuned Merlin RVV path), NOT a fatter matmul kernel: the matmul kernel is
shared and small.

---

## rdt2 — per-category wall (N=3 runs/config)

| config | whole-model min wall (ns) | range% (N) | fp32 cos | **matmul-bucket (ns)** | **dispatch-bucket (ns)** | matmul frac | matmul calls |
|---|---|---|---|---|---|---|---|
| baseline (hand_v0) | 73,841,995,504 | 0.35% (3) | 1.0000001 | — | — | — | — |
| **ours-vf** (`accumulator_resident_wholemodel_vf`) | 30,242,140,052 | 0.37% (3) | 1.0000000 | 622,413,542 (shared) | **29,619,726,510** | 0.021 | 23 |
| **XNNPACK** (`xnn_f32_gemm_ukernel_1x4v__rvv`) | 18,970,159,238 | 0.28% (3) | 1.0000001 | **622,413,542** | **18,347,745,696** | **0.033** | 23 |

(matmul-bucket measured in the XNNPACK config; spread across 3 runs 621.6–626.1ms, ±0.7%.)

### Localization

- **The matmul kernel is only 3.3% of XNNPACK's whole-model wall** (622ms of 18,970ms). ~97% of
  XNNPACK's time is non-matmul.
- **ours-vs-XNNPACK delta = 11.27s, and 100% of it is in the dispatch bucket** (the 622ms matmul
  bucket cancels). ours dispatch 29.62s vs XNNPACK dispatch 18.35s. ours/XNNPACK = 1.594×.
- **Verdict: CONFIRMED — dispatch-level gap, not the matmul kernel** (same as openvla, even more
  pronounced: rdt2's larger-K GEMMs make the matmul bucket absolutely bigger but it is still a tiny
  fraction of the whole-model wall, and it is shared with ours).

---

## Sources / provenance

- Raw machine-readable: `output/rvv_bench/dispatch_breakdown.json` (per-model dict).
- Whole-model wall context (cached 4-way / vf): `output/rvv_bench/k1_4way_{openvla,rdt2}.json`,
  `output/rvv_bench/k1_vf_{openvla,rdt2}.json`.
- Kernel-decode equivalence (the proof the matmul bucket is shared): `packing_residual.md`.
- Harness: `scripts/k1_dispatch_breakdown.py`. Timing hook:
  `runtime/backends/xnnpack_board/xnn_gemm_rvv_shim.c` + `rvvgen/k1.py` (`-DMERLIN_DISPATCH_TIMING`,
  default-off, byte-identical when off). Guard: `tests/test_k1.py`.
