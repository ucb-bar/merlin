# K1 e2e whole-model before/after RVV optimization (real SpacemiT K1 silicon)

PART 1 of the kernel-policy-mining K1 capstone: does the mined **tiled vfmacc**
(`fused_vfmacc_tiled`, the bounded-code/scoped-vectorize/whole-model-safe vfmacc) actually
hold — and help — on a WHOLE model running on the K1 board, not just single-matmul workloads?

- Board: SpacemiT K1, Bianbu Linux 6.6.63 riscv64, VLEN=256 (`vlenb`*8 read at run time), 8 cores, 3.4 GB RAM.
- Timing: K1 traps userspace `rdcycle`; whole-model `wall_ns` from `clock_gettime(CLOCK_MONOTONIC)` (ground truth), N=3 runs, min reported. `time_ticks` = delegated `rdtime` (24 MHz). cycle_accurate=false.
- Correctness: fp32 cosine of the board output vs `golden.npy` (`zephyr_model._gate`).
- BASELINE = `hand_v0` (frozen `RVV_TRANSFORM_SCHEDULE`, untouched). OPTIMIZED = an `impr` fork enabling `fused_vfmacc_tiled` (tile [MR=4,NR=16,KC=16], scoped vectorize, contract→outerproduct→vfmacc).
- Both binaries are cross-compiled with the SAME lowering the runner uses (`build_k1_binary`); `compiler_features` is threaded through to codegen, so the OPTIMIZED binary really emits the tiled vfmacc.

## Headline result — bitvla (the model where the feature applies)

`output/bitvla_fp32_consistent` (9.3 MB weights, golden 1×32×1024, no NaN). Both packages **vectorize the whole model** and stay correct.

| package | lowering path | `fmuladd` in `.ll` | min wall (N=3) | fp32 cos vs golden |
|---|---|---|---|---|
| baseline (hand_v0) | vectorized | 0 | 2,524,538,754 ns (2.52 s) | 0.999995 |
| optimized (fused_vfmacc_tiled) | vectorized | 1217 | 269,986,808 ns (0.27 s) | 0.999995 |

**e2e speedup = baseline_wall / optimized_wall = 9.35x.** Both correct (cos 0.999995 — identical to 6 d.p.).

The baseline emits `0` fmuladd (the known separate `vfmul.vv`+`vfadd.vv` gap); the optimized fork
emits `1217` `llvm.intr.fmuladd` → `vfmacc` across the whole model. So the tiled-vfmacc lever — mined
from the `scalar_broadcast_fma` motif (XNNPACK/OpenBLAS) — **does apply to a whole model on real
K1 silicon and gives a ~9.4x e2e wall speedup with no loss of correctness.** The baseline's
degenerate `<1 x float>` mul+add path is far slower on the K1 VPU than the fused fixed-width vfmacc.

## The whole-model-SAFETY caveat (the most important finding)

`fused_vfmacc_tiled` is NOT universally whole-model-safe. On **tiny_llama** and **small_llama**
(`*_fp8_consistent`, which are all-f32 in the graph — the fp32 vectorized matmul path) the
optimized fork's vectorized lowering **raises `PipelineError` and silently falls back to SCALAR**:

```
error: 'vector.mask' op expects only one operation to mask
  vector.transfer_write (vector<1x4x16xf32>, tensor<1x4x8xf32>) ...
```

Root cause: the llama **attention `batch_matmul` has N=8** (seq length), and the recipe tiles
`batch_matmul [1,4,16,16]` with NR=16 > 8 → a masked `vector<1x4x16> → tensor<1x4x8>`
`transfer_write` that this LLVM-23 vector lowering rejects. The recipe was only verified on
single 32/64/128-cube matmuls where NR=16 divides N cleanly; a real transformer with small
attention dims trips it. `build_k1_binary` catches the `PipelineError` and falls back to scalar,
so a naive run would have looked like it "succeeded" while actually NOT running the tiled vfmacc.
The e2e harness DETECTS the fallback (`lowering_path`) and reports it honestly.

So the claim "the fixed tiled recipe is whole-model-safe" holds for some models (bitvla) but
**breaks on any model with a batch_matmul whose N < NR (=16)** — which includes both llamas'
attention. Concrete compiler work-item: the tiled-vfmacc schedule needs an N-tail path (mask the
write to the real N, or pick NR ≤ N for small-N batch_matmuls) before it is truly whole-model-safe.

## Honest not_run (llama models — see `k1_e2e_llama_notrun.json`)

- **tiny_llama fp8→f32**: optimized falls back to scalar (above). Separately, the whole-model
  embedded-fp32 binary (~420 MB; RW LOAD segment ~398 MB) **SIGSEGVs on the board even on the
  frozen baseline** — a store fault just past BSS; the board `CommitLimit` is ~1.9 GB and the
  embedded-fp32 whole-model image exceeds the board's memory policy. Not a fork issue. not_run.
- **small_llama fp8→f32**: small enough to run (577 KB weights), but the frozen baseline produces
  **all-NaN on BOTH the K1 AND spike** — this particular fp8→f32 capture is numerically broken on
  the RVV path, so it is not a usable correctness target. not_run.

bitvla is therefore the correct, self-contained fp32 whole-model on which the before/after is
both valid and favorable.
