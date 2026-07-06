# Cross-framework baseline comparison on the SpacemiT K1 (RVV)

**What this is.** merlin vs five *independent* external compilers/runtimes — **TVM, ExecuTorch+XNNPACK,
Buddy (buddy-mlir), EXO, llama.cpp/ggml** — each ingesting the *same* models we support and running
them **end-to-end on the same SpacemiT K1 board with RVV**, on its own stack. This is a true baseline,
unlike the earlier "4-way" (baseline/xnnpack/openblas/ours) which only swapped the GEMM microkernel
*inside* merlin's own dispatch runtime.

**Board.** SpacemiT K1-x (`root@10.44.97.186`), riscv64 glibc Linux, **VLEN=256** (vlenb=32), 8 cores,
**~3.8 GB RAM**. Cross-compiled on host with the SpacemiT clang (`-march=rv64gcv -mabi=lp64d`);
deployed + timed over SSH. Cycle counts are K1 `rdtime` **estimates** (24 MHz timer × CPU/timebase),
**not** cycle-accurate — spike/FireSim remain the cycle authorities.

**Honesty contract** (`merlin.baselines`). `not_run_is_not_pass`: a model that doesn't build/run is an
explicit `not_built`/`not_run` cell with a reason, never omitted or faked. RVV is *pushed* everywhere;
wherever an op/region falls back to scalar it is **labeled** (objdump-based `rvv_audit`), never averaged
away. int8-on-RVV was prioritized (gated vs `golden_w8a8`); all 11 model2MLIR models were attempted.

> ⚠️ **RVV audit tooling.** The SpacemiT/GNU `riscv64-unknown-linux-gnu-objdump` silently mis-decodes
> rv64gcv in bulk `-d` mode (emits bare `.insn` words) → a false ~0% RVV. Confirmed independently by the
> EXO and ExecuTorch arms. The shared resolver now prefers **`llvm-objdump`**.

## Status matrix (whole-model, on the K1)

Full machine-generated matrix (all 5 frameworks × 11 models × {fp32,int8}) + CSV:
`artifacts/compare/v1/…/matrix.{md,csv}` (regenerate via `merlin.baselines.aggregate`). Summary:

| Arm | Best on-K1 whole-model result | int8-on-RVV | The wall it hit |
|---|---|---|---|
| **EXO** | tiny_llama fp32 **PASS** — cos=1.0, rel=8.7e-6 (~784 G cyc) | real `vwmacc.vx` i16→i32; int8 ran **cos=0.9981** vs `golden_w8a8` → near-miss at the 0.999 tier | glue is Llama-family only → 9 models `not_built` |
| **ExecuTorch** | tiny_llama fp32 **PASS** — cos=0.99999999, 217 ms | int8 `not_built` (PT2E W8A8 fails on HF Llama index tensors) | full `.pte` 4.1 GB > 3.8 GB board (ran layer-reduced); XNNPACK delegated 10/93 nodes |
| **Buddy** | compiles **8/11 int8** to genuine integer RVV (`vwmacc.vv`) | ✅ builds; int8 RVV > fp32 RVV | **OOM on every model** — whole-model lowering has *no arena planning* (every intermediate live) |
| **ggml** | tiny_llama int8 **ran** — Q8_0 20.3/5.8 tok/s, Q4_K_M 33/10 tok/s | ✅ native low-bit | correctness **uncomparable** (runs the real checkpoint; our capture is a 2-layer random-init graph) |
| **TVM** | **small_llama int8 compile-correct** (cos=0.99999999, 15.7% RVV) — on-board timing pending | 5 models build via **ONNX→Relax** (small_llama/tiny_llama/openvla/rdt int8) | on-board TVM riscv64 runtime not cross-built yet (`not_run`); tiny_llama cos=0.80 = a TVM v0.19.0 ONNX attn-op bug (ORT matches torch at 1.0) |

Two clean whole-model passes *ran on-board* on our captured graph (EXO, ExecuTorch — both tiny_llama
fp32, reduced/small configs). TVM now **builds numerically-correct RVV** for small_llama int8 (cos≈1.0)
but its on-board runtime isn't cross-built, so it's `not_run` rather than a fabricated pass. Everything
else is an explicit, reasoned gap.

**TVM's ONNX pivot.** Its torch-exported-program frontend couldn't ingest our HF/VLA graphs, so it was
switched to `relax.frontend.onnx.from_onnx` (opset-17 ONNX export). That unblocked real builds: 5 models
compile with RVV; small_llama int8 is numerically correct end-to-end (ONNX→Relax→rv64gcv→RVV-audit).
rdt int8 cos=0.9992 and openvla int8 cos=0.9916 are near-passes; tiny_llama's cos=0.80 is a localized TVM
attention-op mis-lowering (positionally scrambled with identical mean/std; ORT is correct on the same ONNX).

## Kernel-level / active-path RVV coverage

The matrix cell shows *whole-binary* RVV%, which is diluted by scalar plumbing (libc, per-op dispatch,
CRT) and undersells arms whose *compute kernels* are well vectorized. The meaningful **active-kernel**
coverage, per arm (source in parentheses):

| Arm | Whole-binary RVV | Kernel / active-path RVV | Notes |
|---|---|---|---|
| **Buddy** | 16.6–34.5% *(is the compute object — links merlin's C runtime)* | openvla int8 **34.5%**, tiny_llama int8 **24.4%** vs fp32 **16.6%**, molmoact 23.7%, xr0 22.8%, rdt2 22.0%, rdt 19.4%, groot 18.5% | int8 vectorizes **better** than fp32; real `vwmacc.vv` (136 in tiny_llama int8) |
| **ggml** | 7.5% (`.so`) / 9.3% (all kernels) | active-quant **Q8_0 18.4%**, **Q4_K 23.3%**; **ternary TQ2_0 51%, TQ1_0 39%** | ternary is ggml's most-vectorized path (BitNet home turf); 153 scalar kernels labeled |
| **EXO** | 0.9% (whole ELF, libc-dominated) | GEMM kernel **~17%** (`vfmacc.vf` fp32 / `vwmacc.vx` int8) | whole-model = EXO GEMM kernel + hand C glue; norm/rope/softmax/swiglu are labeled scalar glue |
| **ExecuTorch** | **11.7%** | delegated subgraph only — **10/93** graph nodes to XNNPACK RVV | 2185 labeled scalar fallbacks ("no XNNPACK RVV ukernel") — the expected most-scalar arm |
| **TVM** | 9–16% (ONNX→Relax, untuned `+v`) | small_llama/tiny_llama int8 **16%**, openvla 10%, rdt 9% | default LLVM `+v` lowering; MetaSchedule tuning needed to lift; on-board timing pending |

## The headline finding

Only **EXO** and **ExecuTorch** complete a whole-model run on our graph, both on a *reduced* tiny_llama.
The dominant limiter for "all models" is the **K1's 3.8 GB RAM + whole-model working set**, not RVV
codegen:

- **Buddy** compiles genuine int8 RVV for 8/11 models but **OOM-kills every one** — its whole-model
  `linalg-to-loops` has **no buffer reuse / arena planning**, so every intermediate tensor is a
  simultaneously-live allocation (tiny_llama int8: 10.1 GB total-vm on a 3.8 GB board). `built=True` +
  RVV verified, `not_run` with an OOM reason — never a fabricated cos.
- **ExecuTorch**'s full fp32 `.pte` is 4.1 GB — over the board — so it ran a layer-reduced config.

This directly **validates merlin's lean, arena-planned runtime direction**: the differentiator on a
memory-constrained edge board is *whole-model memory planning*, not just a fast kernel. A second robust
result: **int8 vectorizes better than fp32** wherever it builds (Buddy tiny_llama 24.4% vs 16.6%), and
int8 weights (~4× smaller) are the only realistic path to fitting these models on the board.

## Correctness comparability (important caveat)

Our capture bundles are **truncated / random-init** graphs (tiny_llama = 2 layers random init;
small_llama = a toy vocab-256 arch), **not** the real checkpoints. Consequences:

- Arms that ingest **our** `model.mlir` / loader (Buddy, EXO, TVM, ExecuTorch) run the *same* captured
  graph → cos-vs-golden **is valid** (EXO fp32 cos=1.0; ExecuTorch fp32 cos=0.99999999).
- **ggml** runs the *real full* checkpoint with its own tokenizer → cos vs our golden is **uncomparable**
  (recorded `cos=None`, throughput-only — never a claimed pass).
- EXO int8 lands **cos=0.9981** vs `golden_w8a8` (matches the numpy W8A8 reference); the exact
  activation-quant scheme in `golden_w8a8` couldn't be reproduced, so it's an honest near-miss at the
  0.999/1e-2 int8 tier, not a fabricated pass.

## Best-effort int8-RVV tuning pass

A dedicated pass pushed each arm toward its *best* int8-RVV (autotuners run, runtimes cross-built,
memory-planning added). Honest before→after — we made real gains but did **not** reach every
framework's ceiling; the remaining blockers are now precisely characterized:

| Arm | Before | After tuning | Reached best? |
|---|---|---|---|
| **ExecuTorch** | int8 `not_built` | int8 **PASS on-board — 10.9 ms**, cos=0.999; MLP fused into **one XNNPACK qs8 delegate** (int8 GEMM 33–47% RVV) | **subgraph yes** — whole-model int8 blocked: PT2E observer corrupts an int-index dtype at calibration on HF Llama (no quantizer toggle avoids it) |
| **EXO** | int8 44.5 s, GEMM RVV 17% | autotuned k-unroll (ku=4) → GEMM −7%, RVV 18–19%; residual/elementwise now RVV (`vfadd`/`vfmul`); **42.5 s** | **near its approach's limit** — GEMM bottleneck is the per-MAC *scalar A-load* (needs output-tile blocking EXO couldn't stage); quant/transpose pre-pass co-dominant; glue ops are <0.1% of wall so vectorizing them is latency-neutral |
| **Buddy** | int8 OOM (10.1 GB) | **OOM fixed** (buffer-deallocation-pipeline: 0→290 deallocs) | **no on-board run** — OOM solved, but int8 now SIGSEGVs in the integer datapath (weights-region fault; a deep memref/ABI bug, *not* dealloc-related; fp32 runs but is too slow) |
| **TVM** | 5 int8 build, `not_run` | riscv64 **runtime + `tvm_rpc` cross-built** | **no tuned/on-board result** — MetaSchedule + on-board runs blocked by *infra*: board→host firewall (kills tracker tuning) + unstable ssh-launched `tvm_rpc` (kills direct RPC). Needs a persistent board-side RPC service + open board→host path |
| **ggml** | Q8_0 20.3/5.8 tok/s, 18% RVV | (unchanged — already near best) | **closest to best** — hand-optimized kernels are ggml's real capability |

**Verdict:** these are best-*effort*, not proven-best. Only **ggml** is at its real ceiling; **ExecuTorch**
has a genuine tuned int8 on-board number (subgraph); **EXO** is near its kernel-approach limit with an
honest bottleneck; **Buddy** and **TVM** build real int8 RVV but are blocked from a full on-board int8 run
(codegen SIGSEGV / RPC-infra respectively). No fabricated numbers anywhere — blocked cells stay `not_run`
with a specific reason.

## Where things live

- Runners: `merlin/python/merlin/baselines/{buddy,exo,ggml,executorch,tvm}.py` (+ shared
  `bundle/k1_exec/rvv_audit/profile/contract/aggregate`). Submodules: `third_party/baselines/`.
- Raw results: `artifacts/measurements/k1_spacemit/<model>/cross_framework_*/baseline_result.json`.
- Matrix: `artifacts/compare/v1/…/matrix.{md,csv}`. Framework builds: `build/baselines/<fw>/`.

## Open follow-ups

- **TVM on-board + MetaSchedule**: runtime + `tvm_rpc` are cross-built; needs a *persistent* board-side
  RPC service (systemd, not ssh-launched) + an open board→host path for the tuning tracker. Then lift
  RVV above the untuned ~16% and fix the tiny_llama attention-op divergence (cos=0.80).
- **Buddy int8 on-board**: OOM is fixed; the remaining blocker is the int8-datapath SIGSEGV — debug the
  memref/weight-arg stride mismatch between merlin's arg table and buddy's `i8×i8→i32` kernel indexing.
- **ExecuTorch whole-model int8**: needs a PT2E path (or a different quantizer) that keeps HF Llama's
  index/mask tensors out of observation — currently only the linear-heavy subgraph quantizes.
- **EXO GEMM**: output-tile blocking to reuse one scalar A-load across tiles (the real RVV ceiling), and
  an RVV int8 quant/transpose pre-pass (co-dominant with the GEMM).
