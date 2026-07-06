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
| **TVM** | control MLP → RVV ELF (harness proven) | — | re-pin **solved** (v0.19.0 + MetaSchedule); torch-Relax frontend can't ingest our HF/VLA graphs → **ONNX path in progress** |

Two clean whole-model passes on our captured graph (EXO, ExecuTorch — both tiny_llama fp32, on
reduced/small configs). Everything else is an explicit, reasoned gap.

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
| **TVM** | ~0% (untuned control) | — (needs on-device MetaSchedule) | no model built yet (frontend gap); ONNX path in progress |

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

## Where things live

- Runners: `merlin/python/merlin/baselines/{buddy,exo,ggml,executorch,tvm}.py` (+ shared
  `bundle/k1_exec/rvv_audit/profile/contract/aggregate`). Submodules: `third_party/baselines/`.
- Raw results: `artifacts/measurements/k1_spacemit/<model>/cross_framework_*/baseline_result.json`.
- Matrix: `artifacts/compare/v1/…/matrix.{md,csv}`. Framework builds: `build/baselines/<fw>/`.

## Open follow-ups

- **TVM via ONNX** (in progress): torch→ONNX (QDQ int8) → `relax.frontend.onnx` → MetaSchedule autotune
  for rv64gcv → on-board — the realistic path past the torch-frontend op-coverage gap.
- Buddy on-board completion would require arena planning (or int8 + streamed weights) to fit 3.8 GB.
- ExecuTorch int8 needs a PT2E path that tolerates HF Llama's index/mask tensors.
