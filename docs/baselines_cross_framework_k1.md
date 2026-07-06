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
| **EXO** | tiny_llama fp32 **PASS** cos=1.0; int8 **1.44 s** (44.5→24.6→1.44, **31×**), cos=0.9981 near-miss | real `vwmacc` i16→i32; **transpose-free `[N,K]` dot GEMM** killed the scatter (461M→17.6M) | fast GEMM is hand-RVV C (EXO can't schedule stride-1+`vredsum`); glue Llama-only → 9 models `not_built` |
| **ExecuTorch** | tiny_llama fp32 **PASS** cos=0.99999999 (217 ms); **int8 PASS — 13.7 ms**, cos=0.99987 (**all 7 decoder layers' linears**) | int8 GEMM 33–47% RVV in 2 XNNPACK qs8 delegates | whole-model int8 **proven impossible** (PT2E transform corrupts int-index even with empty quantizer) |
| **Buddy** | compiles **8/11 int8** to genuine integer RVV (`vwmacc.vv`, up to 34%) | ✅ builds; int8 RVV > fp32 | **0 on-board** — whole-model-scale **Buddy scalar-lowering bug** (dequant/ABI/RVV theories all *disproved*); needs op-bisection + submodule patch |
| **ggml** | tiny_llama int8 **ran** — Q8_0 20.3/5.8 tok/s, Q4_K_M 33/10 tok/s | ✅ native low-bit | correctness **uncomparable** (runs the real checkpoint; our capture is a 2-layer random-init graph) |
| **TVM** | **small_llama int8 PASS on-board — 52.7 ms, cos=1.0**; openvla int8 ran 6.5 s (cos 0.9916, fail) | 5 models build via **ONNX→Relax** (9–16% untuned RVV) | tuning *one handshake from done* (tracker↔server deadlock over tunnel); big `.so` on-board load hangs; tiny_llama cos=0.80 = a TVM ONNX broadcast-`Mul` defect (RMSNorm) |

Four on-board passes now: EXO & ExecuTorch tiny_llama fp32, ExecuTorch tiny_llama int8, and **TVM
small_llama int8** (52.7 ms, cos=1.0). TVM also **builds numerically-correct RVV** for small_llama int8
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

## int8-RVV tuning + follow-up passes (final state)

Three escalating passes pushed each arm toward its *best* int8-RVV: (1) autotuners run + runtimes
cross-built + memory-planning; (2) "double-down" on exo/tvm/buddy (output-tile blocking, reverse-tunnel
RPC, bufferizer diagnosis); (3) "do all of them" (transpose-free GEMM, whole-model int8, tuning
round-trip, bufferizer fork-attempt). Honest before→final — real, large gains, with every remaining
blocker now precisely characterized (and one earlier root-cause corrected):

| Arm | Before | After tuning | Reached best? |
|---|---|---|---|
| **ExecuTorch** | int8 `not_built` | int8 **PASS on-board — 13.7 ms**, cos=0.99987; **all 7 decoder layers' linears** int8 (attn q/k/v/o + SwiGLU) → 2 XNNPACK qs8 RVV delegates | **maximal** — whole-model int8 **proven impossible**: PT2E's `transform_for_annotation` corrupts an int-index dtype at calibration *even with an empty quantizer*, so exclusions can't help. Only embedding/RoPE/mask/softmax/RMSNorm stay fp32 |
| **EXO** | int8 44.5 s, GEMM RVV 17% | output-tile blocking (U=8) → 24.6 s; then **transpose-free `[N,K]` dot GEMM** killed the 461M-tick scatter (→17.6M, −96%) → **whole-model 44.5 s → 1.44 s (31×)**, integer-exact | **at its approach's limit** — no dominant cost center left. Honest: the fast dot GEMM is **hand-RVV C** (EXO's memory classes assume stride-1 + can't schedule the `vredsum` reduction tail); the EXO-authored `vwmacc` kernel is kept + audited for attribution |
| **Buddy** | int8 OOM (10.1 GB) | **OOM fixed** (buffer-dealloc 0→290); pi05 opt-timeout fixed (size-adaptive vectorize) | **no on-board run** — earlier dequant-bufferizer theory **DISPROVED**: the isolated i8→i32→f32 dequant micro-case *runs correctly on the K1*, the ABI is byte-identical to merlin's (which runs int8), and a **scalar-only `rv64gc` build also SIGSEGVs** → it's a **whole-model-scale Buddy scalar-lowering bug**, needing instrumented op-bisection. Correctly *not* patched on speculation |
| **TVM** | 5 int8 build, `not_run` | persistent systemd `tvm_rpc` + `--custom-addr` → **small_llama int8 PASS on-board — 52.7 ms, cos=1.0**; openvla int8 ran 6.5 s (cos 0.9916, fail gate) | **on-board yes (small_llama)** — MetaSchedule now *one handshake from done* (systemd unit + `--custom-addr` real-IP registration + 45 tasks + xgboost all work; the tracker↔server session-alloc over the reverse tunnel **deadlocks** `free 0 pending 1`) → **tuned RVV not measured**; big `.so` scp's but on-board VM load hangs; **tiny_llama cos=0.80 root-caused to a TVM ONNX broadcast-`Mul` defect in RMSNorm** (TVM 0.968 vs ORT/numpy 1.0) |
| **ggml** | Q8_0 20.3/5.8 tok/s, 18% RVV | (unchanged — already near best) | **closest to best** — hand-optimized kernels are ggml's real capability |

**Verdict (final):** best-*effort*, honestly bounded. **TVM** and **ExecuTorch** each land a genuine
on-board int8 **PASS** (small_llama 52.7 ms cos=1.0; tiny_llama all-decoder-linears 13.7 ms cos=0.99987).
**EXO** cut its whole-model int8 wall **31×** (44.5 → **1.44 s**) via output-tile blocking + a transpose-free
`[N,K]` dot GEMM, with no dominant cost center left. **ggml** stays at its hand-tuned ceiling (Q8_0
20.3/5.8 tok/s). **Buddy** builds genuine int8 RVV (8 models, up to 34%) but is the one arm that still
can't run int8 on-board — the blocker is now *precisely bounded* to a whole-model-scale Buddy scalar-lowering
bug (the earlier dequant-bufferizer theory was **disproved**), not our integration. Each remaining gap is
one named step: **TVM** the tracker↔server tuning handshake + big-`.so` on-board load + a TVM broadcast-`Mul`
patch (RMSNorm) for tiny_llama; **Buddy** an instrumented whole-model op-bisection + submodule patch;
**ExecuTorch** would need patching PyTorch's PT2E to go beyond decoder-linears. No fabricated numbers
anywhere — blocked cells stay `not_run` with a specific reason.

## Where things live

- Runners: `merlin/python/merlin/baselines/{buddy,exo,ggml,executorch,tvm}.py` (+ shared
  `bundle/k1_exec/rvv_audit/profile/contract/aggregate`). Submodules: `third_party/baselines/`.
- Raw results: `artifacts/measurements/k1_spacemit/<model>/cross_framework_*/baseline_result.json`.
- Matrix: `artifacts/compare/v1/…/matrix.{md,csv}`. Framework builds: `build/baselines/<fw>/`.

## Open follow-ups

- **TVM MetaSchedule tuning**: everything works except the tracker↔server session-allocation handshake
  over the reverse tunnel, which **deadlocks** (`free 0 pending 1`). Closing it needs debugging TVM's
  matchmaking over the tunnel or a same-subnet host. Also: the big `.so` (rdt/tiny_llama) scp's to the
  board but the on-board VM *load* hangs, and tiny_llama's cos=0.80 needs a TVM patch for the ONNX
  broadcast-`Mul` (RMSNorm) defect — both TVM-internal fixes.
- **Buddy int8 on-board**: OOM fixed; blocker is a **whole-model-scale Buddy scalar-lowering bug** (the
  earlier dequant-bufferizer theory was disproved — isolated pattern + scalar-only build both reproduce/run
  as expected). Needs instrumented whole-model op-bisection to find the culprit op, then a submodule patch.
- **ExecuTorch whole-model int8**: proven impossible without patching PyTorch's PT2E (`transform_for_annotation`
  corrupts int-index dtypes even with an empty quantizer). Current best = all decoder-layer linears int8.
- **EXO**: at its limit for this workload (1.44 s, no dominant cost center). Further gains would need
  RVV rmsnorm/softmax (currently small) or EXO-schedulable stride-1 reduction support (the dot GEMM is hand-RVV).
