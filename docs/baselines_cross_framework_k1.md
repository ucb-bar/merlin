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
| **EXO** | **4 on-board PASSes**: tiny_llama fp32 cos=1.0 + **int8 cos=1.0** (7.3 s), small_llama fp32 cos=1.0 + int8 cos=1.0 | int8 = **weight-only (W8A16)** matching the capture's weight-only golden (per-channel int8 weight dequantized on the fly inside a transpose-free `[N,K]` f32 dot); EXO `vwmacc`/`vfmacc` kernels compiled + audited | fast dot is hand-RVV C (EXO can't schedule stride-1+`vredsum`); glue Llama-only (hf+terse schemas) → VLAs `not_built` (no isolated LLM-backbone golden; BitNet sub-norms) |
| **ExecuTorch** | tiny_llama fp32 **PASS** cos=0.99999999 (217 ms); **int8 PASS — 13.7 ms**, cos=0.99987 (**all 7 decoder layers' linears**) | int8 GEMM 33–47% RVV in 2 XNNPACK qs8 delegates | whole-model int8 **proven impossible** (PT2E transform corrupts int-index even with empty quantizer) |
| **Buddy** | **m2m int8 now RUNS CORRECTLY on-board** — small_llama int8 **cos=0.99993** (bit-matches merlin's numpy W8A8 reference); 8/11 build genuine integer RVV (`vwmacc.vv`, up to 34%) | ✅ builds; int8 RVV > fp32; W8A8 datapath | the old "0 on-board / scalar-lowering bug" was **misdiagnosed**: it was an ABI-marshalling bug in OUR K1 runner glue, not buddy — buddy takes `strided<[?,?]>` **dynamic-strided** operands + a **sret-first** tensor return, and our harness fed it merlin's fixed-8 `sizes/strides` DPS descriptor → inputs read as zeros. Fixed (rank-exact packed descriptors + sret-first call). Residual: W8A8-vs-weight-only-golden rel≈0.012 near-miss at the strict 1e-3 int8 tier |
| **ggml** | tiny_llama int8 **ran** — Q8_0 20.3/5.8 tok/s, Q4_K_M 33/10 tok/s | ✅ native low-bit | correctness **uncomparable** (runs the real checkpoint; our capture is a 2-layer random-init graph) |
| **TVM** | **small_llama int8 PASS on-board — cos=1.0**; openvla int8 ran (cos 0.9916, fail) | 5 models build via **ONNX→Relax** (9–16% untuned RVV) | tuning *one handshake from done* (tracker↔server deadlock over tunnel); big `.so` on-board load hangs; tiny_llama cos=0.22 was a TVM **ONNX `ReduceMean` opset-18 axes-as-input** frontend bug (NOT the "broadcast-Mul" earlier folklore) — **root-caused + fixed → host cos=0.99999999999**; apache/tvm PR prepared (see Phase 4) |

On-board passes now include the **EXO arm's four**: tiny_llama fp32 (cos=1.0) + **int8 (cos=1.0, 7.3 s)**
and small_llama fp32 (cos=1.0) + int8 (cos=1.0). The EXO int8 was fixed by switching from a lossy full
W8A8 activation-quant path (cos=0.949 / 0.998 near-miss vs the wrong reference) to **weight-only int8
(W8A16)** — this full-fidelity capture's int8 `golden.npy` is a weight-only reference (per-channel int8
weight dequantized to f32, activations f32; all zero-points 0; numpy repro matches golden cos=1.000000),
so the glue dequantizes the native int8 weight on the fly inside a transpose-free f32 dot GEMM. Plus
ExecuTorch tiny_llama fp32/int8 and **TVM small_llama int8** (52.7 ms, cos=1.0). TVM also **builds
numerically-correct RVV** for small_llama int8
but its on-board runtime isn't cross-built, so it's `not_run` rather than a fabricated pass. Everything
else is an explicit, reasoned gap.

**TVM's ONNX pivot.** Its torch-exported-program frontend couldn't ingest our HF/VLA graphs, so it was
switched to `relax.frontend.onnx.from_onnx` (opset-17 ONNX export). That unblocked real builds: 5 models
compile with RVV; small_llama int8 is numerically correct end-to-end (ONNX→Relax→rv64gcv→RVV-audit).
rdt int8 cos=0.9992 and openvla int8 cos=0.9916 are near-passes; tiny_llama's low cos was **root-caused
and fixed** (see Phase 4): a TVM Relax-ONNX frontend bug that only read reducer `axes` from attributes,
missing the opset-18 axes-as-*input* form → `ReduceMean` collapsed per-token RMSNorm to a scalar mean.
Fixed → host cos=0.99999999999; small_llama int8 verified on-board cos=1.0 with the fixed frontend.

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

- **Buddy** compiles genuine int8 RVV for 8/11 models and **now runs the fitting ones correctly**
  (small_llama int8 cos=0.99993 on-board) after the K1 runner-glue ABI bug was fixed. The earlier
  "OOM-kills every one" claim conflated two things: (a) a real RAM wall for the multi-GB VLAs
  (whole-model `linalg-to-loops` has no arena planning, so every intermediate is simultaneously live),
  and (b) an ABI-marshalling bug in OUR harness that made *every* model — even the small ones that fit
  — print zeros. (b) is fixed; (a) remains real for the big VLAs and still validates merlin's arena
  runtime. Never a fabricated cos.
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

## Phase 2 — native per-framework flow, full-fidelity, fp32-gated (final)

**Methodology (user-directed):** the shared source of truth is the **PyTorch model** (real/full-fidelity
recaptures at `artifacts/recaptures/<model>_{int8,fp32}_full/`); each framework ingests it via its **own
documented flow** and its **own int8 quantization** (not our torchAO scheme); int8 is gated against the
**fp32 golden**. Fair because merlin also lowers from torch — and, decisively, **using each framework's
native flow knocked down walls that our imposed formats had created:**

| Framework | Native flow used | What it broke (vs the imposed-format result) |
|---|---|---|
| **ExecuTorch** | `examples/models/llama` `WeightOnlyInt8QuantHandler` (eager module swap) | **whole-model int8 exports & the 4.25 GB arena wall is FIXED** — see the dedicated section below. small_llama int8 **PASS on-board 4.26 ms cos=0.999989**; 8-layer real-arch TinyLlama int8 **PASS on-board 918 ms cos=0.99910** (1.35 MB arena, mmap-loaded). Full-22L is export-verified (2.41 MB arena) but its 4.14 GB fp32-const `.pte` currently disk-contends on the shared 14 GB board — no longer a RAM/arena wall. |
| **Buddy** | its own `DynamoCompiler` torch importer (not m2m linalg) | **native IR RUNS on-board** — small_llama int8 completed (**38 ms, 11.8% RVV**); tiny_llama ran the full forward **~28 min with no SIGSEGV**. The m2m-linalg SIGSEGV was an IR-path bug, not fundamental. Remaining: Buddy materializes int8→fp32 → params OOM the board. |
| **ggml** | real checkpoint → GGUF Q8_0, matched input_ids | **comparable cos** (was `None`): Q8_0 **0.9986** on real TinyLlama-1.1B. |
| **EXO** | (kernel DSL — no model import) | ran the real full 22-layer TinyLlama int8 (**9.86 s**). |
| **TVM** | `from_exported_program` (torch→Relax) | **falsified** — TVM's torch frontend can't build our graphs (torchAO subclass / op-coverage); ONNX stays. tiny_llama cos confirmed a TVM ONNX broadcast-`Mul` bug. |

**Coverage:** on-board execution expanded from ~1 model to **6+ framework×model cells**; Buddy builds
**8/11** via native import. **Remaining walls are hardware/framework limits, not effort:** the 3.8 GB board
RAM (Buddy's int8→fp32 materialization, ExecuTorch's whole-model activation buffer, the three 7B-class
VLAs), framework-venv loader deps for some VLAs (mmengine/tyro/lerobot/openpi), TVM's ONNX `Mul` bug, and
correctness on the random-init VLAs is uncorrelated by construction (only tiny_llama + smolvla carry
pretrained weights). Full machine matrix: `artifacts/compare/v1/…/matrix.{md,csv}`.

## Phase 3 — ExecuTorch: the 4.25 GB whole-model int8 arena wall, root-caused and fixed

The prior "full-22L RAM-gaps (4.25 GB activation buffer)" limiter was **not** a fundamental RAM wall —
it was a memory-planner artifact of the weight-only int8 recipe, and it is now fixed.

**Root cause.** `WeightOnlyInt8Linear.forward` is `F.linear(x, weight.to(fp32)) * scales`. That
`int8_weight.to(fp32)` dequant is a *graph op* whose output is a **planned activation tensor**, so
ExecuTorch's greedy planner laid every layer's dequantized fp32 weight at a distinct offset — a
**4,250,738,736-byte non-const arena** for full-depth (22-layer, 155-Linear) TinyLlama (measured by
deserializing the emitted `.pte`). On the board the runner's `make_unique<uint8_t[]>(4.25e9)` throws
`std::bad_alloc` at load. Because the dequant output was seen as co-live across layers, `allow_overlap`
/ greedy reuse could not coalesce it; the XNNPACK delegate boundaries also meant XNNPACK ingested the
**fp32** dequantized weight (so there was no int8 GEMM — the whole-binary RVV stayed ~11.7%).

**Fix (two parts).**
1. **`constant_prop_pass` on the int8-whole-model export** (`_et_export.py`): the dequant's inputs are
   all frozen constants, so folding it turns each fp32 weight into a **program constant**. The planned
   non-const arena collapses **4,250,738,736 → 2,405,952 bytes (~1770×)** at full depth (measured), so a
   layer-reduced or lm-head-trimmed activation buffer is no longer needed to fit.
2. **`executor_runner --mmap_model`** (new flag; ExecuTorch submodule branch `merlin/rv64gcv-mmap-runner`,
   patch under `artifacts/measurements/k1_spacemit/_patches/executorch/`): the const-folded fp32 weights
   now live in the (multi-GB) `.pte` program data, so the stock `FileDataLoader` (reads the whole file
   resident) would re-blow the RAM ceiling. `--mmap_model` loads the program and `.ptd` via
   `MmapDataLoader`/`NoMlock`, demand-paging read-only weight pages that the OS evicts under pressure.

**On-board proof (K1, rv64gcv, VLEN=256).** small_llama int8 whole-model **PASS 4.26 ms cos=0.999989
rel=0.0047**; an 8-layer real-arch TinyLlama int8 whole-model **PASS 918 ms cos=0.99910 rel=0.0425**,
with the runner reporting a **1.35 MB** planned buffer (vs the 4.25 GB `bad_alloc`) and a 17.6 s mmap
load of the 1.67 GB `.pte`. Correctness is gated against the fp32 eager-torch golden for the same config
(weight-only int8 vs fp32). The full-22L `.pte` (4.14 GB) is export-verified with the identical 2.41 MB
arena; its on-board run is currently blocked **only** by concurrent-agent disk contention on the shared
14 GB board (4.14 GB `.pte` vs ~1.9 GB free) — a resource-sharing issue, not a RAM/arena/codegen wall.
XNNPACK still runs fp32 GEMM on the dequantized weights, so whole-binary RVV remains 11.7%; a genuine
int8 XNNPACK `qd8` path is still blocked by the PT2E `transform_for_annotation` cumsum→`index.Tensor`
dtype corruption (re-confirmed here: it fires at PT2E calibration on the full HF-Llama graph, so
dynamic-quant int8 GEMM would need a PyTorch/torchao patch).

## Where things live

- Runners: `merlin/python/merlin/baselines/{buddy,exo,ggml,executorch,tvm}.py` (+ shared
  `bundle/k1_exec/rvv_audit/profile/contract/aggregate`). Submodules: `third_party/baselines/`.
- Raw results: `artifacts/measurements/k1_spacemit/<model>/cross_framework_*/baseline_result.json`.
- Matrix: `artifacts/compare/v1/…/matrix.{md,csv}`. Framework builds: `build/baselines/<fw>/`.

## Phase 4 — final consolidation (authoritative; supersedes earlier text where they conflict)

Directive for this pass: **RAM is not a valid gap** (the board runs these models; nothing has to be
resident all at once — stream/mmap), and **every bug fix must be shaped as an upstream PR**. Five arms
ran concurrently against the live K1 (on-board execution serialized via `board_lock`).

**Authoritative matrix**: regenerate with `.venv/bin/python -m merlin.baselines.aggregate` →
`artifacts/compare/v1/…/matrix.{md,csv}`. Current: **8/79 pass** (EXO ×4, ExecuTorch ×3, TVM ×1),
deduped **latest-executed-per-cell** (`aggregate.dedupe_latest`: an executed pass/fail always beats a
`not_run`/`not_built`, so a timed-out re-verification cannot erase a genuine on-board result).

**The pattern that emerged — most "framework walls" were our own harness bugs, misattributed:**

| Fix | Where the bug actually was | Upstream PR? |
|---|---|---|
| **TVM `ReduceMean` opset-18 axes-as-input** — read `axes` only from attributes, so opset-18 dynamo graphs reduced over all axes → RMSNorm collapse → tiny_llama cos 0.22. Fixed → 0.99999999999; small_llama int8 on-board cos=1.0. | **Genuinely upstream** (apache/tvm Relax ONNX frontend, live on `main`) | ✅ `_patches/tvm/0001-onnx-reduce-axes-input.patch` |
| **ExecuTorch 4.25 GB whole-model int8 arena** — `WeightOnlyInt8Linear`'s `int8→fp32` dequant is a *planned activation* tensor; planner laid every layer's fp32 weight at a distinct offset. `constant_prop_pass` folds it → arena **4.25 GB → 2.41 MB (~1770×)**; new `executor_runner --mmap_model` demand-loads the multi-GB `.pte`. | Our export config + a genuinely-useful runner flag | ✅ `_patches/executorch/0001-executor_runner-add-mmap_model.patch` |
| **Buddy "never runs on-board / scalar-lowering bug"** — MISDIAGNOSED. It was an **ABI-marshalling bug in our K1 glue**: buddy takes `strided<[?,?]>` dynamic-strided operands + a **sret-first** tensor return; our harness fed merlin's fixed-8 `sizes/strides` DPS descriptor → inputs read as zeros (cos≈0). Fixed (rank-exact packed descriptors + sret-first). small_llama int8 on-board cos=0.99993. | **Ours** (`baselines/buddy.py`), proven by host-JIT + K1 micro-repros — buddy is numerically correct | ❌ correctly none (a speculative buddy patch would be wrong) |
| **EXO int8 `fail`** — glue did W8A8; the capture's int8 golden is **weight-only (W8A16)**. Switched to weight-only dequant-on-the-fly → cos=1.0. Also a small-vocab fp32 scratch-overflow SIGSEGV. | Ours (glue), not an EXO compiler bug | ❌ correctly none |
| **ggml tiny_llama `no_gold` + small_llama `not_built`** — golden wired (cos 0.9986); small_llama built via stock `gguf-py` (llama-arch, HF Q/K RoPE permute). | Converter-scope/glue, stock `gguf-py` unmodified | ❌ correctly none |

**Honest reading of the coarse matrix** (do not read cells at face value):
- Most `fail` cells are **quantization near-misses**, not defects: buddy small_llama int8 cos=0.99993
  (W8A8 `rel≈0.012`), ggml 0.9986–0.9991 (Q8_0/Q4_K_M) — real quant error vs the strict weight-only
  0.9999 gate. Frameworks using native W8A8 or their own low-bit near-miss a weight-only golden *by
  construction*; a quant-scheme-aware gate is the honest follow-up.
- EXO's whole-ELF `1%RVV` is libc-dominated; its **compute kernels are 29–31% RVV**. The whole-binary
  figure understates vectorized compute — a known `rvv_audit` artifact. Read per-kernel coverage for the
  compute story.

**Residual gaps — all precise, none "RAM" or "effort":**
- **Board outage (external).** The K1 went SSH-unreachable at the end of this pass (DHCP; IP may change
  on reboot — needs UART/physical recovery). Two on-board items are now **hardware-availability gaps**,
  not code: ExecuTorch's **full-22L** run (export verified: 2.41 MB arena + 4.14 GB `.pte`; execution
  pending a live board) and Buddy's remaining K1-runnable int8 models (bitvla/xr0/rdt2/groot/smolvla).
  Both runners are fail-closed + re-runnable the moment the board returns.
- **Loader-dep wall (the dominant model-coverage gap).** 7 models fail at *torch-load* before any
  framework sees them: `tyro` (groot), `openpi` (pi05), `lerobot` (smolvla), `mmengine` (xr0), and
  custom classes (bitvla BitNet, molmoact, rdt2 numpy-type). These deps exist in each model's own venv
  (per `dev-model-and-m2m-access`); the harness runs in oscar-merlin's `.venv`. Closing it (point the
  export step at each model's venv) unblocks the same models across all arms — but correctness gating is
  only meaningful for smolvla + bitvla (pretrained); the other five are random-init, so run them for
  honest perf/RVV/coverage, not a cos gate.
- **Buddy big-VLA whole-model**: import-time OOM fixed (streamed param write) + on-board weight mmap;
  the standing limit is the **resident activation working set** — buddy's `linalg-to-loops` has no arena
  planning (all intermediates simultaneously live). This validates merlin's arena-runtime thesis.
- **ExecuTorch int8-GEMM RVV**: weight-only int8 → XNNPACK runs fp32 GEMM (11.7% RVV). The int8
  `qd8`/`qs8` RVV ukernel path is blocked by a real PT2E `transform_for_annotation` cumsum→`index.Tensor`
  dtype-corruption bug — a PyTorch/torchao fix, not ours.
- **ExecuTorch whole-model int8 beyond llama (Phase-4 follow-through).** The loader-dep wall is now
  *closed for this arm* by installing the loader deps into the ExecuTorch export venv + two faithful
  non-numeric loader compat shims (`BitNet` capitalized-key config alias; `torch_compilable_check`
  no-op for lerobot-0.6/transformers-5), and the llama-only int8 recipe is generalized by a
  **bias-preserving** per-channel weight-only int8 swap in `_et_export.py` (the official
  `WeightOnlyInt8Linear` drops bias, so its `load_state_dict` rejects the DiT/VLA biased qkv/proj/ffn
  Linears; same math, keeps fp32 bias; bias-free llama path unchanged). Whole-model int8 now **exports**
  for **7/8** K1-runnable models (was 2/8): tiny_llama, small_llama, rdt2, xr0, bitvla, smolvla, rdt
  (rdt's 4.79 GB fp32-const `.pte` RAM/disk-gaps the board; groot's int8 `torch.export` hits a
  model-specific `StopIteration` in dynamo — its fp32 exports). rdt2 int8 confirmed **loading on-board**
  (2.41 MB planned buffer, const-fold verified at full depth) before the shared board went unreachable;
  on-board cos/timing for the newly-exportable set is a hardware-availability gap (fail-closed, re-runnable),
  never fabricated.
- **ggml VLA scope**: structural, not RAM — ggml runs token-in→causal→vocab-logits; the VLA captured
  forwards are inputs_embeds/bi-attn (bitvla), multimodal fusion (openvla), or diffusion/flow action
  heads (rdt/rdt2/xr0/pi05/groot/smolvla) with no token/logit surface.
- **TVM**: MetaSchedule tuning still one handshake from done (tracker↔server deadlock over the tunnel);
  big-`.so` on-board VM-load hang (108 MB) → tiny_llama on-board is `not_run` (host cos=1.0 recorded).
- **EXO**: at its limit for this workload (1.44 s, no dominant cost center). Further gains would need
  RVV rmsnorm/softmax (currently small) or EXO-schedulable stride-1 reduction support (the dot GEMM is hand-RVV).
