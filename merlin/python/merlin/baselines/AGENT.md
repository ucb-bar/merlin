# AGENT.md — merlin/python/merlin/baselines

## Purpose

External-baseline K1-RVV comparison harness.

## Modules

- `aggregate.py` — Collect per-framework results into the cross-framework matrix (merlin vs the 5 baselines).
- `buddy.py` — Buddy (buddy-mlir) baseline arm — ingest OUR ``model.mlir`` and run it on the K1 with RVV.
- `bundle.py` — Resolve a ``(model, variant)`` to its capture bundle — the shared input every baseline ingests.
- `contract.py` — Result contract for external-baseline K1-RVV runs (the shared honesty schema).
- `k1_exec.py` — Generic K1 deploy/run for external baselines + a board lock (single physical board).
- `profile.py` — Two-level profiling: whole-model E2E + per-region "kernel-style" breakdown.
- `rvv_audit.py` — RVV-coverage audit — the mechanical honesty behind "push RVV, label scalar fallback".

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Invariants (read before adding a framework)

- **not_run_is_not_pass** — a model that doesn't build/run is a `not_built`/`not_run` cell with a
  non-empty `gap_reason`, never an omission. `BaselineResult.validate()` enforces it.
- **Push RVV, label scalar honestly** — enforce `-march=rv64gcv`/`+v` via
  `rvv_audit.enforce_rvv_march` at build time, then `rvv_audit.audit_binary(elf)` the emitted
  artifact and record `rvv_coverage_overall` + a `ScalarFallback` per compute-bearing symbol that
  stayed scalar. Never average the fallback away.
- **Two-level profiling** — every runner emits BOTH `MERLIN_E2E` and `MERLIN_REGION` markers
  (see `profile.py`); region names use the `REGIONS` taxonomy so cross-framework region diffs work.
- **Single board** — wrap build+push+run+parse in `k1_exec.board_lock()`; builds may parallelize,
  on-board runs serialize. `board_available()` is fail-closed.
- **Cycles are estimates** — K1 `rdtime` → est core cycles (`cycle_accurate=False`); spike/FireSim
  remain the cycle authorities.

## Buddy (buddy.py) — build + int8 SIGSEGV diagnosis

- **Build**: `git submodule update --init llvm` in `third_party/baselines/buddy-mlir`, then build the
  LLVM fork tools (`mlir-opt mlir-translate llc opt mlir-runner`) into
  `build/baselines/buddy/llvm-build/`. Codegen stays INSIDE the fork (`opt`+`llc`), NOT the IREE
  clang-23 — its IR parser rejects the fork's `float f0x…` hex-float literals.
- **Pipeline** (`_LOWER_PASSES`): `-llvm-request-c-wrappers` MUST precede `-convert-func-to-llvm`
  (else no `_mlir_ciface_forward` export). `-buffer-deallocation-pipeline` (+ hoisting) after
  bufferize is REQUIRED or every intermediate stays live → K1 OOM. For IR > 64 MB (pi05), a bounded
  `-passes=function(loop-vectorize,slp-vectorizer,…)` replaces superlinear whole-function `-O3`.
- **int8** = a real W8A8 integer datapath via `prepare_model_mlir` (merlin's
  `_prepare_model_mlir(int8_compute=True)`): i8×i8→i32 (`arith.extsi`+MAC), which `opt` vectorizes to
  `vwmacc.vv`. Gate int8 vs `golden_w8a8.npy` (`_golden_for`) when present.
- **int8 on-board SIGSEGV (open, buddy bug)**: 8/11 int8 models BUILD with real integer RVV but every
  one segfaults on the K1. Diagnosed (micro-case + scalar/RVV bisection): fault is a scalar `flw` on a
  malloc'd f32 buffer at a bad offset in an f32-elementwise region of `forward`. RULED OUT — the
  isolated int8 matmul+dequant micro-case RUNS CORRECTLY on the K1 (so NOT the dequant pattern); the
  descriptor/stride ABI (merlin's own lowering uses the identical packed rank-N ciface and runs int8);
  RVV (the SCALAR rv64gc build also segfaults — NOT a vector-tail bug); dealloc/opt-O3. It is a buddy
  bufferization/lowering bug for some whole-model op (bad dynamic buffer size/offset) not triggered by
  the isolated pattern; pinpointing needs whole-model op bisection. fp32 runs but is ~18 min/forward.

## Adding a per-framework runner (Part C)

Create `baselines/<framework>.py` that: resolves the bundle (`bundle.resolve`), cross-compiles for
`rv64gcv` (SpacemiT clang via `merlin.rvvgen.k1.toolchain_cc`, or the framework's own LLVM with
`+v`), runs each supported model under `board_lock()`, gates `cos/rel` vs `golden.npy` at
`bundle.tolerance(model)`, fills a `BaselineResult`, and writes it into a
`new_measurement("k1_spacemit", model, "cross_framework")` dir. Build trees go under `build/<fw>/`.
Then `aggregate.collect_dir(...)` renders the merlin-vs-baselines matrix into `artifacts/compare/`.

## TVM arm build (`tvm.py`)

- **Pinned to TVM v0.19.0** (submodule gitlink `c4dc0c2`, tag pinned in `.gitmodules`
  `branch = v0.19.0`). Re-pinned FROM the `main` snapshot `ff937ff`, which (a) wouldn't compile
  against LLVM 23, (b) shipped no MetaSchedule, and (c) had a bool-op LLVM-codegen bug. v0.19.0 has
  MetaSchedule + AutoTVM + the classic `tvm.contrib.cc`, and its codegen lowers our graphs.
- **Build against system LLVM 18** (`/usr/bin/llvm-config-18`, 18.1.3, RISC-V + `+v`) — NOT the
  repo's LLVM 23. `build/baselines/tvm/config.cmake` sets `USE_LLVM=/usr/bin/llvm-config-18`,
  `USE_RPC=ON`; `cmake -G Ninja -C config.cmake <tvm-src>` then `ninja`. v0.19.0 puts the libs in
  the **build root** (`build/baselines/tvm/libtvm.so`), not a `lib/` subdir — `tvm_lib_dir()` handles
  both. v0.19.0 needs NO separate tvm_ffi cython install (self-contained `_ffi`).
- **Runs in the model2MLIR venv** (`$MERLIN_MODEL2MLIR/.venv`, torch 2.x + transformers) — the main
  `.venv` has no torch and TVM's Relax torch frontend needs it. `tvm.py` drives a subprocess there
  with `PYTHONPATH=<tvm>/python` + `LD_LIBRARY_PATH=<libdir>`. That venv needs a few TVM runtime
  deps: `uv pip install decorator tornado cloudpickle attrs` (numpy/ml_dtypes/psutil already there).
- **Target = JSON dict** `TVM_TARGET_CONFIG` (`+v,+zvl256b` for the K1's 256b VLEN, `num-cores=8`).
- **Import via ONNX, NOT the torch-exported-program frontend (re-confirmed Phase 2).** TVM v0.19.0's
  `relax.frontend.torch.from_exported_program` **cannot build these graphs** — verified directly on
  the full-fidelity tiny_llama: int8 fails at `access_subclass_inner_tensor.default` (torchao int8
  tensor subclass unsupported); fp32 fails at `arange.default`, and even with an arange/mm/split
  op-alias shim + `run_decompositions` it hits `ne.Scalar` (comparison ops missing) — an unbounded
  op-coverage whack-a-mole. So the Phase-2 hypothesis "the torch path fixes tiny_llama cos=0.80" is
  **falsified: the torch path never reaches codegen**. The **ONNX** frontend
  (`relax.frontend.onnx.from_onnx`) has far broader coverage and is the only path that builds. Driver
  does `torch.onnx.export` (classic opset17; dynamo opset18 fallback for rope's `aten::diff`) →
  `from_onnx`. (Full-fidelity confirms the RMSNorm broadcast-Mul defect: full 22-layer tiny_llama int8
  builds but cos **0.219** — worse than the truncated 2-layer 0.80 because the per-layer Mul error
  compounds across 22 layers.)
  Two compat shims: (1) reconstruct `onnx.mapping` (removed in onnx 1.22, still imported by the TVM
  frontend) from `onnx.helper`; (2) register `relax.isnan`/`isinf` legalizations (missing → VM
  codegen rejects the un-lowered intrinsic). m2m venv needs `onnx onnxruntime onnxscript` +
  `decorator tornado cloudpickle attrs`.
- **int8-first** (`variant="int8"` default): reproduces the capture's torchao quant (`apply_quantization`
  via `_quant_for`), gates vs `golden_w8a8.npy` when present (`golden_path`). Non-persistent buffers
  are re-registered persistent before export (torch 2.10 lifts rope `inv_freq` into buffers the
  frontend can't resolve). Correctness is gated vs a **torch reference for the exact instance**
  (`host_cos`); `gold_cos` (vs the capture golden) is also reported.
- **Full-fidelity recaptures (Phase 2):** `bundle.resolve` prefers the `<model>_int8_full` bundle
  (real/native architecture) over the truncated `_consistent`. When the resolved dir ends `_full`,
  `_workload_env(model, full=True)` uses `bundle.full_env(model)` (real depths, e.g. 22-layer
  TinyLlama / 30-layer BitNet) so the exported instance matches the full golden — NOT the TOML
  truncation defaults. `bundle.K1_RUNNABLE` = 8 models that fit the ~3.4 GB board;
  `bundle.K1_RAM_INFEASIBLE` = {openvla, molmoact, pi05} (7B-class, attempt-build then RAM-gap).
- **Status: 5 int8 models BUILD via ONNX** (Relax import → rv64gcv `relax.build` → SpacemiT
  cross-link → real riscv64 RVV `.so` → RVV-audit), torchao int8 quant applied. **small_llama int8
  is numerically correct on host (cos 0.99999999)**; rdt 0.9992, openvla 0.9916 (near-pass);
  **tiny_llama cos ≈ 0.80 — root-caused by per-op bisection:** the first diverging op (topological,
  float output, inputs still correct) is the **RMSNorm broadcast `Mul([1,8,2048] × [1,8,1])`** —
  TVM gives cos 0.968 there while `numpy(a*b) == ORT` = cos 1.0 with the SAME inputs, so it is a
  genuine **TVM v0.19.0 ONNX broadcast-multiply defect**, not ours; the 0.968 compounds through the
  layers to 0.80 at the logits. (The topo-first raw diverger `IsNaN` cos 0.0 is a red herring — its
  `Where(IsNaN(x),y,x)` guard resolves to `x` on finite data; forcing IsNaN→all-False leaves cos
  0.80.) Fixing needs a TVM broadcast-Mul patch (C++/TIR) — left as an honest fail/0.80.
  Gaps: rdt2 ONNX-export; bitvla/molmoact loader; groot/xr0/pi05/smolvla miss python deps
  (`tyro`/`mmengine`/`openpi`/`lerobot`); pi05 16 GB (K1 fit gap).
- **Phase-2 full-fidelity breadth (partial — full models are heavy):** on the `_full` recaptures the
  full 22-layer tiny_llama int8 builds via ONNX (RVV 15.8 %) but cos = **0.219** (the RMSNorm
  broadcast-Mul error compounds across 22 layers vs 0.80 at 2 layers). Full-fidelity ONNX
  export + the second host-VM `relax.build` are ~5 min PER model, so a synchronous 11-model host-cos
  sweep is impractical (and contends with the concurrent ExecuTorch arm on the shared board). The
  runner + `bundle.full_env`/`K1_RUNNABLE`/`K1_RAM_INFEASIBLE` wiring are in place; the definitive
  Phase-2 conclusion (torch path can't build → ONNX only; RMSNorm-Mul defect confirmed on the real
  model) stands. Per-model full-fidelity build/cos/latency for all 8 K1-runnable models is a
  follow-up sweep (each model is an independent `run_model(m, "int8")`), not a blocker.
- **RVV coverage** on the default `relax.build` lowering is ~9–16% (LLVM vectorizes some loops for
  the `+v,+zvl256b` target). Higher RVV needs **MetaSchedule autotuning** — but see the network note.
- **On-board execution WORKS** (`build/baselines/tvm-rv64/`: riscv64 runtime + `tvm_rpc` cross-built
  with SpacemiT clang rv64gcv via `riscv64-toolchain.cmake`, `ninja tvm_runtime tvm_rpc`).
  `_run_on_board`→`_rpc_run_driver` (m2m-venv subprocess) deploys them, starts a persistent server,
  connects host→board directly, runs the relax VM, wall-times, gates cos/rel. **On-board results
  (untuned): small_llama int8 = 52.7 ms cos 1.0 (PASS); openvla int8 = 6514 ms cos 0.9916 (fail —
  ran but under the 0.9999 gate).** rdt (435 MB) / tiny_llama (877 MB) .so's: the RPC-channel
  `sess.upload` is impractically slow, so we scp the .so **directly** to the board tmpfs (rdt 435 MB
  in ~32 s) and `load_module` it from the server work-dir without re-upload — BUT loading a 435 MB
  relax executable over the RPC session then **hangs** (the child blocks on the socket read). So the
  big-.so on-board run is an honest gap (direct scp works; the on-board relax-VM load of a
  hundreds-of-MB module over RPC does not complete). Four non-obvious fixes were required
  (all in `_RPC_RUN_TEMPLATE`):
    1. **`tvm_rpc` parses only `--opt=value`** (space form is silently ignored → defaults); pass
       `--port=… --port-end=… --work-dir=…` with `=`.
    2. **The server EXITS on stdin EOF** — a plain `nohup`/`setsid` over ssh dies. Launch it from the
       host via an ssh whose stdin is a never-closing **FIFO**, so the remote server never sees EOF.
    3. **Work dir on tmpfs** (`/tmp/tvm_work`) — the board `/root` (~14 G) is often 100% full from
       other agents' weights, so uploads ENOSPC; `/tmp` (1.9 G tmpfs) has room.
    4. **relax VM over RPC needs `set_input`/`invoke_stateful`/`get_outputs`** — the direct
       `vm["main"](*args)` closure call mis-marshals remote NDArrays (`ArrayHandle` vs `Object`).
  Use our own port range (9193-9199); the other 4 agents' servers hold 9090/9091.
- **MetaSchedule on-device tuning — tracker deadlock BYPASSED via a direct-RPC runner; tuning runs
  and measures on-board but STALLS mid-run (honest gap).** The tracker path deadlocks (board→host
  firewalled; even with an `ssh -R 9190` reverse tunnel + a board systemd `tvm_rpc
  --tracker=localhost:9190 --key=k1 --custom-addr=10.44.97.186` that registers correctly as
  `server:k1 @ 10.44.97.186:9294`, `tracker.request('k1')` hangs at `free 0 pending 1` — the
  tracker↔server session handshake over the tunnel never completes). **The working angle is to skip
  the tracker entirely:** pass `RPCRunner(f_create_session=<callable>)` a picklable session-creator
  that does `rpc.connect(board_ip, 9193)` DIRECTLY (the proven host→board channel). Key API details:
  `f_create_session` must be a **top-level picklable callable** (a `@register_func` name is NOT
  resolvable in the PopenPool worker — `get_global_func_with_default_on_worker` returns a Callable
  as-is); reads host/port from env so it needs no closure. With this, `tune_relax` **runs and
  measures candidates on the K1** — the scheduler sent 64-sample batches to the runner and finished
  tasks #0/#1 (`extract_tasks`=45 after `LegalizeOps`; `xgboost` cost model installed). **Remaining
  blocker:** it **stalls after ~2 tasks** — task #2's on-board measurement RPC hangs (worker in
  `futex_wait_queue`, board idle, ~5 min no progress), the same sustained-RPC-load instability that
  hangs the big-.so load. So the tune does not complete → **no tuned .so built; tuned RVV/latency
  NOT measured; untuned stays ~9–16 %.** A real attempt was made (direct-RPC runner is the closest
  path — it measures on-board where the tracker path measured nothing); recorded honestly, not
  retried forever. Needs a more robust board RPC transport (per-measurement fresh session already
  used; likely the C++ tvm_rpc server's stability under repeated large uploads). Fail-closed
  `not_run` when a run doesn't complete; big .so's / weights > ~3 GB (pi05) are labeled gaps.

## ExecuTorch + XNNPACK arm build (`executorch.py`)

- **The forced-whole-model, most-scalar-fallback arm.** ExecuTorch delegates only what the XNNPACK
  partitioner claims to XNNPACK's RVV microkernels and runs everything else on its **portable
  (scalar) reference kernels**. This is labeled, not hidden: the runner-binary RVV audit + a
  `ScalarFallback` per portable compute symbol carry it. tiny_llama fp32 delegated only 10/93 graph
  nodes → **~11.7% binary RVV coverage, 2185 labeled scalar fallbacks** (real, on-board).
- **Export runs in a dedicated venv** (`build/baselines/executorch/et-venv`, gitignored) built via
  `third_party/baselines/executorch/install_executorch.sh` — it pulls ExecuTorch's pinned torch
  (2.12) + the ET pip package; `transformers` is added on top for the HF loaders. The main `.venv`
  has none of these. `executorch.py` shells `_et_export.py` into that venv (must strip its own dir
  from `sys.path` first, else the sibling `executorch.py` shadows the installed `executorch` pkg).
- **Cross-compile via the source tree's `riscv64-linux` cmake preset**, overriding the toolchain
  file with `executorch_spacemit_toolchain.cmake` (SpacemiT clang-19, `-march=rv64gcv -mabi=lp64d`
  on the WHOLE build) + `-DEXECUTORCH_BUILD_XNNPACK=ON` (XNNPACK's 380 RVV ukernels, compiled
  `-march=rv64gcv`). `PYTHON_EXECUTABLE` MUST be the ET venv python (host codegen: gen_oplist).
- **RVV audit uses `llvm-objdump`, NOT the SpacemiT GNU objdump.** The GNU
  `riscv64-unknown-linux-gnu-objdump` silently mis-decodes rv64gcv in bulk `-d` (emits ~3 vector
  insns for a binary with ~85k) → would fabricate a false 0% RVV. `_preferred_objdump()` pins
  `llvm-objdump` and passes it to `rvv_audit.audit_binary(objdump=...)` (the shared `rvv_audit` is
  NOT patched — other arms depend on it).
- **`.pte` + external `.ptd`**: a whole fp32 LLM's weights blow past flatbuffer's 2 GB program
  limit, so `to_executorch(external_constants=True)` splits weights into a `.ptd`
  (`--data_path`). Input fed as raw bytes (`--inputs=input0.bin`); output dumped
  (`--output_file`) and cos/rel computed OFF-DEVICE vs golden (never fabricated).
- **Board is shared + disk-constrained** (~14 GB rootfs, often >90% full from concurrent arms). The
  full fp32 tiny_llama `.pte` (~4.1 GB) does NOT fit → `_run_on_board` checks free space and
  fail-closes as a `not_run` board-fit gap. A layer-reduced fit-on-board config
  (`export_env={'M2M_LLAMA_LAYERS':'1'}` + `compute_golden=True`, gate = eager-torch-vs-ExecuTorch)
  proves the ExecuTorch+XNNPACK-RVV path end-to-end on real K1 silicon (cos 0.9999999999, rel 2.7e-6,
  217 ms wall).
- **WHOLE-MODEL int8 (`int8_whole_model`, the default for `variant="int8"`) — ExecuTorch's OFFICIAL
  llama recipe:** generic PT2E is IMPOSSIBLE on HF Llama (`prepare_pt2e`'s `transform_for_annotation`
  pass corrupts an integer-index dtype — the position/causal-mask `aten.index.Tensor` on a `cumsum`
  — at calibration; fails **even with an EMPTY quantizer**, so it is the transform pass, not the
  observers, and no `set_module_*`/`filter_fn` exclusion dodges it). The fix is the official
  `examples/models/llama/source_transformation/quantize.py` **`WeightOnlyInt8QuantHandler`**:
  weight-only int8 per-channel by **eager MODULE SWAP** (every `nn.Linear` → `WeightOnlyInt8Linear`).
  Because it is a module swap it NEVER runs `transform_for_annotation`, so it **fully sidesteps the
  index corruption** — **all layers quantize + the whole model exports**. tiny_llama (full 22 layers
  + lm_head, 155 Linears): whole-model int8 exports, **cos=1.0000 (weight-only int8 is near-lossless
  vs fp32)**, delegated ~23/216 nodes to XNNPACK qs8 RVV. Gate 0.99/0.05 (cos is the MEASURED
  int8-vs-fp32 cosine). The fp32 glue (embedding/RoPE/mask/softmax/RMSNorm) stays fp32.
- **`int8_subgraph=True` is the FALLBACK** (only if whole-model won't export): quantizes the decoder
  layer's linear set (q/k/v/o + gate/up/down) on a seeded hidden state, 2 qs8 delegates, cos 0.99987,
  13.7 ms — superseded by the whole-model path for Llama-family models.
