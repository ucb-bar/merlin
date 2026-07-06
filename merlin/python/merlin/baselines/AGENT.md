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
- **Import via ONNX, NOT the torch-exported-program frontend.** TVM v0.19.0's
  `relax.frontend.torch.from_exported_program` lacks ops for HF transformer/VLA graphs
  (`full`/`where`/`masked_fill`/`convolution`/…, plus torchao int8 subclasses). Its **ONNX** frontend
  (`relax.frontend.onnx.from_onnx`) has far broader coverage. So the driver does
  `torch.onnx.export` (classic opset17; dynamo opset18 fallback for rope's `aten::diff`) → `from_onnx`.
  Two compat shims: (1) reconstruct `onnx.mapping` (removed in onnx 1.22, still imported by the TVM
  frontend) from `onnx.helper`; (2) register `relax.isnan`/`isinf` legalizations (missing → VM
  codegen rejects the un-lowered intrinsic). m2m venv needs `onnx onnxruntime onnxscript` +
  `decorator tornado cloudpickle attrs`.
- **int8-first** (`variant="int8"` default): reproduces the capture's torchao quant (`apply_quantization`
  via `_quant_for`), gates vs `golden_w8a8.npy` when present (`golden_path`). Non-persistent buffers
  are re-registered persistent before export (torch 2.10 lifts rope `inv_freq` into buffers the
  frontend can't resolve). Correctness is gated vs a **torch reference for the exact instance**
  (`host_cos`); `gold_cos` (vs the capture golden) is also reported.
- **Status: 5 int8 models BUILD via ONNX** (Relax import → rv64gcv `relax.build` → SpacemiT
  cross-link → real riscv64 RVV `.so` → RVV-audit), torchao int8 quant applied. **small_llama int8
  is numerically correct on host (cos 0.99999999)**; rdt 0.9992, openvla 0.9916 (near-pass);
  **tiny_llama cos ≈ 0.80** — TVM v0.19.0 mis-lowers an attention-block op (same mean/std as ORT but
  positionally scrambled, argmax 2/8; **ORT on the identical ONNX matches torch at cos 1.0**, so a
  TVM ONNX-frontend bug, model-specific). Gaps: rdt2 ONNX-export; bitvla/molmoact loader; groot/xr0/
  pi05/smolvla miss python deps (`tyro`/`mmengine`/`openpi`/`lerobot`); pi05 16 GB (K1 fit gap).
- **RVV coverage** on the default `relax.build` lowering is ~9–16% (LLVM vectorizes some loops for
  the `+v,+zvl256b` target). Higher RVV needs **MetaSchedule autotuning** — but see the network note.
- **On-board execution WORKS** (`build/baselines/tvm-rv64/`: riscv64 runtime + `tvm_rpc` cross-built
  with SpacemiT clang rv64gcv via `riscv64-toolchain.cmake`, `ninja tvm_runtime tvm_rpc`).
  `_run_on_board`→`_rpc_run_driver` (m2m-venv subprocess) deploys them, starts a persistent server,
  connects host→board directly, runs the relax VM, wall-times, gates cos/rel. **On-board results
  (untuned): small_llama int8 = 52.7 ms cos 1.0 (PASS); openvla int8 = 6514 ms cos 0.9916 (fail —
  ran but under the 0.9999 gate).** rdt (435 MB .so) / tiny_llama (877 MB .so) exceed practical RPC
  upload on this board channel (honest gap). Four non-obvious fixes were required
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
- **MetaSchedule on-device tuning — reverse tunnel PROVEN, full tune not landed (honest gap).** The
  board firewall blocks the direct **board→host** tracker callback, so an **ssh reverse tunnel**
  (`ssh -R 9190:localhost:9190`, host tracker) is required. Confirmed working: the board `tvm_rpc
  --tracker=localhost:9190 --key=k1` **registers to the host tracker via the tunnel** (`summary()`
  showed `server:k1`), 45 tunable TIR tasks extract after `LegalizeOps` (raw from_onnx = 0), and a
  tune run reached the MetaSchedule cost-model step (measurement round-trip worked) — the missing
  `xgboost` there is now installed. Remaining blockers, honestly recorded: (a) the tracker hands the
  host runner the server's self-advertised addr `127.0.0.1:9294`, which is not host-reachable —
  needs `--custom-addr=<board-ip>` or a forward tunnel for the RPC port; (b) tracker/tunnel/server
  process lifecycle is fragile to launch from a transient shell (the tracker `--port-end` is
  EXCLUSIVE — use `--port=9190 --port-end=9195`, not `=9190`). A persistent daemon (systemd on the
  board) would make this robust. So tuned RVV/latency are NOT yet measured; untuned RVV stays ~9–16%.
  Fail-closed `not_run` when the on-board run doesn't complete; weights too big for RPC upload / >3 GB
  (pi05) are labeled gaps.

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
- **int8 W8A8 path (`int8_subgraph=True`, XNNPACK qs8 RVV):** whole-model PT2E is IMPOSSIBLE on HF
  Llama here — `prepare_pt2e`'s `transform_for_annotation` pass corrupts an integer-index dtype (the
  position/causal-mask `aten.index.Tensor` on a `cumsum`) at CALIBRATION. It fails **even with an
  EMPTY quantizer** (no annotations), so it is the transform pass, NOT the observers/annotation — no
  `set_module_type`/`set_module_name`/`set_operator_type`/`filter_fn` exclusion can dodge it (the
  raw exported graph runs fine; matches the upstream `aot_riscv.py` note that HF Llama trips the
  quantizer). So `_linear_subgraph` quantizes the maximal self-contained pure-Linear region: **ALL
  of a decoder layer's Linears — attention q/k/v/o projections + SwiGLU MLP gate/up/down (7 Linears,
  actual trained weights)**, seeded fp32 hidden-state input, with the non-linear glue (embedding,
  RoPE index, causal mask, softmax, RMSNorm) kept fp32 and OUT of quantization. Static W8A8 fuses
  the 7 Linears into **2 XNNPACK qs8 delegates (delegated 2/8 nodes)**, dispatching to
  `xnn_qs8_qc8w_gemm_..._4x4v__rvv` (int8 GEMM inner loop RVV; requantize/clamp tail scalar; 8+
  qs8/qc8w RVV GEMM ukernels linked). **On-board PASS: cos=0.99987 (int8 vs fp32), rel=0.017,
  13.7 ms** (int8 gate 0.99/0.05 — W8A8 loses precision vs the fp32 golden; cos is the MEASURED
  int8-vs-fp32 cosine, not faked). This is the maximal int8 an HF-Llama export allows on this
  ExecuTorch/PT2E; `subgraph_note` records exactly which ops forced fp32.
