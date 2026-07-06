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
- **Status (this session):** the ONNX path gets tiny_llama int8 **built** — Relax import → rv64gcv
  `relax.build` → SpacemiT cross-link → real riscv64 RVV `.so` → RVV-audit (**~16% RVV**, 61 scalar
  fallbacks), torchao int8 quant applied. **Remaining blocker: correctness cos ≈ 0.80** — TVM
  v0.19.0 mis-lowers an attention-block op (same mean/std as ORT but positionally scrambled, argmax
  2/8; **ORT on the identical ONNX matches torch at cos 1.0**, so it is a TVM ONNX-frontend bug, not
  ours). Per-model gaps: bitvla/molmoact loader (`'BitNet'`/`'default'`), groot/xr0/pi05/smolvla miss
  python deps (`tyro`/`mmengine`/`openpi`/`lerobot`), pi05 16 GB (K1 fit gap).
- **RVV coverage** on the default `relax.build` lowering is ~16% (LLVM vectorizes some loops for the
  `+v,+zvl256b` target). Higher RVV needs **MetaSchedule autotuning** (`MERLIN_TVM_TUNE=1` + a K1 RPC
  runner) — opt-in because on-device tuning queues on the one shared board.
- **On-board run** (`_run_on_board`) is still fail-closed: a standalone TVM riscv64 runtime +
  `tvm_rpc` (or a C harness linking `libtvm_runtime`) must be cross-built for the board — the
  remaining board step. Weights > ~3 GB (pi05) are a labeled fit gap.

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
- **int8 W8A8 path (`int8_subgraph=True`, XNNPACK qs8 RVV):** full-model PT2E FAILS on HF Llama —
  `prepare_pt2e`'s inserted observer corrupts an integer-index dtype at CALIBRATION (the position/
  causal-mask `aten.index.Tensor` on a `cumsum`); the raw exported graph runs fine, so it is the
  observer insertion, not the annotation — no `set_module_type`/`set_operator_type`/`filter_fn`
  toggle avoids it (matches the upstream `aot_riscv.py` note that HF Llama trips the quantizer).
  The working fallback quantizes the model's REAL linear-heavy subgraph (layer-0 SwiGLU MLP:
  gate/up/down Linears + SiLU, actual trained weights, seeded fp32 hidden-state input), keeping
  embeddings/attention/mask fp32 and OUT of quantization. Static W8A8 fuses the whole int8 MLP into
  ONE XNNPACK qs8 delegate (**delegated 1/2 nodes**, vs the fp32 arm's 10/93), dispatching to
  `xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4v__rvv` (int8 GEMM inner loop is RVV; requantize/clamp
  tail scalar; 15 int8 qs8/qd8/qc8w RVV GEMM ukernels linked). **On-board PASS: cos=0.99909 (int8
  vs fp32), rel=0.043, 10.9 ms** (int8 gate 0.99/0.05 — W8A8 loses precision vs the fp32 golden; cos
  is the MEASURED int8-vs-fp32 cosine, not faked). Full-model int8 honestly labeled not-exportable;
  the subgraph is a genuine tiny_llama slice, not a toy.
