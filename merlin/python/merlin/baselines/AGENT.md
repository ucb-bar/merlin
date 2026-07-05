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

- **Build TVM against LLVM 18, NOT LLVM 23.** The repo's `third_party/llvm-install` (LLVM 23) does
  NOT compile this pinned TVM (`ff937ff`): its `src/target/llvm/*` uses APIs removed in LLVM 23
  (`TargetOptions::NoInfsFPMath`, `Intrinsic::matchIntrinsicSignature`, the ORC `ObjectLinkingLayer`
  ctor). The system `/usr/bin/llvm-config-18` (18.1.3, RISC-V + `+v` present) builds it clean.
  Config: `build/baselines/tvm/config.cmake` sets `USE_LLVM=/usr/bin/llvm-config-18`, `USE_RPC=ON`;
  `cmake -G Ninja -C config.cmake <tvm-src>` then `ninja` → libs land in `build/baselines/tvm/lib/`.
- **`tvm_ffi` must be pip-installed** (its `core` cython extension can't run from a source dir). It
  is installed editable into the model2MLIR venv: `uv pip install --no-build-isolation -e
  third_party/baselines/tvm/3rdparty/tvm-ffi` (needs cython + scikit-build-core).
- **Runs in the model2MLIR venv** (`$MERLIN_MODEL2MLIR/.venv`, torch 2.x + transformers) — the main
  `.venv` has no torch and TVM's Relax torch frontend needs it. `tvm.py` drives a subprocess there
  with `PYTHONPATH=<tvm>/python` and `LD_LIBRARY_PATH=<lib>` (tvm_ffi's libinfo searches
  `LD_LIBRARY_PATH`, not `TVM_LIBRARY_PATH`).
- **Target is a JSON dict, not a CLI string** (this TVM dropped `"llvm -mcpu=..."`); `mattr` is a
  list. See `TVM_TARGET_CONFIG` (`+v,+zvl256b` for the K1's 256b VLEN). `cc` moved to
  `tvm.support.cc` (was `tvm.contrib.cc`).
- **Known blocker (honest gap):** this pinned TVM snapshot can't lower the LLM/VLA graphs — the LLVM
  codegen asserts on non-int/non-float (bool) binary ops (`CreateAdd`: `MatchesCode(kDLFloat)`), and
  `small_llama` hits a Python-side `'PrimType' is not iterable`. A trivial float MLP compiles +
  cross-links + RVV-audits fine (harness proven), so these are TVM-snapshot defects recorded as
  `not_built` with specific reasons, not harness bugs. Coverage on unscheduled TVM kernels is low
  (no MetaSchedule tensorize in this TVM; TIR→LLVM emits scalar loops) — the audit records that.
