# `kernels/qnn/` — agent guide

## Mental model

Qualcomm QNN backend for the kernel-embedding framework. The compile
pipeline (`tools/compile/cli.py`) feeds dispatches through this package
when the manifest's `source_lang == "qnn-context-binary"`, producing
`.qnn-ctx` blobs that the IREE QNN HAL driver loads on a Qualcomm device.

The 7 core modules are organized by pipeline stage:

| Module | Role |
|---|---|
| `emit.py` (1190 LOC) | MLIR → QNN graph emitter. v1 regex parser; production path. |
| `ir.py` | Compact intermediate representation for QNN graphs. Shared by emit + build. |
| `partition.py` | Splits a multi-anchor MLIR module into per-island subgraphs. |
| `route.py` | Picks the per-(island, target) backend based on a profile CSV. |
| `gates.py` | Validation gates for the heterogeneous compile pipeline. |
| `build.py` | Toolchain orchestration: `qairt-converter` → `.qnn-ctx`. |
| `precompile_extras.py` | Hook called from `kernels/core/precompile.py` when a kernel's `source_lang == "qnn-context-binary"`. |

`recognizers/` holds 14 pattern matchers — one file per op family (NCHW
int8 conv, NHWC int8 conv, depthwise conv, maxpool, concat, reshape,
elementwise binary/unary, …). Adding a new pattern: drop a new file in
`recognizers/`, expose `NAME` + `try_recognize`, register in
`recognizers/__init__.py:REGISTRY`.

`headers/QnnKernelHelpers.hpp` is the C++ side consumed by runtime code.

For the test surface, see `tests/` — 5 active files
(`test_qnn_build`, `test_qnn_partition_correctness`, `test_qnn_route`,
`test_qnn_phase5_gates`, `test_qnn_phase6_multimodel`). An older v2
emitter attempt was archived to `tools/archive/qnn_v2/`.

## Pitfalls

- **v1 (`emit.py`) is the production path.** A v2 bindings-based emitter
  was attempted and never reached parity; it's archived. Don't restart
  the v1→v2 migration without first reading
  `tools/archive/qnn_v2/README.md` to know what blocked v1→v2 parity.
- **Recognizer order matters.** Recognizers are tried most-specific
  first. The NHWC int8 conv recognizer must come before the NCHW one in
  `REGISTRY` because NHWC is HTA-compatible while NCHW lowers with a
  Transpose adapter that HTA and Adreno reject. Reordering breaks
  yolov8 / nchw-anchored models.
- **Source must be `.qnn-ctx` or `.qnn.cpp`.** Other suffixes raise.
  Pre-built blobs use the passthrough cache; `.qnn.cpp` goes through the
  QNN SDK (host-side via `qnn-context-binary-generator` or board-side
  via `QNN_USE_BOARD_BUILD=1`).
- **Env-var contract for board builds**: `QNN_USE_BOARD_BUILD=1`,
  `QNN_BOARD_HOST=qdev`, `QNN_BOARD_QAIRT_ROOT=/tmp/qnn_probe`. Required
  when the kernel uses fp16/uint8 — libQnnCpu rejects those on the host
  validation path.
- **Late-imported from `core.precompile`.** The dispatch hook in
  `kernels/core/precompile.py` does `from kernels.qnn.precompile_extras
  import compile_qnn_kernel` lazily so the QNN SDK Python overhead
  doesn't load when no QNN kernels are present. Don't promote it to a
  module-level import.

## Cross-references

- Consumed by: `tools/compile/qnn.py` (the compile-pipeline integration),
  which threads the QNN-embedding flow into `./merlin compile`.
- Consumes: the QNN SDK / QAIRT toolchain (env-vars `QNN_SDK_ROOT`,
  `QNN_BOARD_SYSROOT`, `QNN_CROSS_TOOLCHAIN`).
- Sibling backends: `kernels/spike/runner.py` (RISC-V simulator path).
  No Gemmini / Saturn-OPU kernel-embedding backend exists today; if one
  is added it would live as `kernels/<backend>/` with a similar layout
  (build.py + precompile_extras.py at minimum, optional emit.py).
- Compiler-side: `compiler/src/merlin/Dialect/QNN/` is the in-tree MLIR
  dialect that this package's emitter generates the input for. Schema
  changes there propagate here.

## Update triggers

Re-read this file and update it in the same turn if you edit:

- `kernels/qnn/emit.py` (v1 regex emitter — production path) — refresh
  pitfalls; the v2 history in `tools/archive/qnn_v2/` should not be
  reactivated lightly.
- `kernels/qnn/recognizers/**.py` (add / remove / reorder) — update the
  recognizer count and remind callers that REGISTRY order matters.
- `kernels/qnn/build.py` (toolchain env-vars or qairt-converter calls)
  — cross-ref against the env-var contract listed in Pitfalls.
- `kernels/qnn/precompile_extras.py` (the dispatch hook called from
  `kernels/core/precompile.py`) — confirm the late-import contract.
