# `compiler/` — agent guide

## Mental model

C++/MLIR compiler code that gets statically linked into `iree-compile`
when the right plugin scope is selected. Two top-level subdirs:

| Path | Role |
|---|---|
| `compiler/src/merlin/Dialect/` | Merlin-owned MLIR dialects (Gemmini, NPU, OPU, QNN, Radiance) — ops, types, transforms, lowering, lit tests. |
| `compiler/src/merlin/Target/` | Target hooks invoked by IREE codegen during `iree-compile` (e.g. CPU encoding interfaces for OPU ukernels). |
| `compiler/plugins/target/<backend>/` | Plugin registration glue: registers the dialect + target hooks with IREE's `iree-plugin=<backend>` mechanism. |

The `iree_compiler_plugin.cmake` at repo root wires these into IREE's
plugin pipeline.

## Pitfalls

- **Plugin registration is order-sensitive.** Each backend's
  `PluginRegistration.cpp` must call `RegisterDialect` and target
  hooks at construction. Missing registration → `iree-compile` prints
  "unknown dialect" for IR you know exists.
- **`compiler/src/merlin/Dialect/<X>/Transforms/Passes.td`** is the
  source of truth for pass names. `Passes.h` and `Passes.cpp` are
  derived. Edit the .td, then regenerate.
- **Lit tests live next to the passes** under
  `<Dialect>/Transforms/tests/`. Run via `./merlin build --profile npu`
  (or matching scope) then `cd build/host-merlin-release && ninja
  check-merlin-<backend>`.
- **Dialect changes propagate to `kernels/` and `tools/compile/`.**
  Renaming an op breaks recognizers in `kernels/qnn/recognizers/`
  and YAML flag composition in `tools/compile/cli.py`. Grep both.

## Cross-references

- Built by: `./merlin build` profiles (`npu`, `gemmini`, `qnn-compiler`,
  `full-plugin`). See `tools/build/presets.py:PROFILE_PRESETS`.
- Consumed by: `iree-compile` when the matching `--iree-plugin=<X>` flag
  is in the per-target YAML's `plugin_flags`. See `models/*.yaml`.
- Companion runtime: `runtime/` mirrors the dialect/plugin split — each
  backend that needs a runtime-side hook lives in
  `runtime/src/iree/hal/drivers/<backend>/`.

## Update triggers

Re-read this file and update it in the same turn if you edit:

- `compiler/src/merlin/Dialect/<X>/Transforms/Passes.td` (new pass /
  renamed) — refresh module map; grep `kernels/` and `tools/compile/`
  for stale references.
- `compiler/plugins/target/<backend>/PluginRegistration.cpp` (new plugin /
  registration order change) — refresh Pitfalls.
- Any op rename in `*.td` — Pitfalls warns about this; grep
  `kernels/qnn/recognizers/` and `models/*.yaml`.
