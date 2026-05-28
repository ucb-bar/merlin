# `tools/compile/` — agent guide

## Mental model

`tools/compile/cli.py` is the registered shim for `./merlin compile`. Its
job: take one `.mlir`/`.onnx` plus a target name and produce VMFB +
artifacts under `build/compiled_models/<model>/<target>/`. Heavy lifting
(per-dispatch breakdown, per-chunk emission, QNN-specific per-chunk
compiler) lives here. Model- and target-agnostic by design.

For the module-by-module extension-points map, read `__init__.py`.

## Pitfalls

- **Toolchain paths**: `qnn.py` uses `QNN_SDK_ROOT`, `QNN_BOARD_SYSROOT`,
  `QNN_CROSS_TOOLCHAIN` env vars. Do not hardcode absolute paths
  (no-overfit rule in `AGENTS.md`).
- **Sibling imports**: use package imports
  (`from compile.breakdown_vmfb import …`), not `sys.path` hacks.
- **REPO_ROOT** is `parents[2]` from inside this package, or
  `import utils; utils.REPO_ROOT`.
- **`feedback_overlay`** is at `tools/` top level (not in this package).
  `qnn.py` imports it via a deliberate `sys.path.insert(parents[1])`.

## Cross-references

- Consumers of this package's manifest schema:
  `tools/run/` (consumes per-dispatch VMFBs),
  `tools/perf/` (parses uartlogs from running these VMFBs).
- This package consumes the kernel-embedding precompile pipeline at
  `kernels/` (called by `qnn.py`).
- MCP wrapper: `tools/mcp_servers/compile.py` exposes `compile_model` /
  `compile_list_targets` / `compile_list_models`.

## Update triggers

Re-read this file and update it in the same turn if you edit:

- `tools/compile/cli.py` (new flag / new YAML key consumed / changed
  build-dir auto-selection) — refresh module map and Pitfalls; touch
  `tools/mcp_servers/compile.py` and `models/AGENTS.md` if YAML schema changed.
- `tools/compile/{breakdown_vmfb,chunk_*,dispatch_matrix,qnn,radiance,feedback_overlay}.py` —
  refresh the per-module table.
- `tools/compile/postprocess.py` (vmfb naming / install behavior) —
  cross-ref against `samples/` consumers.
