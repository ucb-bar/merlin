# `tools/perf/` — agent guide

## Mental model

`tools/perf/cli.py` is the registered shim for `./merlin perf-decompose`.
The shim is a thin wrapper; the parsing + rendering + on-board profiling
live here. Two flavors of analysis (decode existing logs, or measure
fresh on a board), both model-agnostic.

For the module-by-module extension-points map, read `__init__.py`.

## Pitfalls

- **Colors and overlays**: `plot_planned_vs_observed.py:build_job_colors`
  accepts a `color_map_overrides` arg — use it for cross-plot consistency
  rather than re-introducing per-model conditionals.
- **SSH/QNN defaults** come from `$MERLIN_BOARD_HOST`,
  `$MERLIN_BOARD_SSH_KEY`, `$MERLIN_BOARD_BENCH_BIN`, `$MERLIN_QNN_LIB_DIR`.
- **REPO_ROOT** is `parents[2]` from here.

## Cross-references

- Inputs: uartlogs/trace CSVs from `tools/run/`, matrix.json from
  `tools/compile/dispatch_matrix.py`.
- Outputs: profiled manifests feed `tools/schedule/` and XPU-RT's
  `merlin_adapter.py`.
- MCP wrapper: `tools/mcp_servers/perf.py` exposes `perf_decompose` with parsed
  top-K hot dispatches + bucket-by-kind totals.

## Update triggers

Re-read this file and update it in the same turn if you edit:

- `tools/perf/cli.py` (new flag / new CSV column) — refresh module map;
  touch `tools/mcp_servers/perf.py` if the CSV schema changed.
- `tools/perf/decompose.py` (output schema) — cross-ref against
  `tools/mcp_servers/perf.py` parser and any consuming notebooks under `tmp/`.
- `tools/perf/profile_*.py` (a new profiler) — refresh per-module table.
