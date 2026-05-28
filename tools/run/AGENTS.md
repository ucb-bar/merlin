# `tools/run/` — agent guide

## Mental model

`tools/run/cli.py` dispatches `./merlin run <mode> [args...]` to one of
the per-mode scripts in this package. Each mode is a self-contained
driver with its own argparse; the shim just routes. Every mode is
model-agnostic and board-agnostic — topology comes from CLI flags or env
vars, never hardcoded.

For the mode-by-mode extension-points map, read `__init__.py`.

## Pitfalls

- **XPU-RT location**: default is sibling of merlin checkout. Override
  via `$MERLIN_XPU_RT_ROOT` — don't hardcode.
- **SSH defaults**: `qdev` is just a default. Pass `--ssh-host
  <user>@<addr>` or set `$MERLIN_BOARD_HOST` for other boards.
- **REPO_ROOT** is `parents[2]` from here.
- **Mode discovery**: `tools/run/cli.py:_MODE_TO_SCRIPT` is the surface.
  A new file in this package that isn't in that dict is unreachable via
  `./merlin run`.

## Cross-references

- Inputs: VMFBs from `tools/compile/`, schedule JSON from XPU-RT's
  `merlin_adapter.py`, breakdowns from `tools/compile/breakdown_vmfb.py`.
- Outputs: uartlogs/traces consumed by `tools/perf/`, hashes consumed by
  `tools/verify/`.
- MCP wrapper: `tools/mcp_servers/run.py` exposes `execute_run` / `run_list_modes` /
  `run_help` with parsed makespan + per-instance latency returns.

## Update triggers

Re-read this file and update it in the same turn if you edit:

- `tools/run/cli.py` (new mode / changed dispatcher) — refresh module map.
- `tools/run/{full_loop,het_e2e,het_matrix,multi_device,roundtrip,schedule}.py`
  (new run-mode behavior) — refresh per-module table.
- `tools/run/sched_*` (the schedule helpers) — cross-ref against
  `tools/mcp_servers/run.py:_parse_run_output`.
