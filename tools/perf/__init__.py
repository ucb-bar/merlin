"""Implementation package for the `perf-decompose` subcommand.

The registered shim is `tools/perf/cli.py`; this package owns parsing,
rendering, and on-board profiling.

Two flavors of analysis:
- **Decode** — given a uartlog or trace, produce per-dispatch timing tables
  and Gantt charts.
- **Measure** — drive a per-dispatch / per-island benchmark on a remote
  board and emit a structured profiled manifest.

Extension points:

- `decompose.py` — parses `[dc]`/`CYC`/`[dn]` lines from FireSim uartlogs.
  Extend for new uartlog formats.
- `profile_dispatch_matrix.py` — on-board per-(dispatch, target) profiler
  via SSH + merlin-dispatch-bench.
- `profile_per_island.py` — Phase-4 per-(island, target) profiler.
- `profile_per_target_qnn.py` — per-dispatch QNN profiler across HTP/HTA/GPU.
- `plot_planned_vs_observed.py` — Gantt from scheduler trace + schedule JSON;
  auto-discovers jobs and targets. `build_job_colors` accepts
  `color_map_overrides` for cross-plot consistency.
- `plot_yolov8_real_island_profiles.py` — QNN island measurement plots
  (CPU/GPU/HTA). Model-agnostic despite the filename.
- `trace_to_profile.py` — fold observed run_us back into a cost matrix.
- `profile_dispatches.sh` / `parse_dispatch_profile.sh` — shell wrappers
  for batch sweeps.

SSH defaults come from `$MERLIN_BOARD_HOST` / `$MERLIN_BOARD_SSH_KEY` /
`$MERLIN_BOARD_BENCH_BIN` / `$MERLIN_QNN_LIB_DIR`.
"""
