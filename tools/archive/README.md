# `tools/archive/` — frozen tools from the 2026-05-25 cleanup

This directory holds tools that were relocated out of `tools/` during the
2026-05-25 de-explosion pass. Nothing was deleted — these scripts are
preserved here so any data, plots, or outputs they produced remain
reproducible.

## Layout

- `qnn_islands/` — HTA conv-island export and profiling utilities (5 files).
  Pipeline-internal helpers used during the yolov8 × QNN-HTA bring-up
  campaign; no longer in active use.
- `compile_internals/` — Lower-level compile-pipeline utilities (5 files):
  constant-arena extraction, dispatch-constants materialization, HAL
  binding-sources generation, CPU-flow capture prep, dispatch call-graph
  extraction. Mostly one-shot debugging helpers from compile-flow refactors.
- `gemmini_bug_a/` — Debug fixtures from the Gemmini × dronet × FireSim
  Bug-A investigation (`diff_dronet_dispatches.py`, `diff_gemmini_intr.py`,
  `probe_dronet_matmuls.sh`). The full investigation evidence is at
  `tmp/archive/investigations/gemmini_dronet_bug_a_2026_05/`.
- `quant_debug/` — Quantization sanity-check helpers (`simulate_requantize.py`).
- `part_c/` — Finished Part-C / C8 workstream artifacts (`aot_vs_runtime.py`).
- `qnn_e2e_demo/` — QNN end-to-end demo orchestrators from `tools/kernels/`
  that were never imported by `compile.py`: `qnn_e2e_compile_all.py`,
  `qnn_e2e_bench.py`, `qnn_e2e_inspect.py`, `qnn_e2e_demo.sh`. The
  production kernel-embedding pipeline (`manifest.py`, `precompile.py`,
  `qnn_emit.py`, `qnn_emit_v2.py`, `spec_gen.py`, recognizers, etc.)
  remains in `tools/kernels/`.
- `yaml_duplicates/` — `models/*.yaml` siblings collapsed by the cleanup
  (currently: `firesim_shuttle_gemmini_os.yaml`, which was functionally
  identical to `firesim_shuttle_gemmini.yaml`).

## Why these were archived (not deleted)

None of the relocated files are imported by other tools or referenced from
samples/tests/benchmarks/scripts (verified by `grep` at archive time). They
are also not duplicates of anything load-bearing. But the preservation rule
says any captured plots/CSV/JSON/logs the scripts may have produced should
remain accessible alongside the scripts themselves — so they live here
rather than at `/dev/null`.

If a future workstream revives any of these, lift them back out of
`tools/archive/<topic>/` and either:
1. Wire as a flag on an existing `./merlin` subcommand, OR
2. Promote to a new subcommand registered in `tools/merlin.py:COMMANDS`.

Do **not** drop new top-level `tools/*.py` scripts going forward — see
the "Where new content goes" section of `CLAUDE.md`.
