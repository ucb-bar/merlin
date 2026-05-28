# `samples/` — agent guide

## Mental model

C/C++ runtime sample applications that link against the merlin-extended
IREE runtime. Organized by **deployment target**, plus a `common/`
subtree for shared building blocks.

| Subtree | Role |
|---|---|
| `common/core/`, `common/runtime/`, `common/dispatch/`, `common/xpu-rt/` | Shared helper libraries used by multiple per-target samples. |
| `common/dispatch_scheduler/` | Target-neutral `merlin-dispatch-scheduler` binary (works on QRB5165, SpacemiT X60, …). Aliased from per-target subdirs. |
| `SpacemiTX60/`, `QRB5165/`, `SaturnOPU/`, `Radiance/` | Per-target samples. Each platform owns its build profile and toolchain. |
| `research/` | Sample-flavored experiments that aren't gated on a production board. |

Top-level `CMakeLists.txt` gates each subtree behind a
`MERLIN_BUILD_<TARGET>` switch that the matching build profile sets.

## Pitfalls

- **Per-target subtrees gate on `MERLIN_BUILD_<TARGET>` from
  `tools/build/presets.py`.** Adding a sample for a new board requires
  matching the gate name. Otherwise nothing builds.
- **`common/xpu-rt/` uses the older IREE
  `iree_hal_driver_create_device_by_id` signature.** It still links
  cleanly against the pinned `iree_bar`. If you bump IREE past the
  `iree_hal_device_create_params_t` refactor, expect breakage here
  — see [[iree_api_runner_port]] memory.
- **`common/dispatch_scheduler/` is aliased, not duplicated.** Per-target
  `<platform>/dispatch_scheduler/CMakeLists.txt` files exist to make the
  platform tree advertise the right entry point — they don't fork the
  source.
- **Per-model sample binaries don't belong here.** They go in
  `benchmarks/<target>/` or use the dispatch-scheduler with a
  `schedule.json`.

## Cross-references

- Built by: build profiles in `tools/build/presets.py:PROFILE_PRESETS`
  (spacemit, qrb5165, firesim, …).
- Consumes: the merlin-extended IREE runtime under `runtime/` (see
  `runtime/AGENTS.md`).
- Docs: `docs/how_to/add_sample_application.md` walks the
  add-a-new-sample flow.

## Update triggers

Re-read this file and update it in the same turn if you:

- Add a new per-target sample subtree (`samples/<NewBoard>/`) — extend
  the layout table and check `samples/CMakeLists.txt` gating.
- Move a sample between per-target and `common/` — refresh the table;
  update `samples/<target>/dispatch_scheduler/CMakeLists.txt` alias
  comments if affected.
- Bump the IREE HAL API used by `samples/common/xpu-rt/` — update the
  Pitfalls warning about pre-`iree_hal_device_create_params_t` signature.
