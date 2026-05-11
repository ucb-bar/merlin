# samples/

End-to-end runtime examples and target-facing app flows. Built via
`./merlin build --profile <target> --cmake-target <target_name>`; binaries
land under `build/<target>-merlin-<config>/`.

- `SaturnOPU/` — Saturn OPU custom-dispatch ukernel samples and FireSim flows.
- `SpacemiTX60/` — SpacemiT X60 (Banana Pi) baseline and dispatch-scheduler
  samples used in the dev blogs.
- `common/` — generic utilities, IREE runtime helpers, and the dispatch
  scheduling library shared across platform samples.
- `research/` — exploratory flows (mlir pipelining, model graph generation,
  promise-devices). Not part of the supported build surface.

For step-by-step "add a sample" guidance see
[`docs/how_to/add_sample_application.md`](../docs/how_to/add_sample_application.md).

Subdir naming uses PascalCase for the target board (`SaturnOPU`,
`SpacemiTX60`); a future cleanup may snake_case these — keep that in mind
when adding new ones (no new PascalCase folders, please).
