# AGENT.md — artifacts/measurements

## Purpose

Hardware-measurement campaigns, keyed **substrate → model → experiment**:
`measurements/<substrate>/<model>/<experiment>_v<ver>_<TS>_<sha>/` + manifest.yaml.

`<substrate>` = the execution environment that produced the numbers, named `<kind>_<design>` so
identical kernels on different bitstreams/designs never collide:
`k1_spacemit`, `firesim_<bitstream>`, `baremetal_<verilator-design>`, `zephyr_<design>`, `spike_<config>`.
Producers: `scripts/k1_*.py` and the firesim/zephyr/baremetal sweeps.

## Invariants

- Contents are gitignored; only AGENT.md / README.md / .gitkeep are tracked.
- Created via `merlin.common.artifacts.new_measurement(substrate, model, experiment, version=...)`.
- Keep inner file names identical across substrates/models so cross-substrate diffs are trivial.
