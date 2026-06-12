# dse — interface-level design-space exploration

Compares baseline / software_visible / hardware_managed / oracle variants for the residency
and accumulator-commit features using the measurable analytical cost model, and runs the
phase-transition experiment over the VLA action-chunk decode region. Implemented as the
`merlin-dse` console script (Milestone 1).

## Backing module

`merlin.dse.cli` (logic in `merlin.dse.{cost_model,variants,harness,exploitability,experiment}`).

## Usage

```bash
# single-point dse_result artifacts only
merlin-dse --workload vla_action_chunk_decode --no-experiment

# full sweep + headline phase_transition.csv (+ .png if matplotlib) + exploitability reports
merlin-dse --workload vla_action_chunk_decode
```

Writes `dse_result.yaml` (per feature), `exploitability_<feature>.yaml`, and
`phase_transition.csv` under `output/dse/<workload>/` (gitignored). The headline result: the
best interface (I0→I2→I3) changes category as weight reuse grows.
