# design-pressure — design-pressure analyzer

Computes the Region Pressure Vector for a workload region and synthesizes the legal I0–I3
contracts. Implemented as the `merlin-design-pressure` console script (Milestone 1).

## Backing module

`merlin.design_pressure.cli` (logic in `merlin.design_pressure.{pressure_vector,synthesize}`).

## Usage

```bash
# synthetic VLA action-chunk decode region
merlin-design-pressure --workload vla_action_chunk_decode --H 16 --reuse 16 --K 256

# an existing benchmark region by name
merlin-design-pressure --workload repeated_rhs_matmul

# an explicit workload_region YAML
merlin-design-pressure --region-yaml path/to/region.yaml
```

Writes `design_pressure.json` + `candidate_contracts.yaml` under `output/dse/<workload>/`
(gitignored). The pressure vector is architecture-independent; the mined `policy_rules.yaml`
are the legality verifiers.
