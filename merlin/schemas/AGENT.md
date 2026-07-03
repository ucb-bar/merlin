# AGENT.md — merlin/schemas

## Purpose

The cross-workstream **coordination contract**. All artifacts exchanged between the three Claude Code sessions (TargetGen, kernel mining, design-pressure/DSE) are defined here as YAML schemas. The sessions coordinate through these schemas, not prose.

## What belongs here

- `*.schema.yaml` files, each with: title, purpose, required top-level fields, example.
- The 10 core schemas: target_contract, dialect_plan, kernel_record, abstraction_candidate, policy_rule, workload_region, design_pressure, interface_candidate, dse_result, exploitability_report.

## What does not belong here

- Instances/data (those go under `artifacts/`, `merlin/benchmarks/`, or target dirs).
- Tool or analysis code.
- Undocumented JSON/YAML blobs that tools secretly depend on.

## Interfaces

Produced/owned per the ownership table in `docs/parallel_workstreams.md`. Consumed by tools under `tools/` and Python modules under `merlin/python/merlin/`.

## Invariants

- **Any cross-workstream artifact MUST have a schema here before a tool depends on it.**
- No undocumented JSON blobs anywhere in the repo.
- Schema changes are coordinated — they affect multiple sessions. Note owner before editing.
- Schemas stay non-empty and include an example block.

## Testing expectations

`python build_tools/scripts/check_structure.py` checks presence and non-emptiness; each schema YAML must parse with `yaml.safe_load`.

## Notes for future agents

These need not be formal JSON Schema yet, but should be structured enough that future code can validate against them. Promote to strict JSON Schema once a format stabilizes.
