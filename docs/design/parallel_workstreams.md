---
title: "Design: subsystem boundaries and the schemas between them"
kind: design
status: current
owner: core
last_verified: 2026-08-23
related: [architecture, kernel_mining, dse, targetgen]
code_refs: [merlin/python/merlin]
---

# Subsystem boundaries and the schemas between them

Merlin is built as three subsystems that advance independently. **They coordinate through schemas,
not prose** — each one's output is a validated artifact the others consume, so a change to what one
produces shows up as a schema change rather than as a broken assumption downstream.

## The three subsystems

| Subsystem | Owns |
| --- | --- |
| **TargetGen** | `merlin/python/merlin/targetgen/`, `merlin/targets/`, `merlin/python/merlin/xdsl_dialects/contract.py` |
| **Kernel mining** | `merlin/python/merlin/kernels/`, `merlin/python/merlin/rvvgen/`, `merlin/experiments/kernel_policy/` (external-tool adapters live in-package — see `docs/design/integrations.md`) |
| **Design-pressure / DSE** | `merlin/python/merlin/design_pressure/`, `merlin/python/merlin/dse/`, `merlin/benchmarks/semantic_memory/` |

## Shared artifacts and who owns them

Every row is a schema in `merlin/schemas/`. The producer is the only subsystem that writes it.

| Artifact | Produced by | Consumed by |
| --- | --- | --- |
| `target_contract` | TargetGen | Kernel mining, DSE |
| `dialect_plan` | TargetGen | TargetGen |
| `kernel_record` | Kernel mining | DSE |
| `abstraction_candidate` | Kernel mining | TargetGen, DSE |
| `policy_rule` | Kernel mining | DSE, and the compiler downstream |
| `design_pressure` | DSE | TargetGen, Kernel mining |
| `interface_candidate` | DSE | TargetGen |
| `dse_result` | DSE | all |
| `exploitability_report` | DSE | all |

## How the data flows

```
Kernel mining:  kernels   -> policy_rules.yaml
                               |
DSE:            workload  -> design_pressure.json + policy_rules -> candidate_contracts.yaml
                               |
TargetGen:      target_contract + candidate_contracts -> dialect_plan -> dialect scaffold
                               |
DSE:            baseline vs software_visible vs hardware_managed vs oracle -> exploitability
```

## Rules

- **Do not invent private file formats.** Add or extend a schema in `merlin/schemas/` first — the
  schema is the interface, and a format that exists only inside one subsystem cannot be consumed.
- **Land schema changes before the code that depends on them**, so a subsystem is never blocked on
  another's in-flight refactor.
- A subsystem may read another's artifacts, but must not reach into its modules. Where that boundary
  is enforced mechanically, `build_tools/scripts/check_structure.py` checks it.
