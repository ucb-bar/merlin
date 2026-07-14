---
title: Parallel workstreams
kind: design
status: current
owner: core
last_verified: 2026-07-14
related: [architecture, kernel_mining, dse, targetgen]
code_refs: [merlin/python/merlin]
---

# Parallel workstreams

Three Claude Code sessions work in parallel. **They coordinate through schemas, not prose.**

## Sessions and ownership

| Session | Branch                          | Owns                                                                 |
| ------- | ------------------------------- | -------------------------------------------------------------------- |
| 1. TargetGen | `feature/targetgen-scaffold`     | `merlin/python/merlin/targetgen/`, `merlin/targets/`, `merlin/python/merlin/xdsl_dialects/contract.py` |
| 2. Kernel mining | `feature/kernel-policy-mining` | `merlin/python/merlin/kernels/` (external-tool adapters live in-package — see `docs/design/integrations.md`), `merlin/experiments/kernel_policy/` |
| 3. Design-pressure / DSE | `feature/design-pressure-dse` | `merlin/python/merlin/design_pressure/`, `merlin/python/merlin/dse/`, `merlin/benchmarks/semantic_memory/` |

## Shared artifacts (ownership)

| Artifact                            | Owner     | Consumers                 |
| ----------------------------------- | --------- | ------------------------- |
| `target_contract`                   | Session 1 | 2, 3                      |
| `dialect_plan`                      | Session 1 | 1                         |
| `kernel_record`                     | Session 2 | 3                         |
| `abstraction_candidate`             | Session 2 | 1, 3                      |
| `policy_rule`                       | Session 2 | 3, later compiler         |
| `design_pressure`                   | Session 3 | 1, 2                      |
| `interface_candidate`               | Session 3 | 1                         |
| `dse_result`                        | Session 3 | all                       |
| `exploitability_report`             | Session 3 | all                       |

## Shared flow

```
Session 2: kernels  -> policy_rules.yaml
                          |
Session 3: workload -> design_pressure.json + policy_rules -> candidate_contracts.yaml
                          |
Session 1: target_contract + candidate_contracts -> dialect_plan -> dialect scaffold
                          |
Session 3: baseline vs software_visible vs hardware_managed vs oracle -> exploitability
```

## Rules

- Do not invent private file formats. Add/extend a schema in `merlin/schemas/` first.
- Do not merge large code until schemas stabilize. Suggested merge order: foundation, Session 3
  schemas, Session 2 schemas, Session 1 scaffold.
- Use `tmp/help/` for local cross-session notes (gitignored).
