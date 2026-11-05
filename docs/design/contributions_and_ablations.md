---
title: "Contributions, abstractions and the ablations that test them"
kind: design
status: draft
owner: core
last_verified: 2026-10-17
related: [macro_scheduling, rtl_derived_compiler_tooling, performance_levers_per_archetype, merlin_supplied_host_placement, output_epilogue_capabilities]
code_refs:
  - src/merlin/xdsl_dialects
  - src/merlin/targetgen/rtl
  - src/merlin/xdsl_dialects/lowering/compute_groups.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/form_perf.py
---

# Contributions, abstractions and the ablations that test them

Phases 0/1/2 are the procedure. This note states what the procedure is evidence *for*: the abstractions
Merlin introduces so a working, performant and portable compiler for a new accelerator can be built,
by an agent or a person, from the accelerator's RTL. Every claim names the experiment that supports it.
Results are recorded under `out/artifacts/experiments/<target>/` and carry the claim ID.

## Core claim

Separating *facts*, *decisions*, *hardware/software abstractions* and *execution* into verifiable IR
layers, all derived from the RTL, makes compiler construction for a new accelerator converge faster and
reach competitive performance, with only a small, target-specific backend seam to write.

## Abstractions

| ID | Abstraction | Where it lives |
|---|---|---|
| A1 | RTL-derived hardware facts with semantic roles (census, datapaths, memories, timing structure, epilogue capabilities) | `src/merlin/targetgen/rtl/` |
| A2 | `contract` dialect and capability contract: what is true, legal and must be proven; decides placement | `xdsl_dialects` (`contract`), target contracts |
| A3 | `schedule` dialect: explicit, verifiable compiler decisions (tiling, layout, placement, grouping, memory state) | `xdsl_dialects` (`schedule`) |
| A4 | `interface` dialect: target-independent HW/SW abstractions (resident tensors, accumulators, async copy and events, command objects) | `xdsl_dialects` (`interface`) |
| A5 | `runtime` dialect: generic execution and measurement | `xdsl_dialects` (`runtime`) |
| A6 | Compute groups and the frozen accelerator/host split; Merlin supplies the host side | `lowering/compute_groups.py`, `merlin_supplied_host_placement` |
| A7 | Capsules with tiered checks (L0, L1, Spike L2, RTL L3), "not run is not a pass", change-aware re-runs | `phase1/feedback`, capsule runner |
| A8 | A cost model whose structure is derived from RTL and whose costs are fitted on RTL; a per-capsule roofline | `docs/guides/spike_timing.md` |
| A9 | Performance stated as scaling claims (fits over sweeps), not single points | `phase0/form_perf.py`, `perf_pk_claim` |
| A10 | The generated backend seam: only the target dialect, its lowering passes and its encoding are per target | `rtl_derived_compiler_tooling` |

## Claims and experiments

Each experiment runs on one target with a fixed budget; arms differ only in the named abstraction.

| Claim | Abstraction | Experiment (arms) | Metric |
|---|---|---|---|
| C1 Layering speeds convergence | A2–A5 | backend written straight from linalg vs the full layered stack | Phase 1 time to convergence; capsules passed per hour |
| C2 Facts as IR beat facts as text | A1, A2 | facts as documentation vs `contract` IR checked by passes | legality defects caught statically vs at RTL; time to convergence |
| C3 A frozen host split removes a failure class | A6 | author writes the host side vs Merlin supplies it | convergence time; failure classes |
| C4 Fine, tiered feedback speeds iteration | A7 | whole-model feedback only vs capsules with tiered checks and change-aware re-runs | iterations per hour; author blocked share |
| C5 Explicit decisions make search safe and effective | A3 | search over `schedule` decisions vs free-form kernel edits | performance at equal budget; correctness regressions per change |
| C6 Interface abstractions expose the real levers | A4 | with vs without `resident_pack` / `async_copy` / events | residency and overlap found; speedup above capacity |
| C7 Scaling claims teach rules, not schedules | A9 | optimize capsule sweeps vs tune single points | performance on held-out models |
| C8 A derived, calibrated proxy is faithful enough | A8 | no proxy vs untuned vs calibrated Spike | rank correlation with RTL; performance per wall-hour |
| C9 The approach ports | A1, A10 | bring up a second target from its RTL plus a descriptor | target-specific lines written; convergence time vs the first target |

## Rules for the evidence

- Final performance claims are measured on RTL or FPGA runs of held-out models, never on the proxy.
- An ablation arm changes one abstraction; everything else (budget, model, prompts, corpus) is fixed.
- Held-out models and reference backends never reach an author-visible input; see
  `agentic_experiment_integrity`.
