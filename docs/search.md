# Search policy

Deliberately small. Three methods only:

> Grid search for small explicit spaces; evolutionary search for improving a candidate;
> Quality-Diversity / MAP-Elites when we need many different good families instead of one winner.

Do **not** add beam search, MCTS, Bayesian optimization, or a generic "search everything"
framework. There is **no `merlin.search` dialect** — search is orchestration/experiment logic.

| Method | Use for | Not for |
| ------ | ------- | ------- |
| **Grid** | small finite sweeps: tile sizes, resident capacity, queue depth, reuse threshold, pass-order | large open-ended design |
| **Evolutionary** | improving candidate policies / schedules / dialect plans / lowering variants / runtime contracts | one-shot deterministic choices |
| **MAP-Elites** | keeping multiple high-quality families (scratchpad / resident-object / command / hardware-managed style) | local parameter tuning |

## Candidate types

`compilation_strategy`, `policy_rule`, `dialect_plan`, `interface_candidate`, schedule/runtime
contracts — anything that can be mutated and evaluated. Search spaces are `search_space.schema.yaml`.

## Shared scoring (all methods)

```
score = correctness + compile_success + verifier_success + workload_coverage
      + compiler_exploitability + speedup_or_cost_improvement - complexity_penalty
priority: correctness > compile_success > coverage > exploitability > speedup
```

Prioritizing correctness/compile/coverage stops search from finding fast-but-invalid junk. LLMs
are **mutation/repair operators**, never the search method itself.

## MAP-Elites archive

A dict keyed by behavior descriptors is enough to start:

```python
archive[(memory_abstraction, control_abstraction, granularity, workload_regime)] = best_candidate
```

This returns a portfolio, so Merlin does not prematurely converge on one abstraction style.

## Phases

1. **Grid** first (deterministic, immediately useful) — e.g. the resident-packed-tensor regime map
   over `reuse_count x resident_capacity x pack_cost x dispatch_cost` -> `regime_map.csv`.
2. **Evolutionary** over YAML artifacts (`policy_rule`, `interface_candidate`, `dialect_plan`),
   keeping candidate lineage.
3. **MAP-Elites** archive once evolutionary search works.

## Per-session use

- **Session 1 (TargetGen):** grid over op/type variants; evolutionary over `dialect_plan` + verifier
  conditions; MAP-Elites over dialect families (micro-op / command / resident-object / hw-managed).
- **Session 2 (Kernel mining):** grid over policy thresholds; evolutionary over `policy_rule`;
  MAP-Elites over optimization families (packing / vectorization / accumulator / dispatch-grouping).
- **Session 3 (DSE):** grid over HW/interface parameter sweeps; evolutionary over
  `interface_candidate`; MAP-Elites over HW/SW contract families.

## Modules

`merlin/python/merlin/search/{candidate,evaluator,archive,grid,evolutionary,map_elites,mutations,reports}.py`.
Scoring delegates to `merlin/python/merlin/dse/harness.py`. See `docs/compilation_strategies.md`.
