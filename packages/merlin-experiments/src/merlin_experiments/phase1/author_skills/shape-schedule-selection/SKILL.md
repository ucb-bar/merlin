---
name: shape-schedule-selection
description: "Organize shared schedule alternatives for dense, convolution and attention into a few geometry classes with layout, tail and residency modifiers. Use the existing shape/resource predicates and paired-cycle selector; keep lowering shared, derive storage legality from hardware facts, and validate empirical cutoffs on representatives and neighbors. No per-capsule lookup, new selector, startup profiling or simulation during ordinary compilation. The capacity_pipeline_example.py script exercises dense schedule IR, loop transformations, pipelines, retention and readout. Use before changing a global schedule default or when shapes need different reusable plans."
---

`SKILL_DIR` = the directory holding this file. Run from the workspace root.

1. **Share the upstream, specialize the plan.** Keep source semantics and instruction lowering
   shared: one `DenseProblem`, storage and resource view derived from capacity facts, then a per-shape
   plan (block, pipeline kind, retained operand). Run the example; read `reference/capacity-pipeline.md`:
   ```
   python SKILL_DIR/scripts/capacity_pipeline_example.py              # rtl_facts/facts.json if present
   python SKILL_DIR/scripts/capacity_pipeline_example.py --shape M,K,N --pipeline segmented
   ```
   It derives stores with `merlin.targetgen.address_space`, lets `dense_blocks` offer only blocks that
   fit, and checks compact, retained (loop transforms, panel pipeline, array retention) and segmented
   (readout overlap) plans against the source product.
2. **Name each choice as a knob or registered path.** Each per-shape decision becomes a knob in
   `submission/TUNING_KNOBS.json` or a `PipelineVariant`; today's behavior is the incumbent.
3. **Freeze cases and classes before measuring.** `cases.json` holds every case you judge by, on both
   sides of each capacity boundary you suspect, plus model layers; no capsule or model names select
   anything. The script holds each structural class out in turn (`ValidationPartition`).
4. **Collect comparable results.** Screen emitted programs with the `explore-schedules` skill
   (`explore_candidates`, `program_sha256`), then time each value on the same cases (`ab-library-pass`
   step 4, `tuning_knobs set`, `qa/idle_tuning.json`). Formats: `reference/choice-table.md`.
5. **Find the cutoff.**
   ```
   python SKILL_DIR/scripts/shape_choice_table.py --knobs submission/TUNING_KNOBS.json \
       --cases cases.json --results results.json [--facts rtl_facts/facts.json] [--knob NAME]
   ```
   Per class: best candidate by paired ratio to the incumbent, and whether the winner of the other
   classes holds or regresses there. Per case: store fits and `PolicyFacts.resources` values. Last:
   each fit that decides a knob's winner, as two `ResourcePredicate` guards and the boundary cases the
   panel still lacks. Performance cutoffs may use dimensions, aspect ratios or tile features;
   explain the crossover and check neighbors rather than keying a rule to a measured case. If
   nothing separates, retain the incumbent or derive another useful feature.
6. **Select with shape_policy.** Put the table's resources in `PolicyFacts`, the guard in a
   `ShapeScheduleClass` with a frozen design and its incumbent, and paired `CompletedObservation`
   feedback in `ShapeCaseFeedback`; `assess_shape_policy` switches a class only when the panel covers
   the cutoff with no regression. Read its report with `shape_choice_table.py --shape-policy FILE`.
   Workflow and execute call: `reference/shape-policy.md`. A runnable version over registered
   library PipelineVariants fed by `--panel` results: `library_pipeline_policy.py` in the
   `heavy-pass-recipes` skill.
7. **Confirm, then keep the alternatives.** Change a default only after `ab-library-pass` on every
   affected class; keep knobs declared so idle timing keeps covering other values.

### Construction and feedback rules

- Preserve source views, storage effects, configuration state and completion edges in the common
  schedule representation before terminal instruction emission. Keep one target encoder across
  alternatives; a new schedule should not require rewriting that encoder.
- Explore residency as a spectrum: complete operands, capacity-bounded phases, then streaming.
  Derive phase footprints from simultaneously live storage and the declared transfer limits.
  Capacity establishes fit; source independence, observers and completion establish legality.
- Put completion at proven ownership/reuse boundaries. Disjoint groups may share a phase only
  after their actual storage assignments and every original operation are checked together.
  Independent source operations do not automatically have independent physical buffers.
- Check the emitted application receipt before interpreting a trial: selected family, storage,
  loop/order changes and completion policy must match the requested alternative. An enabled flag
  or an imported pass does not establish application. Keep unexpected-route trials as diagnostics.
- Use same-window source-service overlap, traffic amplification, padding and capacity diagnostics
  to choose experiments. Low useful compute attainment alone does not identify the cause; command
  reductions or moved waits do not establish a cycle improvement.
- Try interacting transformations in both legal orders. Unrolling, retention and wider movement
  can change each other's input form; retain refusal reasons and a working incumbent. Prefer
  bounded counted loops when expansion makes source checking or compilation expensive.
- Promote understandable resource-relative predicates from paired shape panels and neighbors.
  Rank UNKNOWN by `fast_estimate`; explore source-legal alternatives. For connected workloads,
  include invocation frequency, preparation lifetime and complete-region latency before promoting
  a kernel improvement as a model improvement.

Library: `merlin.transforms.pass_exploration` (`ExplorationGeometry`, `ValidationPartition`,
`explore_candidates`, `rank_observed_metrics`), `merlin.transforms.shape_policy`,
`merlin.transforms.pipeline_policy.evaluate_variants`, `merlin.compile.scheduling.schedule_ir`,
`loop_schedule`, `dense_panel_pipeline`, `dense_array_retention`, `dense_readout_overlap`,
`merlin.compile.capacity`. Contracts: the `scheduling-api` skill.
