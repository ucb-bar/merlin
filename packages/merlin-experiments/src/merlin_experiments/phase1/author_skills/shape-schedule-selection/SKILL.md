---
name: shape-schedule-selection
description: "Turn per-shape schedule choices (block policy, storage windows, retention, pipeline kind, transfer coalescing) into guarded per-class selections instead of hand-written if-thresholds and variant A/B directories judged on a few shapes: declare the alternatives as tuning knobs, freeze shape classes (square, long-M, long-N, M=1, ragged, model layer) as ExplorationGeometry/ValidationPartition hold-outs, find the capacity cutoff that separates the winners with scripts/shape_choice_table.py (ResourcePredicate guards over RTL-fact store fits), then let merlin.transforms.shape_policy keep each class incumbent until paired cycle feedback covers that cutoff. scripts/capacity_pipeline_example.py builds one dense kernel from RTL-fact capacities through the schedule IR, loop transforms, panel pipeline, array retention and readout overlap. Use when a lowering picks a schedule with an if on M, N or K, when a default was tuned on few shapes, or before changing a default schedule."
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
   panel still lacks. Never cut at the M or N where measurements changed; if nothing separates, keep
   the per-class choice, add boundary cases or derive another fact.
6. **Select with shape_policy.** Put the table's resources in `PolicyFacts`, the guard in a
   `ShapeScheduleClass` with a frozen design and its incumbent, and paired `CompletedObservation`
   feedback in `ShapeCaseFeedback`; `assess_shape_policy` switches a class only when the panel covers
   the cutoff with no regression. Read its report with `shape_choice_table.py --shape-policy FILE`.
   Workflow and execute call: `reference/shape-policy.md`. A runnable version over registered
   library PipelineVariants fed by `--panel` results: `library_pipeline_policy.py` in the
   `heavy-pass-recipes` skill.
7. **Confirm, then keep the alternatives.** Change a default only after `ab-library-pass` on every
   affected class; keep knobs declared so idle timing keeps covering other values.

Library: `merlin.transforms.pass_exploration` (`ExplorationGeometry`, `ValidationPartition`,
`explore_candidates`, `rank_observed_metrics`), `merlin.transforms.shape_policy`,
`merlin.transforms.pipeline_policy.evaluate_variants`, `merlin.compile.scheduling.schedule_ir`,
`loop_schedule`, `dense_panel_pipeline`, `dense_array_retention`, `dense_readout_overlap`,
`merlin.compile.capacity`. Contracts: the `scheduling-api` skill.
