# shape_policy over library pipelines (`library_pipeline_policy.py`)

## What it replaces

```python
# before: a hand-written cutoff, tuned on a few shapes
if m * n > SOME_ROWS * edge and k >= 2 * edge:
    kernel = segmented(...)
elif k <= SOME_K:
    kernel = retained(...)
else:
    kernel = compact(...)
```

Each branch is a library pipeline; the cutoffs are guesses. Replace them with:

1. **Registered passes** over one source IR (`RegisteredPass(PassSpec(...), action)`), sharing an
   upstream analysis (`dense_capacity_view`) and ending in the same output (`dense_program`). Every
   family is realized by the one checked entry `dense_ready_readout.schedule_dense_pipeline` (no plan:
   compact; a typed `schedule_ir` plan: retained full-K or K-segmented with a readout mode).
2. **Variants** = ordered pass paths, each with its LEGALITY predicate:

   | variant | passes | legality (`ResourcePredicate`, facts from capacities) |
   |---|---|---|
   | `compact` | `dense_capacity_view`, `compact_dense` | none (unconditional fallback) |
   | `retained` | ..., `retained_panel_pipeline`, `dense_array_retention` | `full_k_panel_blocks >= 1` |
   | `segmented` | ..., `segmented_block`, `segmented_readout_overlap` | `segmented_blocks >= 1` |

3. **Classes** = PERFORMANCE guards, each a `ShapeScheduleClass(name, predicates, incumbent,
   ValidationPartition(training, held_out))`. The recipe uses accumulator fit
   (`accumulator_output_copies >= 1`, `accumulator_output_stores_needed >= 2`), the resources
   `shape_choice_table.py` reports. The incumbent is today's behavior.
4. **Evidence** = paired panel cycles per geometry (`ShapeCaseFeedback`), from one
   `agent_selfcheck.py --panel dense` response per variant.
5. **Selection** = `assess_shape_policy(classes, variants, cases, fallback="compact")`, then
   `policy.execute(session, registry, facts, output="dense_program", domain=current_domain)` per
   shape. The selected path's passes run on the current source; the result is the emitted kernel.

`PolicyFacts.resources` are derived per shape from the same RTL-fact capacities the passes use
(the recipe's `facts_for`), so a new shape is classified by facts, never by a shape list.

## Feeding panel results

Run the panel once per variant with that variant active (a tuning knob or a build-time switch),
save each response, and pass them:

```
python SKILL_DIR/scripts/library_pipeline_policy.py --panel compact=qa/panel_compact.json \
    --panel retained=qa/panel_retained.json --panel segmented=qa/panel_segmented.json \
    --pins pins.json --holdout model_layer,thin
```

Each panel class's `shape_policy` record has `geometry` (`ExplorationGeometry` fields), `identity`
(basis, facts and scope pins the response states), `author_supplies` (pins only your pipeline
session knows) and `observation` (`CompletedObservation` fields). `pins.json` supplies
`abi_sha256`, `inputs_sha256`, `reference_sha256`, `library_sha256`, `provider_sha256`,
`roi_sha256`, `artifact_format` and `implementation_sha256` (your full compiler source digest),
plus `facts_sha256`/`scope_sha256`/`basis` if the response lacks them. The script derives
`source_sha256` and `context_sha256` from its own session for that shape; observations of one
geometry must share one identity to be paired. Only cycle evidence has `timing_status: complete`; a
Spike count or a provisional estimate is `unknown` and never ranks.

## Reading the report

Per class: `status` (`empirically_supported_declared_panel` or `incumbent_retained`), `incumbent`,
`selected`, and per alternative a `reading`:

| reading | meaning | what to do |
|---|---|---|
| supported on the declared panel | complete pairs on every design case, no matched regression | it can win if its training ratio is lowest |
| missing/UNKNOWN/failed complete paired cycles | some design case lacks a complete pair for it or the incumbent, or it is not legal there | time that case, or narrow the panel to cases where every alternative is legal |
| missing boundary cases (value/outside) | the panel lacks the guard's exact boundary value or the adjacent outside value | add a case at each (e.g. an output that exactly fills the accumulator, and one that needs one more store) |
| missing panel geometries | a design geometry has no feedback | rerun the panel or drop the geometry from the design |
| regresses on a matched case | its worst matched ratio is above `max_regression_ratio` | keep the incumbent; the alternative stays registered |
| more than one performance domain | observations from different providers/facts/scopes | rerun under one domain |

**Missing evidence keeps the incumbent.** A class switches only when every design case has complete
pairs, both guard boundaries are covered, at least one training and one held-out geometry match it,
and no matched case regresses. Held-out geometries can veto, never tune: training chooses by the
geometric-mean paired ratio.

## Execution, legality and stale domains

`AssessedShapePolicy.execute` tries the class's selected variant (class guard AND variant
legality), then the class incumbent, then the fallback, and runs the first applicable path's real
passes. The recipe executes every panel shape plus two unseen ones: one overflowing shape the class
guard sends to the selected pipeline, and one where that pipeline's legality fails (a single K tile:
no segment to pipeline) so the incumbent compiles instead. `domain` must equal the performance
domain the evidence was measured in (`PerformanceDomain.from_observation`); after you change the
compiler, the old selection is not used (incumbent) until the panel is rerun.

The demo (no `--panel`) builds a SYNTHETIC panel in the record form above so both outcomes show:
the overflow class switches to `segmented`; the fit class keeps `compact` because `retained` lacks a
complete pair on one held-out case. Its cycles are a toy table, never evidence.
