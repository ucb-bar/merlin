# Per-class selection with merlin.transforms.shape_policy

## Specialize schedules without duplicating lowering

Keep source semantics and target instruction lowering shared. Expose schedule families and pass
orders as alternatives; choose between them using shape, layout, output precision, reuse and
available resources. A legality predicate establishes whether an alternative is safe. A performance
predicate requires comparisons of legal alternatives; fitting in storage does not establish speed:
the capacity guards `shape_choice_table.py` reports say where to look, cycles say what to pick.

Before testing, declare representative classes and reserve neighboring boundary and held-out
shapes. Cover square, tall, wide, thin and ragged cases where applicable. Do not dispatch on capsule
or model names. Keep a correct incumbent for each class; a new global default must not silently
replace good schedules in unrelated classes. Missing timing, a regression or incomplete boundary
coverage retains the incumbent while exploration continues.

Rank comparable whole-call cycle observations with matching hardware/model and measurement scope.
Report roofline headroom, useful compute attainment, traffic, preparation, storage pressure and
resource activity alongside them. Instruction counts and structural estimates help explain a
result; they do not substitute for missing cycle evidence. Never mix physical occupancy with
useful-work/peak attainment, or compare model versions as if they were the same experiment.

Inspect the emitted product of each retained alternative: loop order, actual transfers,
configuration reuse and completion dependencies must reach target emission. Test affected
capabilities against immutable compiler snapshots while other helpers continue authoring. Reuse
unchanged execution evidence by its identities; do not rerun the full corpus for every policy edit.

## From the choice table to a policy

`shape_choice_table.py` is the planning step; `shape_policy` makes the selection.

| table output | shape_policy input |
|---|---|
| `cases[].resources` (`<set>_copies`, `<set>_stores_needed`, `resident_tiles_cover`) | `PolicyFacts(..., resources=tuple(sorted(resources.items())))`, computed in your lowering from the same RTL facts |
| `separating_predicates[].guards.when_true/when_false.predicate` | `ResourcePredicate(**predicate)` in `ShapeScheduleClass.predicates` |
| `missing_boundary_values` | cases to add: `assess_shape_policy` keeps the incumbent until the frozen panel holds each guard's exact boundary value and the adjacent outside value |
| `classes[].trained_on_other_classes` (`ValidationPartition`) | `ShapeScheduleClass.design`: training and held-out geometries |
| the current setting (incumbent) and ratios against it | `ShapeScheduleClass.incumbent`; `assess_shape_policy` trains on the same geometric-mean paired ratio and vetoes any held-out regression (`max_regression_ratio`) |

Alignment (`n % edge == 0`) is a legality predicate of the variant, not an empirical cutoff:
`ShapeScheduleClass` refuses a `ShapePredicate` with `multiple_of`. Put alignment on the
`PipelineVariant` and capacity cutoffs on the class.

## The call

Inventory 22 exposes `merlin.transforms.shape_policy`. It operates on real `PipelineVariant` paths
and `CompletedObservation` feedback; it does not run simulations or infer the numerical status. Use
the selected full-compiler implementation digest, including authored pass and target-adapter
overrides, in addition to the generic library digest:

```python
from merlin.transforms.pipeline_policy import ResourcePredicate
from merlin.transforms.shape_policy import (
    ShapeCaseFeedback, ShapeScheduleClass, PerformanceDomain, assess_shape_policy,
)

rows = tuple(ShapeCaseFeedback(geometry, facts, tuple(pairs), compiler_source_sha256)
             for geometry, facts, pairs in completed_cases)        # facts.resources from the table
group = ShapeScheduleClass(
    "overflows_accumulator", (ResourcePredicate(**guard),), incumbent_name,
    ValidationPartition(training_geometries, heldout_geometries),
)
policy = assess_shape_policy((group,), paths, rows, fallback=original_path_name)
domain = PerformanceDomain(current_provider_sha256, current_facts_sha256,
    current_library_sha256, current_scope_contract_sha256, "modeled_roi",
    current_compiler_source_sha256)
result = policy.execute(current_private_session, registry, current_facts,
                        output=actual_output_analysis_name, domain=domain)
```

Consume `result.artifact` and retain `result.receipt` and `policy.report`; write the report as JSON
and read it per class with `shape_choice_table.py --shape-policy policy.json` (status, selected vs
incumbent, and per alternative what is missing: geometries, boundary cases, complete pairs, a single
performance domain, or a regression). Incomplete support keeps the incumbent. Facts must describe
each observation's original source. Construct the current domain from the actual selected
compiler/runtime, not from old feedback.
