---
name: compose-pass-pipeline
description: "Open before adding, moving, reordering or replacing a compiler pass or transformation, when a pass fails with 'lacks prerequisites', 'invalid explicit pass ordering', 'missing or stale analysis' or 'different IR level', or to A/B two pass orders. Gives the phase order the library expects and how PassSpec requires/produces/invalidates encode it. Covers inserting a pass safely, keeping an order generic and debugging an illegal one, with a script that lists every declared pass contract and checks an order."
---

# Compose passes in a legal, generic order

Merlin has no global pass manager. Each IR level runs small pipelines of
`merlin.transforms.pipeline_state.PassSpec(name, stage, requires, produces, invalidates, mutates_ir, ir_level)`.
`merlin.transforms.pipeline_state.validate_pass_order` rejects a bad order before anything runs, and
`merlin.transforms.pipeline_state.PipelineSession` refuses stale analyses. The full phase order,
with citations and an error-to-fix table, is in `reference/phase-order.md`. For calling one pass,
see the `scheduling-api` skill; for fusion, see the `fusion` skill.

## 1. See the pipeline you are changing
`python scripts/pass_specs.py --grep schedule_pipeline` prints the stages, requires, produces and
invalidates of each declared pass, plus each entry point's declared stage order. Pass `--root` to
read your own tree. `reference/pass_specs.md` is the copy generated from the library as shipped.

## 2. Respect the phase order
Front end, then source/host-graph rewrites, then grouping and fusion decisions, then route
planning into a schedule `Kernel`. Next comes `compile_schedule` (admission, configuration,
fusion, prefetch, storage, post-storage, completion, emission, final), then counted-loop
cleanup, then staged or LLVM lowering and codegen. Decision rules:
- A consumer goes after every producer of its `requires`.
- `stage` numbers never decrease. An order is built for one `ir_level`; crossing levels means a
  lowering step and a new session.
- A mutating pass (`mutates_ir=True` or `invalidates=("*",)`) discards every earlier analysis.
  Re-run the analyses a later pass needs.
- Order by the input each pass accepts:
  - Passes that need flat call streams run before compaction into loops.
  - Fusion runs before allocation, which may fall back.
  - Emission runs after completion, and final checks run last.

## 3. Insert a pass the way the library does
Use the same pattern as `compile_schedule`: add the new product to the consumer's `requires` so the
dependency is checked, then validate the order before running it.

```python
from dataclasses import replace
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession, validate_pass_order

def insert_before(specs, new, consumer):
    at = next(i for i, spec in enumerate(specs) if spec.name == consumer)
    wired = replace(specs[at], requires=(*specs[at].requires, *new.produces))
    order = (*specs[:at], new, wired, *specs[at + 1:])
    validate_pass_order(order)
    return order

def run_order(module, specs, actions, context=None, **session_options):
    session = PipelineSession(module, context=context or PipelineContext(), **session_options)
    for spec in specs:
        session.run(spec, lambda spec=spec: actions[spec.name](session))
    return session, [(r.name, r.produced, r.invalidated) for r in session.records]
```

An action returns exactly its `produces` names. It reads inputs with `session.analysis(name)`
and mutates the IR by assigning `session.module`. A non-xDSL IR also needs `ir_level`,
`fingerprint` and `verify_ir`, as `compile_schedule` passes for its `Kernel`.

## 4. A/B two orders
- Static check: `python scripts/pass_specs.py --check compile_schedule --order a,b,c`. It reports
  the first pass whose prefix the library rejects, and why.
- Runtime: the library entry points take an order, e.g. `compile_schedule(pass_order)`,
  `compile_source_pointwise(pass_order, schedule_pass_order)` and
  `apply_common_pipeline(PipelineOptions(pass_order))`. To run alternatives on private clones,
  use `merlin.transforms.pipeline_policy.evaluate_variants`. To enumerate the legal orders, use
  `merlin.transforms.pass_exploration.generate_candidates`.
- Time the alternatives as variants or knobs with the `ab-library-pass` and `explore-schedules`
  skills. Keep the original order as the unconditional fallback.
- Per-shape choice among library pipelines from panel cycles: the `heavy-pass-recipes` skill.

## 5. Keep it generic, and debug it
- Gate an order on `PolicyFacts`: operation families, shapes, dtypes and manifest resources.
  Never gate it on test-case or model names.
- If an order fails, run `--check` and look up the error in `reference/phase-order.md`. If the
  output is wrong, read `session.records` and re-run `agent_selfcheck.py`: a legal order does
  not make emission correct.
