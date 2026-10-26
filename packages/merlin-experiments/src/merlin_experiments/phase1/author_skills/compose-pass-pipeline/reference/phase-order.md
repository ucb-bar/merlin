# Phase order, derived from the library's own declarations

There is no global pass manager. Each representation (`ir_level`) has small pipelines built from
`merlin.transforms.pipeline_state.PassSpec`, and moving between representations is an explicit
lowering step. The order below comes from the entry points' declared stage tuples
(`compile_trace.declare`) and their `PassSpec`s. `scripts/pass_specs.py` prints both kinds from
your tree; `pass_specs.md` is the copy generated from the library as shipped. Every cited symbol
is resolved against the library by this skill's test.

## Whole program

1. **Front end.** Capture produces a linalg-on-tensors module. Its stage is `capture`
   (`merlin.compile_cli`).
2. **Source and host-graph rewrites** (`linalg`, `host_tensor`). The order is fixed by
   `merlin.llvmlower.lower.STAGES`: source-recomputed, host-graph-optimized, host-row-scheduled,
   host-padding-gather-forwarded, host-destination-reused, host-rounding-selected,
   host-permutation-forwarded, host-tensor-preparations-shared, operation-profiled, then the xDSL
   preprocess stages, then upstream MLIR passes. Each step is its own three-pass session (source
   check, rewrite, conformance):
   - `merlin.transforms.recomputation.plan_source_recomputation`
   - `merlin.llvmlower.graph_optimizations.optimize_host_graph`
   - `merlin.llvmlower.automatic_gathers.prepare_automatic_gathers`
   - `merlin.llvmlower.automatic_destinations.prepare_automatic_destinations`
   - `merlin.llvmlower.automatic_rounding.prepare_automatic_rounding`
3. **Grouping and fusion decisions** (linalg). The order is: form groups, decide each
   contraction's epilogue cut, then form the groups again with those decisions. See
   `merlin.llvmlower.source_compute_groups.form_exact_groups` and the `fusion` skill. Offload
   selection follows: `merlin.llvmlower.offload_plan.realize_offload_plan`, with passes
   offload-source, offload-plan and offload-conformance.
4. **Per-route planning, which builds a schedule `Kernel`.**
   - Block and tile choice: `merlin.transforms.dense_planning.choose_dense_block`.
   - Loop proposals: `merlin.compile.scheduling.loop_schedule.propose_loop_transforms`. These are
     proposals; the route's verifier admits them.
   - Route sessions, for example `merlin.compile.scheduling.compact_dense.schedule_compact_dense`
     with admission, blocking, emission and realization.
5. **Schedule pipeline:** `merlin.compile.scheduling.schedule_pipeline.compile_schedule`. Each pass
   is listed with its stage number:
   - semantic_admission(0)
   - configuration(1), if a configuration adapter is given
   - fusion(2), if fusion capabilities are given
   - prefetch(2), if admitted
   - storage(3). Copy elimination runs inside allocation, and copy pipelining inside this pass.
   - post-storage rewrite(4), for example dead writes
   - completion(4)
   - emission(5)
   - final_program(6)

   Each optional pass adds its product to its consumer's `requires`, using `dataclasses.replace`.
   Use the same pattern when you insert your own pass.
6. **Target-instruction cleanup**, at the `target_instruction` level, e.g. the configuration
   cleanup session in `merlin.compile.scheduling.configuration_ir`. The following standalone passes
   are not wired into any pipeline, so where they go follows from the input each one accepts:
   - Flat `Call` streams: `merlin.compile.scheduling.configuration.eliminate_redundant_configuration`
     and `merlin.compile.scheduling.fusion.plan_fusion`. The fusion realizer leaves counted loops
     unchanged.
   - Pre-allocation objects and banks: `merlin.compile.scheduling.copy_coalescing.coalesce_private_copies`
     and `merlin.compile.scheduling.copy_elimination.eliminate_private_copies`.
   - Compaction into counted loops: `merlin.compile.scheduling.command_compaction.compact_command_stream`.
     It changes representation only, so run it after the flat-stream passes.
   - Intact counted loops, so after compaction:
     `merlin.compile.scheduling.configuration_loop_cleanup.eliminate_counted_configuration`.
7. **Lowering.** There are two routes:
   - Staged: `merlin.xdsl_dialects.lowering.pipeline.STAGES` runs input, contract, schedule,
     interface, target and runtime, and ends in the command buffer.
   - LLVM: `merlin.llvmlower.lower.STAGES` ends at llvm-final, followed by
     `merlin.llvmlower.codegen.STAGES` (object, then link).

   The configuration-IR route and the target-schedule route are mutually exclusive in one device
   build.

Structured scalar/loop transforms on xDSL IR use `merlin.transforms.pipeline.apply_common_pipeline`
in this order:
- normalization and simplification (stage 0)
- storage analysis, then buffer reuse (stage 1)
- loop segmentation and unrolling (stage 2)
- dependency analysis, then dependency schedule (stage 4)
- final analyses, then verification (stage 5)

Pass `PipelineOptions(pass_order)` to change it. Each change carries a certificate:
`merlin.transforms.certificates.make_change` builds it and
`merlin.transforms.validation.validate_change` re-derives its witnesses. Certificates are not
passed between passes. A later pass trusts only fresh analyses, stamped by the session with the
IR digest and the contracts.

## Rules `validate_pass_order` and `PipelineSession` enforce

`validate_pass_order` checks a whole order before anything runs. `PipelineSession.run` checks each
pass again as it runs.

| Error text | Cause | Fix |
|---|---|---|
| `lacks prerequisites: X` | `X` is not produced earlier, or a mutating pass in between invalidated it | move the producer later, or re-run the analysis after the mutation |
| `invalid explicit pass ordering before P` | a stage number decreases | reorder, or give `P` a stage between its neighbours |
| `requires a different IR level` | two IR levels in one order | end the session, lower, start a new session |
| `duplicate pass` | the same name appears twice | register a separately named variant |
| `missing or stale analysis: X` (runtime) | the IR changed since `X` was produced | re-produce `X` after the change |
| `analysis pass P changed IR or context` | `mutates_ir=False` but the IR changed | declare it mutating, or keep it pure |
| `produced a different analysis set` | the action's mapping keys are not `produces` | return exactly the declared names |
| `changed IR level without an explicit lowering boundary` | a lowering inside a session | split the session at the lowering |
| `pipeline repair budget exhausted` | fallback replanning hit `max_repairs` | fix the failing allocation or raise the budget deliberately |
| `explicit schedule order must contain every selected pass exactly once` | `compile_schedule(pass_order)` is not a permutation | list every selected pass |

Two details matter when you order passes by hand:
- A pass with `mutates_ir=True`, or with `"*"` in `invalidates`, makes every analysis that existed
  before it unavailable.
- Replacing an analysis also invalidates, transitively, every analysis derived from it.

## Running two orders on private clones

```python
from merlin.transforms.pipeline_policy import PipelineVariant, PolicyFacts, RegisteredPass, evaluate_variants

def compare_orders(module, specs, actions, orders, output, **search_options):
    registry = {spec.name: RegisteredPass(spec, actions[spec.name]) for spec in specs}
    variants = tuple(PipelineVariant(name, tuple(order)) for name, order in orders.items())
    facts = search_options.pop("facts", lambda session: PolicyFacts(session.stamp()))
    return {result.variant: (result.status, result.error, result.result)
            for result in evaluate_variants(module, variants, registry, facts, output=output, **search_options)}
```

Every order is validated before any pass runs: one illegal order raises `ValueError` for the whole
search. Each variant then runs on its own clone. Its `status` is `compiled`, `failed` (with the
error), `unsupported` or `unknown` (when its predicates do not hold or cannot be decided). For a
compiled variant, `result.artifact` is the `output` analysis and `result.receipt["passes"]` lists
what each pass produced and invalidated. For a non-xDSL IR, pass `clone`, `fingerprint` and
`verify_ir`. Correct and faster are still decided by `agent_selfcheck.py` and timing.

## Keeping an order generic

- Decide the order from facts, never from a case or model name: the operation's semantics
  (families, shapes, element types, live outputs) and your capability manifest (units, readouts,
  capacities). With `merlin.transforms.pipeline_policy.evaluate_variants`, state those facts as
  `PolicyFacts` and gate variants with `ShapePredicate`, `PropertyPredicate`, `ResourcePredicate`
  or `GraphPredicate`. A predicate that cannot be decided yields `unknown`; it never grants a
  permission.
- `merlin.transforms.pass_exploration.generate_candidates(families, output, before)` enumerates
  every topological order that `validate_pass_order` accepts. `before` holds explicit
  `(earlier, later)` edges, used only when the data dependencies do not already imply them.
- An unconditional fallback (the original order) must stay available.
