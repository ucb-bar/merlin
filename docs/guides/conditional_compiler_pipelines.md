---
title: Conditional compiler pipelines
kind: guide
status: current
owner: core
last_verified: 2026-10-23
code_refs: [src/merlin/transforms/pipeline_policy.py, src/merlin/transforms/pipeline_state.py]
---

# Conditional compiler pipelines

`merlin.transforms.pipeline_policy` selects an ordered path through the existing
`PipelineSession`. It provides bounded branching and reusable compiler policy,
with no new dialect, pass execution engine or target implementation.

## Compile additional graphs

Additional models enter through the normal frontend and its typed source IR.
Derive `PolicyFacts` from that IR and selected numerical/hardware contracts, using
the current `session.stamp()`. Available predicates cover operand dimensions,
element types, layouts, operation semantics, graph fanout/live boundaries/liveness,
numerical semantics, capabilities and resource capacities. Missing facts are
unknown. Model names, capsule identities and measured private scores are not
predicate facets. Source stamping detects stale facts; it does not prove that a
caller's analysis or declared hardware capability is sound.

Register reusable shared passes or reviewed target adapters using their actual
`PassSpec` and callable. `registry_metadata()` exposes input IR kind, prerequisites,
produced analyses, mutation and invalidation declarations. Closures can bind
explicit pass options; no policy parameter is inferred from an import name.
The author chooses variants and their pass order. Dependency, invalidation and
stage checks run before any action, then the original session checks every pass.

```python
registry = {
    "prepare": RegisteredPass(prepare_spec, prepare_current_ir),
    "fuse": RegisteredPass(fusion_spec, fuse_legal_regions),
    "emit": RegisteredPass(emit_spec, emit_current_ir),
}
policy = PipelinePolicy((
    PipelineVariant("shared_input_regions", ("prepare", "fuse", "emit"), (
        GraphPredicate("max_fanout", at_least=2),
        PropertyPredicate("numerical_semantics", "preserve_layer_boundaries"),
        ResourcePredicate("scratch_bytes", at_least=required_bytes),
    )),
    PipelineVariant("unfused", ("prepare", "emit")),
), fallback="unfused")
result = execute_policy(session, policy, registry, facts, output="emitted_ir")
consume(result.artifact)
```

The example's specs, actions and capacity requirement are compiler inputs; it does
not implement or authorize fusion. Shared fusion adapters must preserve complete
region live-ins/live-outs, alias/effect legality, numerical policies and host
placement through their existing checks. Their `requires`/`produces` contracts
must refresh any analysis invalidated by a preceding rewrite. Unknown fusion
applicability selects the explicit unfused path. A transform that fails after
execution begins poisons the session; it cannot silently continue from changed IR.
Ordinary covered transforms still require the original host verification broker.

## IR and emission boundaries

Use separate policies at existing graph/region, schedule and target lowering
boundaries. A session keeps one declared IR kind. A lowering action consumes the
current session IR and produces the next IR as a named analysis; its native
adapter verifies that result. Open the next session on that returned artifact
with its actual fingerprint and verifier. Cross-dialect type checking and
live-boundary checks remain adapter responsibilities, not inferred from strings.

The final selected pass must produce the requested output, which is reopened
through `session.analysis()` before returning. Receipts include before/after IR
and context stamps, evaluated predicates, selected path, actual pass records and
invalidation. They do not certify semantic preservation, decoded target execution
or timing. A recorded plan alone is not an emitted optimization.

Compact counted-loop lowering can register its existing `Kernel`/compact-IR route
and verifier. This API does not force it through finite command schedules or
materialize iterations. Policies have at most 16 variants, 64 passes per path and
32 predicates per variant; selection is ordered first-match, with an explicit
fallback. Variant exploration uses separate private sessions and the existing
agent's bounded job/measurement ownership.

## Explore alternatives without another agent turn

`evaluate_variants` executes a bounded list of explicit alternatives through the
same registry and pass manager. Unlike first-match `execute_policy`, it visits
every applicable alternative. It yields each completion immediately, so the
caller can submit its actual artifact to the normal execution queue while the
next variant compiles. A skipped or failed alternative has no artifact. It
provides no simulator, ranking policy, correctness verdict or timing score.

```python
from merlin.transforms.pipeline_policy import evaluate_variants

for evaluation in evaluate_variants(
    source_module, variants, registry, derive_current_policy_facts,
    output="emitted_ir", context=selected_context,
):
    publish_compact_feedback(evaluation.summary())
    if evaluation.status == "compiled":
        submit_immutable_artifact(evaluation.result.artifact,
                                  evaluation.result.receipt)
```

The caller supplies the selected variants, source-derived facts, existing pass
registry and execution callbacks. Register parameterized shared transformations
under distinct names to explore options without an agent response for each
value. Keep predicates on every alternative whose legality needs them. Missing
capability/resource facts skip that alternative as `unknown`; the baseline is
an explicit unconditional variant, never an implicit rescue of failed IR.

Every alternative starts from a distinct clone of the same immutable source.
The full search space is checked for pass-order, prerequisite, IR-level and
output errors before executing any pass. A failed transformation discards its
partial result while later alternatives can continue on fresh IR. Changing the
original source aborts the iterator. Counted non-xDSL representations supply
explicit clone/fingerprint/structural-verifier adapters; no loop expansion is
introduced by the search driver. The maximum sixteen alternatives bounds retained
IR; consume additional candidate groups under the caller's normal resource budget.

Full receipts and transformed IR stay in caller-owned artifacts; `summary()`
contains only status, source/output-IR identities, executed pass names and a
bounded error. Independent numerical checks and comparable timing remain the
only basis for accepting and ranking performance candidates. Existing host
transform verification is unchanged; `plan_only` is still unverified planning.

## Readiness

Focused checks exercise a real shared normalization pass, emission of its changed
IR, the same policy on a scalar graph and a multi-output fanout graph with
interpreter comparisons, stale facts, invalid dependencies, failed transforms,
unknown applicability and a billion-iteration loop retained as one loop.
The checks use explicitly unverified planning for the transform test, not a
production semantic certificate. No target simulation is required by this API.
The machine-search checks additionally consume real normalized/unrolled IR on
zero, small, tail and full-loop shapes, compare interpreted outputs, retain a
trillion-iteration loop compactly, and test failed/unknown alternatives and
incremental delivery.

This module is usable by an ordinary compiler import. A sandboxed author also
needs it added to a new reviewed immutable compiler-library export; existing
exports and active author epochs are not changed by adding this source file.
