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

## Prepare shared IR once, then evolve alternatives

Pass libraries should reduce repeated implementation and compilation work, not
only describe which functions exist. `evaluate_variants(..., prefix=(...))`
executes common registered transformations once before cloning the resulting IR
for each alternative. The prefix can normalize the source graph and simplify
exact arithmetic; alternatives then explore fusion, tiling or schedule policy
at their declared IR boundary. Use only transformations legal for every branch
in this unconditional prefix. Shape/capability-dependent transformations retain
their predicates inside the alternatives.

For example, connect the existing shared normalization implementation directly:

```python
from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline
from merlin.transforms.pipeline_policy import RegisteredPass, evaluate_variants
from merlin.transforms.pipeline_state import PassSpec

def prepare_common_ir(session):
    prepared = apply_common_pipeline(
        session.module,
        PipelineOptions(index_bits=selected_index_bits, reuse_buffers=False),
        verifier=host_transform_verifier,
        context=session.context,
    )
    session.module = prepared.module
    return {}

registry["prepare_common_ir"] = RegisteredPass(
    PassSpec("prepare_common_ir", stage=0), prepare_common_ir,
)
for evaluation in evaluate_variants(
    source_module, variants, registry, derive_current_policy_facts,
    output="emitted_ir", context=selected_context,
    prefix=("prepare_common_ir",),
):
    if evaluation.status == "compiled":
        submit_immutable_artifact(evaluation.result.artifact,
                                  evaluation.result.receipt)
```

The selected index width, verifier, alternatives and target emission remain
caller-owned inputs. The real shared transform and its existing host verifier
run once for the prefix, not once per schedule alternative. This is preparation
within one search invocation, not a persistent source/verdict cache. A new source,
pass implementation or selected contract requires a new invocation.

Each branch receives a fresh session with **no prefix analyses**. Derive policy
facts from the prepared clone; register branch-local analyses before any pass
requiring them. Values containing SSA or schedule-object references cannot be
carried from the prefix into another clone. Stage ordering across the prefix and
branches is checked before work begins, while branch prerequisites must be
satisfied within that branch. The original source and prepared IR/context are
rechecked before and after each alternative. Failed shared preparation stops the
search; a failed branch discards only its own partial IR.

Receipts distinguish the original source, prepared IR, prefix pass records and
branch pass records. An agent can inspect that compact record, improve a shared
transformation or add a guarded policy, and let deterministic search evaluate
the alternatives without writing separate kernels or repeating normalization.
Promoting a successful alternative still requires independent correctness,
timing and additional shapes; neither preparation reuse nor a registry entry
establishes a compiler optimization's generality.

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
Prefix checks additionally run a real shared transform once across branches,
rederive separate IR-bound analyses, compare emitted-program numerical results,
and refuse source, prepared-IR and selected-contract mutation.

This module is usable by an ordinary compiler import. A sandboxed author also
needs it added to a new reviewed immutable compiler-library export; existing
exports and active author epochs are not changed by adding this source file.

## Generate pass orders and option alternatives

`merlin.transforms.pass_exploration` builds bounded alternatives for the existing
registry and evaluator. A `PassFamily` binds a real `RegisteredPass` for each
explicit finite knob tuple. Static or knob-dependent predicates use the same
source-stamped shape and resource facts described above. Existing stages,
prerequisites and invalidations, plus optional `before` edges, prune invalid
orders before transformations run. New or edited passes can join the registry
through the same factory; the generator contains no target or workload handlers.

`generate_candidates(..., output=..., max_candidates=8)` produces at most sixteen
alternatives under a separate finite enumeration budget. `explore_candidates`
executes those alternatives on private IR through `evaluate_variants`. The caller
must return `EmittedProgram` with actual emitted bytes, their explicit format and
the transformed schedule's digest. LLVM dialect IR is an emission boundary, not
machine-code execution evidence. Identical program bytes under identical selected
contracts retain their pass profiles but invoke the feedback callback once.

Optional caller-observed feedback must match the emitted program hash. Compilation
never invents cycles, chooses a fastest candidate or establishes numerical/hardware
qualification. Compact summaries contain options, executed passes, source/IR/program
identities, measured host pass times and duplicate links; full IR and independent
execution receipts remain in caller-owned artifacts. Read-only analyses may commute,
while transformations with dependencies or invalidated facts constrain legal order.
