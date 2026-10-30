# Proof and domain review of the incoming bounded profiles

Read-only review of `8f30c77ac..591687da8` on 2026-10-29. The comparison contains
19 commits and 94 changed files. Review concentrated on generic-value loop
projection, index-control roles, recursive configuration hoisting, source
eligibility and coverage wording. All source references below refer to
`591687da8` unless explicitly identified as the local cutoff harness.

Bare `GraphState*.lean` filenames refer to
`src/merlin/verify/lean/Merlin/`. Bare verifier Python filenames refer to
`src/merlin/verify/`, and bare `test_*.py` filenames refer to
`merlin/tests/ir/`. These are distinct from the tensor tree under
`merlin/lean/MerlinVerify/`.

No runtime, specification, approval, ledger or guide was changed. No Lean build,
pytest cohort, native executable or fresh replay was run for this review. The
older partially completed full-model campaign belongs to `8f30c77ac` and must
remain separate from new-source qualification.

## Prioritized findings

### Reporting compatibility defect in the local cutoff harness

The local `build_tools/scripts/qualify_pass_cutoff.py` predates the new bounded
policy. Its `summarize_case` function at lines 200–235 decides success from LLVM
equality and submitted proof jobs. At line 234 a zero-job compilation is labeled
`lowered_no_changed_selected_boundary`. That inference is no longer valid.

`src/merlin/verify/pass_validation.py` now records an original-source exclusion
and returns without submitting a job. `compiler_coverage.py` lines 62–64 and
139–143 record the boundary as unverified and give it no proof credit. The
ordinary transformation is still allowed to change and produce compiler output.
Thus a changed, excluded boundary can be mislabeled as unchanged by the local
campaign harness. An excluded boundary can also be hidden in `lowered_checked`
when another boundary has an accepted job.

This is a reporting integration issue, not evidence of an unsound Lean theorem
or an incorrect new production coverage journal. Existing old-cut receipts are
not invalidated by the later policy change.

Suggested correction for the new campaign is to bind and audit the finalized
`coverage.json`, count verified, unverified and unchanged boundaries separately,
and give any successful compilation containing an exclusion a distinct partial
coverage status. Missing or inconsistent required coverage must not become a
fully verified result. Preserve accepted job counts even in a partial case.

A meaningful regression should supply matching baseline/gated LLVM and a
changed unverified coverage boundary with zero jobs, then assert zero proof
credit and a partial status rather than the unchanged status. A second case
should combine one accepted job with one excluded changed boundary. Existing
`test_compiler_bounded_policy.py` lines 103–120 provides a real per-channel-i64
Q/DQ example of intentional zero-job ordinary lowering; its explicit selector
must remain separate from the default-five campaign.

### Generic UNROLL needs a checked-in semantic negative at its own new gate

The inspected checked-in `test_compiler_loop_generic.py` covers a real positive
worker and fresh replay, custody failures, and a target rejected before the SDK
for unsupported literal metadata. Its latter test at lines 79–100 is a domain
refusal, not a valid wrong-rewrite proof failure. The replay mutation at lines
132–156 tests changed evidence, not an incorrect transformed loop.

By comparison, `test_compiler_loop_generic_tile.py` lines 77–113 changes the
actual inner bound while retaining valid IR and checks that the emitted query
reaches Lean, fails without a heartbeat timeout, and produces no accepted
report. That is meaningful semantic-negative coverage for TILE. It should not
be counted as exercising UNROLL's different generated proof body.

Suggested regression is to start from the admitted generic UNROLL fixture and
change only the second cloned `tick` call's IV or same-typed captured tensor.
Keep the source, function ABI, source profile and target syntax valid. Require
preparation to reach the query, imported modules to build successfully, the
closed query to fail, and the compiler result to remain quarantined. This is a
qualification gap, not a demonstrated soundness defect.

### Several source headers still describe integrated code as ineligible

Examples are `GraphStatePipelineValueCandidate.lean` lines 1–3,
`compiler_loop_generic_intake.py` line 1,
`lean_graph_generic_projection.py` lines 1–5 and
`compiler_configuration_hoist_recursive.py` lines 1–5. Their candidate/uncompiled
or future-checker wording predates actual integration. The current generic gate
imports those modules and runs their queries, and recursive hoisting now has a
closed checker. These headers can mislead a reviewer about implementation state.
Updating explanatory headers would not require changing theorem statements or
approval flags. No header was changed during this review.

## Generic-value loop soundness boundary

No theorem-level soundness defect was identified in the inspected core. The
following distinctions are essential to keeping that conclusion bounded.

`GraphStatePipelineValueCandidate.lean` defines a `Codec Value` with a left
inverse only for encoded indices. Opaque tensor, float and complex payloads stay
in `Value`; `sumCodec Payload` specializes this to `Int ⊕ Payload`. There is no
claim that arbitrary payloads survive an integer encode/decode round trip.
`External` permits partial outcomes, errors and terminal world, constrained by
the successful result arity. Shared pure primitives retain the existing
parametric interpretation.

`lean_graph_generic_regions.py` closes structured non-SCF operations over their
actual explicit operands and implicit captures. Exact descriptors include
operation metadata, types, nested blocks, result identities and locations.
`lean_graph_generic_projection.py` independently decodes complete source and
target function rosters, while operational SCF keeps explicit bounds, carries
and bodies. Defined/internal callees, unknown symbol dependence, function values,
unmodeled CFG and resource literals are not silently assigned opaque meanings.

Control-role retagging is deliberately occurrence-sensitive. Ordinary prefix
events keep their arbitrary partial meaning even when they share the metadata
of a generated index operation. The UNROLL source profile restricts the changed
slice to control literals and a single zero-result opaque event, so its broader
suffix role binder does not admit arbitrary source index computations in the
body. TILE's role binder explicitly selects only setup and generated outer
add/min occurrences, leaving the retained event opaque.

The generated query checks exact role application, unchanged ABI, scope joins,
the complete function joins and fitting selected controls. See
`lean_graph_generic_unroll_emit.py` lines 313–365 and
`GraphStatePipelineControlRolesCandidate.lean` lines 19–39. These are syntactic
and model obligations. They are not a theorem that an arbitrary raw external
interpretation equals the selected mathematical control interpretation. T7 still
owns that actual-IR correspondence. The ledger and bounded-profile guide now
say this explicitly.

`GraphStatePipelineReachableCandidate.lean` establishes fitting actual IVs and
successors using symbolic counted-range membership. The compiler source profile
also restricts generated control bounds to signed index64. Neither this proof
nor the abstract equality theorem supplies concrete floating arithmetic, LLVM
lowering or native fidelity.

The generic compiler source profile is exactly the following narrow addition
to the older scalar-word profile, not a general tensor-loop optimizer.

- Exactly one `scf.for` in the entire module, at top level, with no carried
  values and three immediately preceding literal controls.
- One zero-result opaque event using the IV and one captured data value, followed
  by the loop yield and an immediate function return of one captured value.
- UNROLL uses factor two with an odd trip count of at least three, covering a
  nonempty bulk and one tail iteration.
- TILE uses a nonempty loop and tile size 2 through 65536.
- Static nonnegative selected bounds, positive stride, signed-index64 fitting
  controls and explicit syntax/work/byte/depth budgets.
- Exact common prefixes and unchanged other definitions. Additional SCF loops,
  carries, multi-event bodies, effectful suffixes and generic UNROLL→TILE
  composition remain excluded.

Scalar NaN attributes remain excluded because the parser can lose literal
identity. Runtime opaque NaN values are not excluded by that literal rule.
Supported ordinary dense literals retain byte identity; floating-complex dense
and normalization-sensitive assembly remain outside the current parser profile.

## Recursive configuration hoisting

No theorem-level soundness defect was identified in the recursive proof.
`GraphStateConfigurationRecursiveHoist.lean` explicitly interprets enclosing
loops, restores the parent IV environment for a suffix, derives successful-state
fixedness through nested bodies, and composes exact moves through lexical
contexts. The final checker requires exact frame and atom rosters, stage joins,
nonempty moves and positive moved-loop extents.

The decisive remaining assumption is `Recursive.Models` at lines 40–43. It is
universal over IV environments and admitted calls. `movable_sound` and
`check_sound` require this premise; the new theorem does not derive ISA fidelity
from the selected contract files. Moved setters must be total exact constant
overwrites without hidden effects. Relevant state, read locality, configuration
nonaliasing and frame behavior remain A10, with actual projection under T7.

The actual projection refuses IV shadowing, carried completion fields and
unsupported expressions. Ordinary static extents retain the one-million bound.
An original `ConfigurationCConsumer` can separately select the signed-index64
wide profile after checking both endpoints. That consumer identity is not
implied by an unrelated `PassValidation(index_bits=64)` setting.

The checked-in recursive suite includes inner and multi-level motions, indexed
outer/inner expressions, overlapping setters, legacy-domain relabeling refusal,
original C64 ownership, genuine target operand/frame negatives and finite native
state/failure checks. The native fixture uses the constant-data `inner` case;
the indexed case gets a real gate proof but is not the 27-observation native
fixture. A useful additional test would use both IVs and an outer-IV-dependent
suffix through actual C emission, to exercise restoration at the trusted
projection/emission boundary. This would qualify the bridge, not discharge
universal `Models`.

## Coverage and claim conclusions

`DEFAULT_CHECKS` is still five. The new generic checker is an additional route
for existing loop selectors, not an extra default family. Configuration,
proposed Q/DQ, draft MACC and host fusion remain explicit selections.

The largest policy change is that known original-source exclusions now run
ordinary transformations with final unverified coverage. Source-supported
target failures, unsupported targets and tool/custody failures remain blocking.
The current guide accurately distinguishes individual bounded claims from
whole-pipeline preservation through an unverified suffix.

The guide also correctly identifies the LLM-context experiments as added SCF
probes surrounded by recaptured unchanged prefixes. They do not show that
original LLM loops were optimized, or any performance gain. Native examples and
prior recorded proof counts should retain their own source cuts.

Consequently, “all kernels lowered with verification selected” requires separate
counts for fully covered selected boundaries and partial runs containing
exclusions. It cannot by itself strengthen the paper to every transformation,
all tensor loops, arbitrary nested scheduling, numerical fidelity or hardware
correctness. Q/DQ source review and MACC numerical approval remain pending.
