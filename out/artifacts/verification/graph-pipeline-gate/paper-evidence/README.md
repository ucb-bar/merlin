# Latest source-only successor: empty/full-inline ordered unroll

See ../development/unroll-inline-v1/{README.md,UNROLL-VARIANTS.md,QUERY-INVENTORY.json,
CHECKER-STATUS.json}. Complete actual source/target/observer-frame query candidates
cover empty/trip1/2/3 variants, including both escaping carries and partial failures.
Five changed captures,5 SDK-free tests,13 valid host-refused mutations; new Lean
UNCOMPILED/outcomes UNOBSERVED/native UNQUALIFIED. Historical no-tail qualification
and tail candidate remain separately pinned and unchanged. No correctness/approval
claim yet. Existing candidates should be checked before further expansion once a
new exact common native context is explicitly authorized.

--- Previous paper evidence (historical) ---

# Latest source-only paper successor: ordered-unroll nonempty remainder

See ../development/unroll-tail-v1/{README.md,EIGHT-KIND-COVERAGE-PROPOSAL.md,
QUERY-INVENTORY.json,CHECKER-STATUS.json}. Complete actual bulk+tail operational
query candidates preserve return/fault/terminal world and all entries under one
registry/model. Four changed actual pairs, structural nonzero success/bulk-fault/
tail-fault witness,11 valid host refusals and5 SDK-free tests. Native hold remains;
new Lean UNCOMPILED/outcomes UNOBSERVED/native UNQUALIFIED. The old no-tail domain
and qualified evidence do not expand. No paper correctness/approval claim yet.

--- Prior paper evidence (historical) ---

# Compiler gate repair — source checkpoint, not qualification

Current assignment: `planner/paper-deadline-20261026/gate.md`. Review findings
D1-G09/D3-G10 (local/identity proofs granted whole-program authority), plus shared
F admission for D2-G01/G02. Primary `feat/pass-verification`; corrective producer
base4499 with an explicitly different uncommitted patch. Original4499 is unchanged.

## What can currently be said

The corrective compiler gate rejects legacy local/identity completions. SDK-free
actual-producer tests establish that enforcement and exercise the new supported
route with a **mocked** checker. The real new gate returns UNAVAILABLE under the
selected-runtime hold. No new compiler-enforced Lean acceptance is claimed.

The candidate supported route covers `scalar_redundancy_elimination` (CSE/DCE)
and `scalar_loop_invariant_motion` (LICM). Proposed eventual claim, pending native
checking, fixed-source qualification and review:

> For admitted complete module pairs, a fresh kernel check of the generated
> `merlin_transform` establishes equality of the modeled outcome of every defined
> entry, including returned values, error identity and terminal world, under one
> common scalar/SCF interpretation and the explicit local contracts below. The
> trusted host maps actual compiler IR into both programs and binds their complete
> signatures, contexts, bytes and the selected checker/source identities.

This is not a claim about every graph pass, full model execution, poison freedom,
native instruction fidelity, divergence, or approved numerical policy. Shape,
integer simplification, storage, segmentation, unrolling and scheduling do NOT
inherit this new route. Their legacy local completions fail compiler admission.
Their existing separately scoped standalone evidence remains historical.

## Existing roots and new candidate root

Existing proved/reviewed roots reused unchanged:
`Merlin.GraphState.Scalar.check_sound`, `check_typed`, `check_initialized`,
`check_operands_initialized`; and `Merlin.GraphState.Pipeline.entries_cons` /
`entries_nil` from the prior full-roster pipeline proof (historical basis). The current
additive `Merlin.GraphState.GateEntries` source retains those exact definitions and
proof bodies using ONLY the unchanged Scalar import; its own compilation is pending. `check_sound` derives
semantic equality from typed normalization, with actual SSA slots substituted
through lexical environments. It does not assume whole-program equality.

The new generated `Merlin.GraphPipelineGateInstance.complete` applies the
per-function proofs to `PreservesEntries`, preserving missing entry and wrong
arity behavior as defined by `evalEntry`. `merlin_transform` is the exact public
root. Both programs are independently serialized from their actual buffers;
there is no single-blueprint after-mode shortcut. This new wrapper and the
arithmetic witness are **UNCOMPILED** at this checkpoint. Historical proofs of
its dependencies do not constitute checking the new wrapper.

## Complete domain, relation and premises

- Complete nonempty module/function/declaration roster, identical module and
  function metadata/signatures. Defined functions have one block; unsupported
  CFG/region geometry refuses the whole boundary.
- Static slot typing and lexical operand availability are checked in the model.
  Abstract runtime values need not themselves be machine bit patterns for the
  generic equality theorem. Typing/initialized reads do not assert successful
  execution or absence of poison.
- Pure primitive whitelist: unflagged signless fixed-width constants, arithmetic,
  extensions/truncations with exact descriptors and empty effects. Totality,
  determinism and world independence are explicit primitive contracts. Empty
  `get_effects` alone does not make an arbitrary operation pure. Other retained
  operations are opaque descriptors, never silently dropped.
- Opaque meaning depends only on the exact ordered operands and complete world;
  successful result arity is a local contract. Failure exposes the error and
  terminal world. Nondeterministic choices, if needed, belong in the world.
- Scalar SCF uses one common finite `iterations` and index interpretation.
  Positive constant stride/shape admission is trusted Python. Concrete loop
  control/native execution and divergence are NOT proved by this route.
- No local/indirect calls: F shared admission scans raw metadata BEFORE parsing,
  then recursively inspects attributes/properties/types/symbol references, in
  `external-only` mode. Resources/opaque attributes outside its policy refuse.
- Parser, projection, literal serialization, source identity and host process
  integrity are trusted correspondences. Only std3 is allowed in the planned
  native exact-root/transitive audit. No extra axioms or reflection mode.

## IR-field map

| Actual field | Mapping/check | Lean relevance / trust |
| --- | --- | --- |
| Original before/after UTF-8 bytes | SHA256 and fresh parser calls; raw admission before either parse | Byte commitment, not a semantics theorem |
| Module and function attrs/properties, names, ordered declarations/signatures | scalar `metadata` comparison and complete roster | Names/metadata correspondence is trusted; generated entries include every defined body |
| Ordered block args and nested captures | lexical environment, newest results prepended, operand lookup | Independent `Program` terms; `checkTypes` / normalization soundness |
| Pure opcode, widths, flags, literals, operands | exact whitelist/metadata and literal `Prim` constructors | Total primitive contract; no numerical reassociation law |
| Opaque operation/call descriptors, signatures and operands | one jointly numbered descriptor registry | Identical descriptor means common local effect interpretation, not native proof |
| SCF bounds/carries/yields/body/context | exact operand indices, carry types, recursive body projection | Shared abstract finite control; supported lexical environment preservation |
| Complete return vector | literal ordered `.ret` indices in both terms | Complete returned values and terminal world/error relation |
| Entry roster | complete separate before/after `List Entry`, per-entry proof | `entries_cons` includes wrong arity and missing entry behavior |
| Compiler transformation/certificate | exact type, registered kind/scope, canonical entire change hash | Structural producer certificates alone carry no proof authority |
| Checker/root/model/domain | host-fixed generated query/roots/relation/premises | No caller-supplied JSON/domain/query can issue completion |

## Retained worked example and negative

`../development/minimal-roster-request-v1/` contains actual before/after,
projection, generated complete query, source/parser closure, and real gate output.
The actual CSE removes duplicate `%a + %b`; both uses feed multiplication and
then the external observer. An unchanged second defined entry is included.

The separate `ArithmeticWitness.lean` candidate uses the retained exact wrapping
integer interpretation and an observer that ECHOES its operand, advancing world
by one. Proposed exact runs: `[3,5] -> [64]`, `[4,5] -> [81]`, initial world7,
final world8. The partial version faults with error17/world8 at the observer.
Source success and initialized reads are candidate conclusions, not premises.
The model is not self-approved, and these assertions have NOT run in Lean.

`wrong-result.mlir` changes the valid multiplication to subtraction. It passes
actual parser/IR verification and yields a different independently projected
`WrongResult.lean`. Its Lean rejection is PENDING. It is not counted as a native
negative, and the host example evaluator is not a proof checker.

`GATE-OUTPUT.json` currently says unavailable/selected runtime held. There is no
retained positive native checker output for the new gate yet. The earlier scalar
worked example at `planner/graph-first-20261025/worked-example/` is a distinct,
historical standalone result; it cannot supply missing live gate completion.

## Current tests, failures and remaining gate

41 SDK-free tests pass in `../development/complete-route-tests-v26.txt`.
The actual producer is used for CSE and empty/nonempty LICM cases; live success
is exercised ONLY with a typed mocked kernel/revalidation. Tests cover legacy
identity/local receipts, arbitrary returns, saved/copy/forged/stale/wrong-domain
objects, skipped/wrong-root/nonstd3 checker output, complete unrelated functions,
resources and indirect calls, and unavailable-vs-unsupported classification.
Source-only replay reconstructs all data and must call a fresh checker; no saved
receipt seeds the process-local ledger. The native backend is hard-held.

First failures are retained: malformed memref custom syntax, actual legacy false
acceptance, producer/primary optional runner import mismatch (removed by extracting
the unchanged pure serializer), and malformed initial mutation fixtures. Corrected
fixtures are separately named. Nothing was reclassified as old qualification.
The old integer v2 broad ValueError-status limitation remains historical.

Next: root-approved NEW context transport from F, source/import/object closure,
exact-root/transitive std3 audit, real changed `plan_only=False` acceptance and
valid semantic negatives, fixed-source qualification population and ONE retained
replay, then review/integration. Installed SDK and old native profiles remain held;
no inherited guard or archive binary is opened by this worker. Coordinator alone
stages/commits/integrates/pushes. All old source snapshots/capsules/counterexamples
and numerical failures are preserved.

## Shared native context adapter candidate

The INACTIVE adapter in `lean_graph_pipeline_gate_native.py` consumes F's typed
Context/CheckedQuery API. It checks exact19-module source selection, raw query
bytes, the complete final/support root roster, std3 axioms and context identity,
and retains the original process-issued CheckedQuery for immediate revalidation.
Ordinary kernel diagnostic maps to `not_established`; a context/binding failure
maps to `unavailable`, never `unsupported`. SDK-free tests mock these methods and
forbid native guards. Public `check` and `revalidate` still unconditionally refuse;
root's basic-image probe alone has not authorized this gate context.

The named local diagnostic now returns `local_checked`, with a `local_` method,
and is removed from the generic completion ledger. It is neither a generic nor
a pipeline verification receipt. This is a NEW corrective source version, not a
reinterpretation of the old broker's receipts.

## Pure-declaration query v2 — prelaunch review correction

Root review found that the earlier candidate emitted `#print axioms` while the
common Context required empty compile stdout. The new Prepared schema v2 explicitly
binds `pure_declarations_separate_root_audit_v1`; neither complete nor witness
queries contain in-source print/check commands. Separate exact-root/std3/type
audits remain mandatory in the common Context. Historical generated queries were
NOT edited. `development/FIRST-CONTEXT-INTEGRATION-FINDING.json` retains this
source-review blocker; no native attempt occurred. F/root separately own an exact,
source-pinned compile-audit-root policy for unchanged frozen dependency sources
that themselves print axioms. Arbitrary compile output is never admitted.

## New development attempts and frozen-provider integration

Two explicitly authorized development contexts stopped on their first raw
infrastructure failures: q-gate-context-v3 could not resolve the Lean source prefix;
q-gate-context-v4 reached ResourceGeometry compilation (exit0) but its loader trace
contained an unreviewed optional symbol, l_Std_DHashMap_Raw_term___x7em___00__boxed.
Neither yielded a complete query, root audit or accepted live result. All raw
records stay in the private contexts and development/authorized-context-dev-v1/v2.
No failed context was retried or promoted; the installed SDK incident is unresolved.

Source-checkpoint-v5 adds exact frozen-provider module/class binding and includes
all9 owned sources. The fresh complete IR/query/projection and proposed UNAPPLIED
activation delta are in development/frozen-provider-integration-v1. Its actual
plan_only=False test traverses the full adapter using MOCKED Context methods only.
Public native activation remains held. Source/host trust and all earlier premises
remain explicit; these improvements do not discharge numerical or native fidelity.

## Latest source cut and exact remaining prerequisite

Source-checkpoint-v6 / frozen-provider-integration-v2 bind frozen F v5 controller3
and the exact reviewed-runtime-intent-v2 plus provenance. SDK-free38-test adapter
wiring remains passing; no public activation. The third authorized development
attempt (q-gate-context-v5) stopped in a frozen dependency's derived Repr compilation:
GraphLoopsSegmentModel line42 requires a referenced executable specialization that
the compiler reported missing. Exit1 then correctly fails the nonzero loader policy.
Raw source/compiler/runtime evidence is preserved; no frozen proof or observer
policy was changed. Root owns explicit resolution/new authorization of this exact
prerequisite. New gate wrappers remain UNCOMPILED and UNQUALIFIED.

## Current paper row — minimal adapter source, kernel check pending

| Transformation / roots | Domain and relation | Actual retained boundary | Current evidence / limits |
| --- | --- | --- | --- |
| Corrective scalar CSE/DCE; unchanged Scalar.check_sound plus NEW GateEntries.entries_cons/nil and generated merlin_transform | Complete admitted module roster; every defined entry, all arguments/worlds; same Scalar.evalProgram returns/error/world, missing entries and wrong arity preserved; explicit total/pure primitive and local opaque arity/effect contracts | minimal-roster-request-v1 before.mlir/after.mlir/Query.lean/PREPARED.json; observer and unchanged second entry included |41 SDK-free tests and exact source/namespace correspondence. Adapter/query UNCOMPILED, GATE-OUTPUT unavailable. No new complete native query, no public activation, no qualification/replay/human approval. |

The new adapter imports only GraphStateScalar; its project closure is7 modules,
with no Pipeline/index/resource/loop-storage dependencies. This avoids importing
unneeded frozen definitions but does NOT establish repair of the shared Std
executable-code failure. Entry fields and evalEntry/PreservesEntries/proof bodies
are exactly the prior definitions after namespace rename and omission of unused
deriving instances. Complete query source differs only by that import/namespace.
No substituted scalar evaluator or new whole-program-equality premise is used.
This is a source-level correspondence argument, not a claimed cross-namespace
kernel bridge. Both source copies remain available for human/independent review.

Gate checking-path contract now requires an actual bool plan_only. A changed
planning result remains explicitly marked planning and carries no completion
authority. A failing second CSE→LICM boundary does not return a partial pipeline
result or mutate the caller's module; cloned-IR atomicity is SDK-free host evidence,
not rollback of callback side effects or a Lean theorem of cross-stage composition.
Public native dispatch is still held. First positive/negative/witness and fixed
qualification/replay remain subject to exact source/context authorization.

## Latest minimal attempt and activation boundary

The first minimal native context compiled six frozen scalar dependencies, then
stopped on an unused-outputs-binder warning in the new roster adapter (compile
exit0, strict stdout policy refusal). No complete query/final-root acceptance was
issued. Exact raw record: authorized-minimal-dev-v1 / q-gate-minimal-context-v1.
New minimal-roster-request-v2 renames only that binder; its theorem statement,
proof and generated query bytes are unchanged. It requires fresh authorization.

Source-checkpoint-v8 and activation-review-v1 complete a typed construction recipe
for the exact Fv5/minimal7 context;42 SDK-free tests cover actual checking-path
integration with mocked native methods. Public activation remains held. The full
CommonTransformService still constructs legacy LeanToolchain; coordinator-owned
broker/profile integration is a separate explicit gap. The leaf route's correctness
must not be presented as end-to-end full broker or compiler-service verification.

## Current overriding runtime limit

The corrected minimal-v2 attempt stopped during post-validation because the
PRIVATE selected libleanshared.so did not match its original pinned hash. Exact
first report/raw evidence is retained in authorized-minimal-dev-v2 and the private
context. Cause UNKNOWN; no reread, rehash, repair, retry or observed-pin adoption.
This is a separate retained incident from the installed SDK mismatch and prior
compiler/warning failures. No complete gate query or root audit has been accepted.
All native/qualification/public-activation claims remain false.

## Latest source-only service integration (supersedes missing-injection gap)

Source-checkpoint-v9 adds explicit trusted GraphServiceBinding injection to the
corrective CommonTransformService. The actual service→whole-boundary gate→pipeline→
live compiler eligibility path is exercised with mocked kernel and bundle/profile
custody.46 SDK-free tests plus final4 targeted service regressions pass. Saved/copy/
mutated reports, missing profiles, unsupported passes, held contexts and later-stage
refusals cannot grant eligibility. No default legacy toolchain fallback exists on
this graph path; graph-bound authored-check also refuses it.

service-integration-v1 retains actual before/after/query and marked MOCKED service
output, profile, source closure and exact release recipe. This closes the owned
host wiring source gap; genuine ordinary-release/profile admission and native
acceptance are still NOT established. No shared profile/import policy or frozen
source was changed. The returned conditional-model status and per-boundary scope
must not be restated as whole-chain, whole-model, hardware or numerical verification.
All old native permissions remain suspended under the private-runtime mismatch.

The final service test supplement exercises actual LICM through the same injected
minimal graph profile and confirms changed profile digest, extra project source
and changed dependency-audit policy refuse before any checker call.6 final SDK-free
service cases pass; implementation is byte-identical to the independent v1 review
cut. Source-checkpoint-v10 captures the final test addition. These results remain
mocked host/API evidence, not kernel acceptance or ordinary release qualification.


## Two-kind common-model endpoint candidate (new source-only cut)

Exact proposed paper row: for supported complete scalar CSE/DCE then LICM modules,
a new source-to-final full-entry operational query reuses Scalar.check_sound and
GateEntries roster soundness under one shared descriptor/type interpretation. The
relation includes every entry, missing entry, arity error, return, external failure
and terminal world. A separate composed theorem joins both actual boundaries.
No whole-program equality or source-success assumption is added.

Current evidence is SOURCE ONLY: five additive sources and nine SDK-free controls
(actual passes/service, mocked kernel/custody). Actual dead-i16 CSE changes retained
memref observer signature codes from opaque3 to opaque2 between independent
projections; the new common-universe route fixes the mapping obligation rather
than inferring a common Model. New queries, including finite-loop nonzero
success/error/initialization witness candidates, remain UNCOMPILED.
`compiler_pass_chain=not_established`; native/poison/numerical/model/review claims
remain unestablished. Old per-boundary service cut is unchanged.

Worked source/middle/final buffers, exact 48-root endpoint request, actual negative
observer query, field map/premises, five-source patch and complete source closure:
[chain-endpoint-v1](../development/chain-endpoint-v1/README.md). No retained Lean
checker output exists for these new queries; tests-v4-guarded.txt is host evidence
only. The runtime hold prevents closing that paper FIXME today. Existing old
qualified examples, if cited elsewhere, do not qualify these new modules.


## Larger scalar/SCF and canonical-prefix source profiles

New proposed coverage row and worked actual historical examples:
[larger-profiles-v1](../development/larger-profiles-v1/README.md). Frozen Pipeline
composition covers complete returned/error/world equality in the static no-tail
CSE/DCE→LICM→unroll→segment submode. Frozen Integer composition adds exact integer
normalization on canonical entries with local typed external-result contracts.
Historical engineering qualification belongs to those exact old source cuts.

The NEW gate request adapters, corrected source intake and added complete adjacent
entry roots are UNCOMPILED/UNQUALIFIED; five SDK-free tests are host evidence only.
Exact five/six buffers, one shared registry,24/30 project-source closures,57/82 roots,
field/semantic maps and historical checker TEXT are retained. No old receipt grants
new provider or compiler authority. No possible-success substitution, coverage
percentage, all-graph flag or native/numerical/poison/model approval.


## Registered sequential dependency schedule — new source-only row

[Schedule profile review packet](../development/schedule-profile-v1/README.md) and
[eight-kind coverage proposal](../development/schedule-profile-v1/EIGHT-KIND-COVERAGE-PROPOSAL.md)
retain actual pinned producer buffers and full-entry Scalar query/model obligations.
The region-free submode allows closed total primitives to reorder under complete SSA
matching and EXACT opaque call order; it assumes neither opaque commutativity nor
source success. Complete returns/errors/terminal-world relation is retained.

Two changed examples, one identity-order example, actual stateful observers and a
compiler-valid observer-reorder failure-world countermodel are retained. Six SDK-free
tests are host evidence;17 query roots/33 witness roots remain UNCOMPILED/outcomes
UNOBSERVED. Public scheduling unsupported/held; no existing service or qualified
ledger changed. This is NOT panel/double-buffer/gSIM verification or numerical/native
approval. Memory owner relations and canonical-integer premises remain separate.
