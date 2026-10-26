# Whole-entry roster successor — SOURCE CANDIDATES, 2026-10-26

**No new Lean theorem has been compiled or audited. All Lean outcomes remain
UNOBSERVED; no new qualification/replay/native result exists.** The global private
runtime hold is unchanged. This successor preserves `source-whole-v1` and
`whole-source-review-v1` byte-identically. It addresses their selected-entry scope,
not a new pass family or numerical semantics.

Primary: `/home/jeremy/Projects/merlin`, `feat/pass-verification`; observed HEAD
`9d7fe0846c43574ca6f83a5cccc5dc147b06ca6e`. Root alone publishes. No owned live jobs.

## Scope finding and proposed claim

The old root `After.checked_actual_chain` quantifies `kernel.function.name`.
Serializing surrounding functions does not establish their entry behavior. The
new candidate `After.checked_entry_roster` uses that selected-entry result and
finite-call-depth closed-submodule congruence to derive every listed definition's
possible successful observations. `GraphMemoryAfterEntries.every_definition`
also derives that every source definition has a row; exact source/final headers
and unique names prevent silently omitted, added or changed interfaces.

**Claim available today:** implemented uncompiled source candidates plus SDK-free
preparation/binding tests for the retained shape→reuse→segmentation boundary.

**Proposed claim only after compilation, root/axiom audit, fixed checks/replay and
independent review:** for this checked complete module boundary and its explicit
per-entry typed/caller domains, source and final have equal sets of finite
successful return/public-heap observations under the same declared partial
operation interpretation and total/private/unobservable allocation model.
This is not all pass inputs, compiler/native/hardware correctness, equality of
errors/divergence, numerical fidelity or universal allocator success.

## Complete actual roster

| Defined entry | Actual interface | Calls / proof route | Explicit input domain |
|---|---|---|---|
| `kernel` | `(i32,i32,memref<1xi32>) -> (i32,index)` | two calls to `touch`; derived selected-entry chain | typed arguments; caller canonical singleton descriptor outside private partition; external views outside that partition |
| `touch` | `(i32,memref<1xi32>) -> i32` | no calls; closed unchanged-body correspondence | typed arguments; descriptor conforms to complete type fields and allocation lies outside private partition |
| `untouched` | `(i32) -> i32` | no calls; closed unchanged-body correspondence | typed scalar argument |

All have default-public visibility; there are **no declarations** in the retained
positive. Exact visibility strings are bound as metadata, not native linkage or
access-control semantics. The new route explicitly refuses declarations instead
of dropping them from its coverage denominator. Missing entries and wrong arity
have no successful derivation in the declared model; the generic no-success
lemmas and emitted per-entry arity roots do not assert native error identity.

Heap contents are arbitrary for preservation; reads stay partial. Sample input
witnesses use a descriptor at allocation `plan.objects.length`, base/offset zero,
actual type shape/strides/element/space, and a heap whose cells contain Unit. The
kernel caller witness is explicitly constructed. Caller/caller aliasing is not
excluded. Private descriptor escape is excluded by the existing selected-kernel
return facts; unchanged entries retain their stronger complete-world equality.

## Actual semantic extension, not vacuous coverage

The previous fixed `finalMeaning` has no scalar `arith.constant` rule. Therefore
`untouched` had no successful execution in that interpretation. We do not count
its vacuous old result as nonvacuous whole-entry verification.

The successor declares a **shared partial, effect-free scalar literal relation**
indexed by the exact operation metadata key (including the textual literal and
result type). It does not assume that the value is numerically 1 or prove integer
numerics. `LiteralModuleFacts` checks zero inputs, one plain scalar output and no
internal-call collision. `KernelLiteralSupport` checks actual closed call sets
and absence of all literal keys from reachable kernel/touch actions in both raw
modules. `closed_literal_extension` derives that this extension leaves the old
selected-kernel interpretation unchanged. This is an explicit new interpretation,
not an edit or retroactive qualification of the old semantics.

`entry2_source_execution` constructs the literal/binary/return derivation for an
inhabited Unit interpretation, from the **entire actual source body**, selected
primitive rules and typed SSA facts. It takes no source-execution premise.
`entry2_final_outcome` transports that constructed observation through the new
roster candidate. All three entries receive inhabited typed-domain candidates;
only the unchanged literal/binary entry receives this new explicit execution
witness. Unit is an abstract nonvacuity witness, not a real i32 execution test.

## Call closure and refused context

`closed_invoke_agreement` inducts over every finite call depth. Actual action keys
select call targets; `ClosedEntry` derives each target's membership; lookup/body
agreement is required for the entire reachable set, including absent targets.
This handles unchanged closed call subgraphs without assuming callee equivalence.
The unchanged local body's syntactic equality alone is insufficient.

A valid unchanged wrapper calling the changed kernel reaches a body that is not
identical in the expanded module. Its explicit preparation refusal is
`unproved_changed_kernel_caller_context`; the complete finite obligation is still
retained. We do not replace it with an assumed public/full-world congruence.
Future support needs a contextual relation respecting private allocation clocks,
caller heaps, effects and aliasing, not middle-file hashes.

## Exact delivery and evidence

- `../source-entries-v1/`: immutable integrity-only source successor, **50 owned
  files**, consisting of the unchanged 41 plus **9 new** (5 Lean, 3 Python, 1 test).
  Static closure: **111 project Lean sources, 46 Python sources**. Counts overlap.
- `NEW-OWNED-SOURCES.json`, `OWNED-SOURCES.json`, `NEW-DECLARATIONS.json`, import
  graphs, closure manifests and `successor-source.patch` identify exact delivery.
  The additive patch depends on the unchanged old 41-source cut. No wholesale
  producer merge, shared-code edit or runtime object is included.
- `QUERIES.json` ties each of **12 distinct whole-entry queries** to its exact
  original raw source/final buffers, preparation report, full root list and roster.
  The positive proposes 248 support roots plus `every_definition`; these are
  prospective requested roots, **not audited roots**.
- `evidence/<case>/` copies raw `source.mlir`, `final.mlir`, `Entries.lean` and the
  complete preparation report. Each SHA was recomputed from the retained bytes.
  The old raw-buffer/whole-query association gap is closed **as data only**.
- `../tests-entries-v4/`: **12 SDK-free tests passed**, with all raw/query files
  retained and source bindings unchanged. Python/xDSL validates each input.
  A test forbids the old pair matcher from producing either boundary. This does
  not independently validate the parser/projection or execute Lean/MLIR.
- Cases: positive; untouched return, scalar literal, recursive callee, caller
  buffer, visibility, observer operands, selected kernel return, missing entry,
  external declaration, unchanged wrapper→kernel, parameter arity. Eleven are
  **binding mutations/domain refusals**, not proved semantic wrong-result cases.
  Their Lean outcomes are UNOBSERVED. In particular, earlier moved-free cases may
  be observationally benign in the total private allocator model; no blanket
  semantic-negative or detection-rate claim is made.
- Four optional Z3 dlopen attempts were blocked by the SDK-free test hook;
  two expected pytest timeout-plugin warnings remain. No SDK/runtime read or
  native proof/solver invocation occurred.
- `tests-entries-v1`: first setup failure, system Python lacked pytest; no tests
  ran. `v2`: 10 passed, then a missing closing brace in new wrapper test text.
  Both raw failures remain. `v3`: 11 passed; v4 adds arity/root/raw-hash checks.

`THEOREM-PREMISE-MAP.json`, `FIELD-MAP.json`, `ENTRY-DOMAIN.json`, `EVIDENCE.json`
and `OPEN-OBLIGATIONS.json` provide machine-readable review/navigation.
Source lexical and Python AST checks are not a Lean kernel/std3 audit.

## Preserved limits and next work

1. Runtime investigator/root must authorize a justified new context. All prior
   failed W profiles remain held; do not retry, rehash, repair or silently select.
2. Compile the exact 111-source closure and concrete query, correct ordinary
   diagnostics in a NEW cut, then inspect exact theorem types/transitive axioms.
   Both the old 31 new memory modules and these five entry modules are uncompiled.
3. Check the actual positive and changed finite bindings separately; retain first
   failures. Do not classify infrastructure failure as semantic rejection.
4. Freeze a new qualification and one fresh replay only after successful checks.
5. Graham/Prepare independently review parsing, all metadata/SSA/call/descriptor
   mapping, the literal extension, caller/allocator premises and claim wording.
6. Leave wrapper contextual congruence and external declaration models explicitly
   unsupported. No new broad family/numerical work is justified by this source cut.

The original SDK/runtime incident, first W v1/v2/v3 failures, historical capacity
and 8→16B allocation countermodels, elision crash, numerical/review/model limits
and all old qualified capsules remain unchanged. Historical Linux receipts are
not portability or laptop qualification evidence. Panel-pipelining/gSIM execution
figures are separate and are not implied by these candidate graph theorems.
