# Ordered unroll with a remainder: source-review candidate

This packet extends the modeled ordered-unroll boundary from the old **no-tail**
submode to an actual bulk loop followed by a nonempty remainder loop. It is
**UNCOMPILED / UNOBSERVED / UNQUALIFIED**. Five SDK-free tests passed, including
four actual changed producer captures and eleven compiler-valid output mutations.
Those are Python preparation observations, not Lean checker acceptance. There
are zero new native checks, root/type/axiom audits, qualifications or replays.

Primary: `/home/jeremy/Projects/merlin`, `feat/pass-verification`, observed HEAD
`9d7fe0846c43574ca6f83a5cccc5dc147b06ca6e`. Producer: read-only
`/home/jeremy/Projects/merlin-graph-proof-intake`, pinned
`4499eebb959c0f92cefd92cc6d80646ba66474ed`. Exact executed Python and project Lean
source identities are in SOURCE-INVENTORY.json; only source files were copied.
Root alone publishes. No SDK/runtime/archive/object access or native launches
occurred. F retains sole runtime recovery ownership. No owned jobs remain live.

## Exact proposed claim, conditional on future checking and review

For an admitted complete module pair emitted by pinned Merlin
`ordered_loop_unrolling`, acceptance of the generated complete-entry query would
establish equality of every defined entry's returned values, modeled fault and
terminal world. Missing-entry and wrong-argument-count behavior is also retained.
The pair uses ONE frozen Scalar Program/evalProgram interpretation, common
canonical descriptor registry, total pure word-primitive meaning, explicit
partial external meaning and concrete finite index iteration semantics.

The generic body equation in `Tail.ordered_bulk_tail` is a LOCAL obligation.
Generated `body_group`, `tail_body` and `suffix_same` declarations discharge it
against the actual decoded source and target syntax. It is not a free premise of
`merlin_unroll_tail_transform`. That final root has only common primitive and
External model parameters; External requires successful result arity, not success,
purity, totality, whole-body equality or whole-program equality. The unchanged
Scalar evaluator, including error cases, supplies all execution semantics.

This is a candidate claim, **not** a current paper correctness claim. The previous
qualified no-tail chain, earlier source-only adapters and all fixed packets keep
their original domains/statuses. Nothing transfers their qualification here.

## Admitted domain

* Static nonnegative lower/upper, upper >= lower, positive constant stride;
  signed index width 32 or 64, factor 2..4, trip count > factor and remainder > 0.
* One changed top-level loop per changed two-scalar-argument entry, one or two
  ordered carries, no region inside its body. Actual bulk and tail must be
  consecutive top-level loops. Unchanged entries remain in the complete roster.
* Complete scalar signless word/index modules; declared external calls only.
  Recursive/internal/nested-symbol calls, CFG and other regions are refused.
  Dynamic/negative control, inline/empty/no-tail forms and source index arithmetic
  requiring additional range evidence are not licensed by this profile.
* Pre-loop prefix must be supported total primitives or checked index controls.
  No effectful prefix is silently discarded. Arbitrary modeled partial operations
  in admitted loop bodies and final observers retain exact descriptors/order/world.
* Local argument/result arity and static slot checks do not establish arbitrary
  runtime value canonicality, successful execution, poison absence, native integer
  fidelity, alias completeness, allocator correctness or asynchronous completion.
  The witness provides success only in its explicit inhabited modeled domain.

## Producer form and actual field mapping

`unrolling.py:_unroll_one` derives `bulk_upper = lower + groups*factor*step`,
emits new stride/bound constants and the bulk SCF loop, clones each lane in order,
then (for a nonzero remainder) emits an SCF loop with lower=bulk bound,
upper=original upper, stride=original stride and iter_args=bulk results. The
replacement uses the TAIL results. `_repeat_body` forwards each cloned yield into
the next lane; the tail clones the original block with its own arguments.

| Actual IR field | Projection / emitted obligation |
| --- | --- |
| Complete module attributes, function/declaration roster, signatures and metadata | `project(source,target,context=(source,target))`; actual full producer replay and complete entry lists; no dropped declarations/functions |
| Typed operand/result/call signature, lexical SSA uses | Frozen scalar `block_code`, independent source/result stack indices; `typed2/typed3`, common registry |
| Original lb/ub/step constant operands | `static_controls`, exact closed index classifier, `controlAccepted`, `literalsAccepted`, `boundsChecked` |
| Bulk cutoff and factor-scaled stride | Actual target prefix values compared to derived controls; `form3`, `ordered_bulk_tail`, `constantsFit` |
| Lane offset/add and forwarded yields | Actual decoded B3 plus `body_group`; ordered partial calls are cased, not assumed total; `lanesFit` |
| Tail lb/ub/step and initial carries | Exact actual tail operands, initial vector `[0..carryCount)`, `after_bulk` |
| Tail body's arguments and captured outer frame | Actual BT, explicit `tailScopeL/R`, `Scope.check_sound`; extra bulk-result frame cells are not equated by SSA number |
| Tail and bulk failures | Frozen `loopEval`, absorbing fold/grouping and `Tail.loop_append`; same complete fault/world; no call after failure |
| Final return/observer and both loop results | Actual N2/NT, `suffixScopeL/R`, `suffix_same`; post-loop observer runs only after BOTH loops succeed |
| Surrounding entries, absent entry, wrong argument arity | Frozen Pipeline Entry/evalEntry, all entry signatures, `entries_cons`, final `PreservesEntries` root |

The Python producer replay validates host association with the selected production
implementation. It is not the operational soundness proof. The emitted typed SSA,
body grouping, scope and continuation obligations must still be checked in Lean.
Python/xDSL parsing, classification and IR-to-Lean correspondence remain disclosed
trusted mapping requiring independent review. No xDSL-to-Kernel/native bridge is
silently introduced.

## Proof reuse and new source candidates

Frozen roots reused: Scalar `evalProgram`, `loopEval_congr`, `loopEval_arity`,
`checkTypes_sound`, `check_operands_initialized`; Pipeline `unroll_loop`, packed
absorbing folds, concrete `indexModel`, `Scope.check_sound`, complete-entry roots;
W `CountCorrect.drop/aligned`, `grouped_repeat_iff`, bounds constants/lane/source
control. None of those sources was edited.

`GraphStatePipelineTail.lean` adds counted progression concatenation and an exact
`loopEval` append/bind lemma, then composes the existing no-tail grouping root
with the remaining original interval. Bounds remain actual checked data. The
proof does not expand execution proportional to the trip count. A bulk failure
absorbs the tail; a tail failure absorbs the continuation. No operation is
reassociated or commuted.

`GraphStatePipelineTailWitness.lean` supplies structural constant-carry success,
progression membership/length and first-fault-after-prefix lemmas. Generated
witness equations close their local body obligations by reducing the ACTUAL B2
syntax under a concrete provider. They are not successful-body assumptions.

Each Query.lean has 34 final/support roots and a 25-module project source closure.
REQUEST.json gives the exact statement, query hash, source closure and root list.
It is an inactive future request: provider is UNSELECTED_HELD, compiler eligibility
is always false, and constructing provider arguments raises. No default SDK,
minimal7/two-kind profile extension, saved report or plan-only result can authorize
compiler delivery. Future delivery requires the SAME live complete source/result
completion and checked actual boundary selection after new review/qualification.

The larger source closure still includes frozen GraphLoopsSegmentModel and its
historical executable-code/Repr dependency issue. This packet does not claim to
repair that issue, remove it from the closure, select a new runtime or resolve the
shared libleanshared incident. Elaboration, termination/tactic/linter diagnostics,
exact root types and transitive standard-three-axiom audits are UNOBSERVED.

## Actual cases and witnesses

Authoritative final captures: `final-actual-examples/`. Each contains exact raw
before/after modules, original registered CHANGE, independent projection/control
registry, complete Query, future REQUEST and explicit status. Earlier
`actual-examples/` is a retained pre-final source-development observation; it is
superseded, not a replay/qualification or a portability claim.

| Case | Source (lb,ub,step) | Factor | Index bits | Carries | Bulk / tail |
| --- | --- | --- | --- | --- | --- |
| trip5_factor2 | (0,5,1) | 2 | 32 | 1 | bulk [0,4) step2; tail [4,5) step1 |
| nonzero_nonunit_two | (3,18,3) | 2 | 64 | 2 | bulk [3,15) step6; tail [15,18) step3 |
| remainder_two | (1,17,2) | 3 | 32 | 1 | bulk [1,13) step6; tail [13,17) step2 |
| wide_index | (2147483650,2147483660,2) | 2 | 64 | 2 | bulk ends2147483658; tail retains original upper/stride |

Every case retains three external declarations, changed main and an unchanged
entry. The first case's Witness.lean has49 candidate root names and uses the
existing exact wrapping word primitive. The actual step echoes its nonzero i8
carry and increments world; the actual final observer echoes/increments world.
For inputs[3,5], initial world7, the proposed successful result is[3],world13.
A provider fault at source IV1 (bulk) proposes fault17/world9; at IV4 (tail) it
proposes fault17/world12, without running the final observer. Source results use
structural finite-fold lemmas and actual-body equations; target results use the
complete operational theorem. Reached-read initialization is a separate derived
candidate and does not mean poison freedom or native execution. ALL these
outcomes remain UNOBSERVED and model approval is absent.

## Bounded SDK-free evidence and retained failure

`final-tests.txt`:5 tests pass. They check actual production file identity,
complete common projection/source/root binding, actual changed case preparation,
unsupported forms, stale source/profile/domain/root reports and held authority.
Eleven xDSL-verified mutation outputs plus host refusal records are retained in
`final-actual-examples/mutations`: omitted tail, wrong cutoff, tail stride/upper,
added lane overflow flag, lane carry, tail carry, return, observer callee, tail
body capture, dropped surrounding function. They are valid actual IR with wrong
association/behavior; no kernel refusal was observed. These are not broad
statistical qualification results or native counterexamples.

First new failure: `first-sdk-free-failure.txt` plus the exact earlier test source.
The added flag mutation initially used an invalid xDSL constructor string and
failed before verification. It was corrected to the enum iterable; the final
mutation verifies before host refusal. The first failure is retained and is not
counted as detection. All older runtime, SDK, elision and numerical failures stay
preserved. No old native suite or replay was rerun.

Reproduce ONLY the SDK-free host tests with the explicit source paths and audit
hook (no native launch):

```
.venv/bin/python -I -B merlin/tests/ir/test_lean_graph_pipeline_tail.py \
  /home/jeremy/Projects/merlin /home/jeremy/Projects/merlin-graph-proof-intake -v
```

These absolute paths identify this Linux source observation. They are not laptop
qualification; a relocated checkout requires independently bound inputs/context.

## Delivery and remaining gates

Six owned new sources,111 copied source dependencies, source-only additive patch,
raw examples/queries/root inventory and this review packet are listed exactly in
SOURCE-INVENTORY.json and MANIFEST.json. PRESERVED-CUTS.json confirms141 endpoint,
77 service,186 larger-profile and127 schedule packet files byte-exact, plus
5 endpoint/4 larger/4 schedule live owned source matches. Other old frozen sources
were not edited. No public gate, protected semantic definition, production pass,
shared F/W source, numerical policy or lock changed. Root owns publication.

Before any paper acceptance claim: independent mapping/model/premise review;
a separately authorized justified runtime context; compile the new modules and
exact complete queries; check final/support root types and transitive std3 axioms;
qualify the fixed supported population/valid negatives and retain one fresh
replay; review same-live-result service binding/admission. Native/model/poison,
allocator/alias/completion and backend evidence remain separately scoped. The
panel/double-buffer/gSIM figure is outside this claim. There is no all-graph flag,
coverage percentage, policy approval or broader no-tail qualification transfer.
