# Scoped loop-configuration hoisting: paper evidence

Proposed claim: **For admitted complete Kernel before/after programs, moving one
invariant pure configuration setter out of a positive counted loop preserves
the complete modeled outcome, including the first failure and its terminal
state, under the explicit per-call interpretation below.** This is a conditional
graph-transformation claim. Python projection, primitive/effect fidelity and
human model approval are separate obligations.

## Theorem, existing roots and domain

`ActualConfigurationHoist.complete_kernel` applies
`Merlin.GraphState.Configuration.PartialHoist.check_sound` to the actual exported
programs. `complete_observation` covers arbitrary observers of complete outcomes;
`surrounding_context` covers a common partial prefix and suffix. Three new modules
extend unchanged `StateEffects`, `GraphState`, `GraphStateConfiguration` and
`GraphStateElision` roots. The earlier total `hoist_nonempty` alone did not address
retained-call failures; this partial-loop proof closes that gap.

The host domain is exactly one top-level loop with static extent 1..1,000,000,
at most 4096 static calls, exactly one call in the maximal invariant pure-setter
prefix, and a nonempty flat body. Calls have constant integer operands under
the actual schema/codec range checks. Body effects are known and ordered; all
writes/clobbers are disjoint from every setup-written field. Identical writes to
setup fields are also refused. Loop iter_args/yields/results and Call completion,
unit and staging metadata must be empty. Unchanged surrounding calls may fail or
observe state. Nested loops, multiple setters, IV-dependent setup and async or
pipeline cases are outside this cut.

`Outcome Value Error = Except (Error × State Value) (State Value)`. For every
initial complete state, the theorem equates `program run otherLoop before
initial` and `program run otherLoop after initial`. Its `Models` premise requires:

- A common indexed interpretation for each retained static call and iteration.
- Setup always succeeds with exactly the declared constant assignments, leaving
  every other state cell unchanged, with no hidden failure/observation/memory/
  pulse/token/completion effect.
- Successful retained calls obey their declared ordered effects and frame.
  Retained calls may fail, including after changing state. There is no successful
  source-run, total-body or whole-run-equivalence premise.
- A nonaliasing interpretation of field identities and complete representation
  of relevant caller/memory/return/error/observer state. Original instruction
  meaning must agree with these local contracts.

`otherLoop` is a common arbitrary interpretation for unsupported completion
loops; accepted templates force their completion lists empty. No asynchronous
semantics is invented. Adjacent-pass composition needs a common indexed
interpretation and compatible state/field map, which is not established here.

## Actual before/after worked example

The producer is `configuration_loops.py:hoist_loop_configuration` at exact
`e0484f4886ae0780d0325cdfa998842b61bb647c`. The root source manifest
`bb35450b5374e4fea125d0f766588830eea7a771131de7d1e961d5a4a4ea2626`
binds the original 2009-file Python roster. Original F capture and current
canonical source-bound transport are retained separately. There is no middle
program in this two-endpoint boundary.

[before.mk](before.mk) has an earlier unknown call, then three iterations of
`pair(a=3,b=5); work; work`, then a later observing `work` call.
[after.mk](after.mk) moves that pair setter immediately before the loop. Both
complete programs preserve input/output descriptors and attributes and pass the
original pinned Kernel parser/structural verifier. **These are Kernel `.mk`
programs, not MLIR or LLVM execution evidence.**

| Actual exported field | Lean representation / checked correspondence |
| --- | --- |
| Kernel name, every buffer name/shape/dtype/access, all attrs | Complete canonical `frame` equality |
| Every Call instruction, argument expression/value, annotations and flags | Full typed dataclass `token`; dense original static IDs; identical complete atom table |
| Loop variable, extent, body and completion fields | Explicit descriptor; exact source/target command templates and empty completion lists |
| Decoded operands and resource fields | Injective finite field registry and `Elision.Effect`; codec/field-to-state agreement stays trusted |
| Full earlier/setup/body/later roster | Exact dense-ID partition and order; complete changed-boundary templates |
| Indexed partial execution | `Models`, successful body frames, `ready_loop`, `check_sound` |
| Errors, terminal state, observers and context | Complete outcome equality, `checked_observation`, `context_preserved` |
| Original bytes, query, sources/objects and audit | Canonical candidate/receipt binding, same-live consumer, fresh replay |

The exact [Lean query](ActualConfigurationHoist.lean), [checked report](CHECKED-REPORT.json)
and [binding record](BINDINGS.json) are retained byte-for-byte. Baseline checks
15 exact root types and 315 transitive theorems with only `propext`,
`Classical.choice`, `Quot.sound`. Before/after SHA256 values are respectively
`0266a1d9d2b679920b47f97885c23e8bc5d5993f5ead955221cdf67942d27213`
and `fc1a887a63148fa8659223b552e50e0dc588b596f10543fcb343838808f08bf9`.
Query SHA256 is
`19392427c2a2917fc51e79da9877d2441229b0f0b9cdbe0f64bc74c12da5c62a`.

An independently rebuilt actual-query witness checks 124 root types and 345
transitive theorems using F's own checked objects, explicitly auditing both
custom modules. It derives local contracts from the actual atom table: success
has observer61/trace1234567; selected first body error73 has observer37/trace1234;
prefix error91 retains initial state before setup. These are constructive model
proofs, not native hardware or numerical samples.

## Qualification, negative example and limits

Q5/F1 passed seven actual changed pairs, nine explicit unsupported boundaries,
fifteen structurally valid mutated pairs rejected by the exact
`ActualConfigurationHoist.accepted` Lean obligation, and one valid token-change
pair refused by the host scope guard. Control/descriptor mutations test the
admitted correspondence; not every one is a semantic counterexample outside this
checker. Extent1 is a textual hoist with no saved dynamic executions or speedup
claim. [negative-wrong-value.lean](negative-wrong-value.lean) changes the final
setter value; its [raw output](negative-wrong-value.stdout), stderr and process
record retain the bound acceptance-root closed-false diagnostic and ordinary
exit1. Infrastructure failures are not counted as proof refusals.

Sixteen forged candidates were refused before native output. Twelve genuine
live-artifact controls cover originals/source manifest/snapshots/query/objects/
raw log/receipt/membership/symlinks; constructed/copied/saved tokens and copied
candidate also refuse. Fresh separate-process replay started with an empty
ledger, rebuilt/audited baseline and issued one new completion. Source postguards
passed. Sixteen F source tests, owned lint and repository structure checks passed.
No native infrastructure failure occurred in this campaign.

Preserved counterexamples include zero-trip/hidden-setter-pulse models and a valid
loop-token counterpair accepted by the production rewrite verifier. That last
finding justifies independently refusing completion metadata; it does not refute
the optimizer's generated output. Five invalid Kernel controls and 192 finite
state/error comparisons are supplemental tests, excluded from formal-proof and
valid-mutation counts.

Root source review/engineering qualification do not establish independent human
policy approval. Synthetic fixture declarations have a parser-required
`reviewed` label but remain **unapproved test premises**. Python import delivery,
Kernel parsing/projection, codec, local primitive fidelity and field identity
remain explicit trust boundaries. No hardware/numerical result, compiler-wide
enforcement, complete model coverage, pipeline theorem or automatic adjacent-pass
composition is claimed. Absolute Linux receipts are historical execution evidence;
a laptop needs new local replay, not a portability relabeling. Old cuts and first
failures remain preserved. Original preparation prose is retained in PREPARATION.md.
