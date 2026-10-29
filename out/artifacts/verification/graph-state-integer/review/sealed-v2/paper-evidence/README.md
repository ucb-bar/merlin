# Exact-integer normalization in complete graph context

**Proposed paper sentence, pending independent review:** For the enumerated
unflagged signless scalar-integer boundaries and supported scalar/SCF chains,
our proof-producing validator establishes equal complete entry outcomes—returned
vectors, external errors and terminal worlds—on initialized canonical word inputs
under explicit local operation contracts. Runtime representation of reached
intermediates is derived from those local contracts and typed SSA, rather than
assumed globally.

ENGINEERING QUALIFICATION COMPLETE for corrected-source v2: **75 tests,17
original receipts (14 changed/3 identity),ONE retained fresh-process batch of17
replays, exact postchecks; all transitive standard-three audits clean.** No
review/native/poison/model/full-graph approval follows. Exact bindings are in
EVIDENCE.json and qualification-v2/PLAN.json/BATCH.json.

Qualification-v1 remains an immutable UNQUALIFIED host-classification diagnostic
(66 tests/17 originals/17 replays measured, ZERO final qualified originals/replays).
Its exact source/controller/raw files are preserved; its historical BATCH label
does not override CLASSIFICATION_HOLD. V2 corrects a known Python guard source
bug and adds nine MOCK process tests, under the same original SDK bindings;
no unknown native context was retried, repaired or rehashed to promote it.

## Registered pass, actual source and existing anchors

The registered kind is `exact_integer_simplification`, actual producer
`transforms.normalization.simplify_integers` at read-only pinned
4499eebb959c0f92cefd92cc6d80646ba66474ed. It folds unflagged signless scalar
add/sub/mul constants modulo width and removes both +0 sides, right -0 and both
*1 sides. Earlier replacements may expose later rewrites; the independent
checker follows actual complete operand bindings, not optimizer witness counts.
The producer's witnesses remain diagnostics only.

Existing anchors are `Scalar.normalize_sound`, `checkTypes_sound`, `check_sound`,
`check_composes` and `typed_operands_initialized`; sealed scalar-loop complete
correspondence and the W counted/signed-bound/scope/failure bridges are reused
READONLY. The older local integer broker/Lean-MLIR semantics is not assumed equal
to this operational model. This route uses existing core BitVec facts directly;
no Foundation/Lean-MLIR, xDSL↔Kernel or old numerical-specification bridge is assumed.

New kernel roots are `Integer.typed_result`, `typed_reached`, `Relation.sound`,
`programs_sound`, `entries_cons`, `compose_suffix` and `index_model_typing`.
Generated roots are `Merlin.GraphStateIntegerInstance.prefix_actual`, per-function
`F#.actual/reached/localRewrites`, and composed `complete`, `originalRegistry`
and `exactMiddle`. The prefix and sealed CSE/DCE→LICM→unroll→segment suffix share
one actual registry, word primitive, external meaning and concrete index model.
`exactMiddle` checks the full normalized entry roster/programs; JSON success or
independently numbered cells are never equated. This is component composition,
not a separately claimed direct original→final validation algorithm.

## Numerical carrier and complete relation

For a width w, `word w x = Int.ofNat ((BitVec.ofInt w x).toNat)`. A word value is
canonical exactly when `0 ≤ x < 2^w`; signless bit patterns use unsigned Int
representatives. Signed source/result literals are interpreted by the same
BitVec.ofInt conversion, so signed canonicalization preserves exact bits.
Core ofInt_add/ofInt_neg/ofInt_mul and word idempotence prove constant folding;
neutral rules require a proved canonical retained expression. No bitwise/native
agreement is presumed. Selected pure widths are1..256; actual examples cover
1/8/16/32/64/128/256. i1's signed one-bit literal is -1, so its actual producer
coverage is not mislabeled as all +1 neutral rules; i8+ exercise those rules.
Index, tensor, floating and overflow-flagged arithmetic are not integer rewrites
in this theorem. Unchanged region-free operations may remain explicit opaque
steps; unsupported changed structures are refused.

The unchanged Scalar Model/Program/Code evaluator uses initialized Int values
and an arbitrary persistent World/Error. Pure word primitives are explicitly
total, deterministic and state-free. Retained operations have the SAME complete
descriptor, ordered operands and full world transition, and may fail. Successful
opaque results require LOCAL arity and representation contracts on correctly
typed inputs. No global intermediate-typing, source-success, bulk-body agreement
or whole-final-equality premise is an input to preservation.

`PreservesEntriesCanonical` reuses the unchanged dispatcher. It compares full
successful return vectors and persistent worlds, or the same modeled fault/error
and terminal world. Missing entries and wrong input arity remain observable and
preserved; correctly sized entries require canonical input values. Private SSA
stacks may differ. Costs/timing, printer spellings, debug locations, compiler
diagnostics and hidden observers of deleted private SSA representation are not
observations. Retained effect/call order is preserved, so no hidden independence
hypothesis is inferred. Physical meaning/completeness of the world remains a
separate effect/alias/allocator/completion contract.

## Static typing, runtime typing, success and poison are distinct

- Old `Typed`/checkTypes discharge static slots, lexical availability and arity.
- Old `Initialized` proves reached SSA reads resolve; it does not type values.
- New `typed_reached` derives runtime operand/result/capture/carry canonical
  representations from canonical entries and LOCAL result contracts, including
  nested loops. Failure paths stop; their later instructions are not assumed to run.
- `typed_success` derives success from typed syntax and LOCAL total-on-typed
  operation behavior. Only the explicitly named provider witness supplies that
  local total model; general preservation continues to allow failures.
- The mathematical Int carrier has no poison constructor. A faithful initialized,
  no-poison entry/provider/native representation is a separate domain/fidelity
  contract. Neither static typing, tests nor this carrier proves native poison
  freedom, universal successful execution or primitive/model approval.

The explicit provider witness uses canonical nonzero inputs[3,5], initial world7,
ordinary wrapping word functions and an arity/representation-correct constant-word17
provider. Both ACTUAL original and ACTUAL final programs return[17],world32.
A partial-provider first-observer fault gives external error17/world8 on both.
Success/failure is derived through the actual syntax and sound composition; no
source-success assumption is supplied. The loop proof uses a structural
constant-result theorem and derived range equality, not literal dynamic trace
expansion. The old literal execution timeout remains unqualified and unretried.

Preserved Lean countermodels show: i8 input256 makes x+0 wrap to0 while deleting
it returns256; a statically typed/arity-correct provider returning256 causes the
same failure; an incorrect primitive interpretation breaks neutral deletion;
and a hidden physical world update breaks deletion even with the correct word
value. These explain necessary premises, not a counterexample within the admitted
canonical state-free word domain.

## Exact compiler-IR → Lean map and trust boundary

| Actual field | Checked interpretation and authority |
| --- | --- |
| Module and complete function/declaration roster, attributes, interface | Trusted structural Python matching binds all before/after headers; Lean entry roster/signatures/dispatch are explicit. Untouched functions are checked, not omitted. |
| Operation class, overflow flags, signedness, literal, width | Closed actual-operation whitelist reconstructs Prim; kernel width/arity/type checks and BitVec local rules justify the value computation. |
| Every ordered SSA use and result, argument owner | Python reconstructs independent lexical indices; kernel checkTypes/normalize/lookup/shift proofs bind each use. No cross-program SSA-number equality. |
| Nested region/block arguments, captures, carries, yields, continuation | Program.loop and checked scoped Code bindings; operational typing, local relation recursion and failure-absorbing loop proofs preserve them. |
| Call target and declared input/result signature | Python requires exact declared external nonnested symbols/types; Lean fixes descriptor/order and derives result representations from LOCAL successful-result contracts. Internal calls/fuel are refused. |
| Pure dead computations | Old SSA normalization removes only admitted total/state-free primitives. Their original syntax and static typing remain in the kernel-checked Program; partial/opaque computations remain retained. |
| All six complete buffers and joins | Common registry/type table; checked raw-program normalizations, local rewrite sequence, exactMiddle and the unchanged suffix theorem under ONE interpretation. |
| Source/tool/proof/semantic identity | Strict v1 receipts bind originals/snapshots, model, query/object, exact roots/statements/imports/audit, source/profile/runtime bindings. Fresh replay independently reparses originals and reconstructs/checks/audits the query. |

Python parser/projection, host/process/hash/path/report transport and identity
provenance remain trusted. Lean proves the local certificate rules and whole
operational relation, not arbitrary Python status strings or an unverified
production optimizer implementation for every input. Receipt hashes do not
prove issuer/archive authenticity or actual runtime delivery. External pinning
of original identity is supported and used by qualification replay.

## Domain, actual coverage and adversaries

Standalone arithmetic prefixes admit the existing supported structured SSA/SCF
projection with checked positive constant stride and a common finite iteration
interpretation, including nested/zero-trip/captured/two-carry cases. No CFG,
unknown-region, internal-call or arbitrary divergent native semantics is hidden.
The composed suffix keeps its SEALED narrower domain: one changed static
nonnegative top-level loop, two inputs, positive stride, no-tail bulk count>
factor divisible by factor, factors2..4/tiles2..16,1..4 carries and checked
signed32/64 generated-control bounds. No silent expansion of the sealed suffix.

The frozen v2 roster is75 tests and17 original receipts:14 changed and3 unchanged
flagged/index/floating discriminators. Four six-buffer cases change EVERY pass
(three independent width/factor profiles plus the explicit provider mode).
Width257 changed input is refused, not silently dropped from the denominator.
Twenty compiler-valid adverse edits cover constants, later-exposed sites,
operands/order/opcodes/flags, signatures/width, captures/carries/yields, observers,
callees/error behavior, returns and unrelated context/headers. Four malformed
operand/result/region/call cases remain separate. Wrong-middle and false bit
certificates are rejected. Ten coherent fresh-rehash attacks cover query/object,
header/projection/model, audit/import/statements/profile and original IR.
Every attack owns mutable candidates and preserves exact before bytes; the
original query/object is read-only. Forgeries do not rerun duplicate originals.

The corrected v2 host guard admits ONLY a structured known input-domain refusal.
Signals, other exits, setup/unknown errors and missing/malformed payloads are
UNAVAILABLE and hold the context. Nine MOCK process tests exercise this distinction
without causing real native faults/SDK writes. Timeouts/signals/setup failures
are never counted as proof refusals. This source fix is why v1 stays diagnostic.

## Qualification, worked example and remaining review

EVIDENCE.json contains repo-relative exact original/middle/final/query/object/
checker/audit/negative/replay bindings. WORKED_EXAMPLE.md annotates an actual
five-changed-pass i32 chain alongside unabridged buffers, with readable excerpts.
The frozen suite supplies the SOLE17 measured originals, then ONE retained
fresh-process batch replays them. The duplicate positive development replay is
excluded; negative fresh forgery probes are distinctly counted. Each positive
replay reconstructs the query, reproduces the object at its exact original
source path into private output, fresh-checks/audits and postchecks frozen pins.
Only standard3 axioms; no reflection/new axioms/sorry/admit/native_decide/bv_decide,
implementation substitution, downloads or SDK repair. Full Lean-MLIR/broker
selection remains unavailable. Broad structure checking observed a FOREIGN
zephyr_model.py size-ceiling failure; owned Ruff checks pass, no foreign edit.

Graham's independent mapping/premise review is pending. Ordinary compilation
receipt gating, actual retained-model capture selection, faithful word/opaque/
index/native representations, no-poison/provider delivery, physical effects/
alias/allocator/completion, native LLVM/OOT/ABI/hardware correctness and the
remaining shared-graph denominator are separate obligations. Selected model
coverage is NOT asserted. The gSIM panel-pipelining figure is separate execution
evidence, not proved by these theorems. No all-graph, numerical, policy or full-model
approval is issued. Coordinator alone owns staging/commits/push/integration.

All75 earlier owned sources/capsules/patches remain read-only. UNKNOWN Std.Time
binding incident and elision SIGSEGV (ZERO qualification), old monolithic/literal
timeouts and all numerical counterexamples remain. Native pow→mul equality stays
refuted; graph typing cannot repair it. SPEC_LOCK remains unchanged. Old absolute
Linux receipts are historical on a laptop; no portability/qualification fix is
asserted by copying this source/evidence packet.
