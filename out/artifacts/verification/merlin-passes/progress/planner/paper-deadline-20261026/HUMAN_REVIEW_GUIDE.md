# Jeremy's code and paper-claim review guide

Review cut: 2026-10-26. This is a review procedure, not a correctness certificate
or completed human sign-off. Start with the compiler-to-proof connection, then
read the theorem and its semantics. Proof counts alone cannot answer what the
compiler has verified.

The deadline target is **conditional correctness of specifically enumerated
graph-transformation validations**, accompanied by a real before/after example
and explicit assumptions. Numerical correctness, LLVM-owned transformations,
hardware correctness and whole-model coverage are separate obligations.

Start here: [current assignments](README.md),
[scalar model](../../../../../../../src/merlin/verify/lean/Merlin/GraphStateScalarContract.lean),
[new compiler gate](../../../../../../../src/merlin/verify/lean_graph_pipeline_gate.py),
[shared input admission](../../../../../../../src/merlin/verify/lean_graph_input_admission.py),
[worked pipeline example](../../../../graph-state-pipeline/paper-evidence/PIPELINE_WORKED_EXAMPLE.md),
and [qualified integer packet](../../../../graph-state-integer/review/sealed-v2/README.md).
The new gate/admission files are active candidates, not historically qualified
sources. The sections below explain how to review them.

## 1. Freeze what you are reviewing

The published verification baseline is `feat/pass-verification` at
`58dbc7c00acdb6770d440b773cc2954328c78472`. There are intentional uncommitted
development changes. A file visible in this checkout is not necessarily in that
commit or covered by its historical qualification.

There are three different source cuts:

| Cut | Where | What it establishes |
| --- | --- | --- |
| Original actual producer | `/home/jeremy/Projects/merlin-graph-proof-intake`, `4499eebb959c0f92cefd92cc6d80646ba66474ed` | The production passes used to generate historical examples. Keep read-only. |
| Corrective producer | `out/build/graph-paper-deadline/producer-corrections`, `fix/graph-paper-gate-capture`, based on that same revision | New gate/capture fixes. Until targeted integration and qualification, this is a separate candidate. |
| Verification implementation | This checkout, `feat/pass-verification`, baseline above plus explicitly recorded patches | Standalone validators, Lean models/proofs, new admission and gate work. |

Graham reviewed `origin/feat/pass-verification-verf` at
`5a7f872db33d378d6b0c4a294e4099e80efd4956`, based on the published verification
baseline. His deliverables 1–3 are review evidence, not an approval of subsequent
fixes. Read them without changing branches:

```sh
git rev-parse HEAD
git status --short --branch
git diff --stat
git show origin/feat/pass-verification-verf:out/artifacts/verification/graham-graph-review/deliverable-1/gaps.md
git show origin/feat/pass-verification-verf:out/artifacts/verification/graham-graph-review/deliverable-2/omissions-bugs.md
git show origin/feat/pass-verification-verf:out/artifacts/verification/graham-graph-review/deliverable-3/assumption-review.md
```

`git diff` excludes untracked files. For a candidate review, require an exact
owned-source inventory/snapshot or commit, including new files, and identify its
qualification packet. Do not infer a source cut from a dirty branch name. Review
the **new corrected version** separately from preserved old receipts and failures.
The design document `docs/design/merlin_lean_verification.md` describes intended
architecture; its broad introductory goals are not a current coverage result.

## 2. Use this review order

Reserve the first 2–3 hours for sections 3–5 and one worked example. Then review
each additional family you intend to name, its premises and its packet. A careful
review across families can take another day; sampling one family cannot approve
the others. Jeremy and Graham can split families and independently check the
same gate and one example. Agustin's specification/policy review remains a
separate recorded decision.

For **each proposed paper sentence**, write down these eight items before
approving it:

1. Exact transformation variants and syntax accepted, including compositions.
2. Actual submitted source, middle and final program identities.
3. Final Lean theorem name and full type, with quantified inputs and domain.
4. Observable behavior preserved: results, errors, world, successful outcomes,
   or selected observers. State which relation this family proves.
5. Every semantic premise and every trusted correspondence step.
6. Fixed source/tool/parser identity, original qualification and fresh replay.
7. Counterexamples, unsupported variants and remaining obligations.
8. Human reviewer, decision, exact reviewed cut and permitted paper wording.

Use the worksheet in section 10. An empty field means the corresponding claim
is still open. A test result, proof of a local law and proof of a whole changed
boundary belong in different fields.

## 3. First inspect the live compiler gate

This is Graham's highest-impact finding, D1-G09/D3-G10. The original producer's
`src/merlin/transforms/pipeline.py` calls validation and checks
`accepted_pipeline_receipt`. Its original `src/merlin/verify/lean_checker.py`
could grant completion authority using a shape identity theorem or local integer
equations. Those do not establish the changed program's behavior.

Read those files in the **corrective producer checkout**, starting with
`validate_change`, `verify_pipeline_local_obligations`,
`accepted_pipeline_receipt`, and the pipeline's `run()` path. Then read the new
primary files `src/merlin/verify/lean_graph_pipeline_gate.py`,
`lean_graph_pipeline_gate_projection.py`, `lean_graph_pipeline_gate_emit.py`,
`lean_graph_pipeline_gate_native.py`, and
`merlin/tests/ir/test_lean_graph_pipeline_gate.py`.

Trace one actual call, without jumping from a function name to its intended
meaning:

```text
actual pass input/output
  → admission and complete semantic projection
  → deterministic query containing the actual program terms
  → Lean kernel checks the exact preservation roots
  → completion binds query, program, model/domain/premises and source identity
  → the live compiler requires that completion before proceeding
```

Ask: can a JSON dictionary, old local receipt, identity lemma, callback, stale
completion, changed query, changed source or mismatched domain authorize this
path? Is missing support a fail-closed result? Does an actual changed program
reach the complete-program theorem? Is unchanged code covered too?

As of this guide's creation, the gate worker has a source-level fail-closed
repair with SDK-free regression evidence. The new native gate is deliberately
held; **a genuine new Lean-accepted compiler-gated positive is still required**.
Mocks establish Python behavior, not kernel acceptance or integration
qualification. Recheck the latest gate handoff before changing this status.

Require a retained real changed positive and these separate negatives: a valid
wrong-result program, an old/local-only completion, missing evidence, wrong
source/domain, and a forged/stale completion. A malformed IR parse failure is
useful, but cannot replace the valid wrong-result case. Native crashes, timeouts
and setup failures are infrastructure failures, not successful semantic refusals.

Until this path has real qualified evidence, use **standalone-validator** wording.
The mere existence of stronger Lean leaves does not make ordinary compilation
use them.

## 4. Review the IR-to-Lean mapping

Start with `src/merlin/verify/lean_graph_state_scalar_capture.py` and the selected
family's capture/projection/emitter. Follow raw submitted bytes, parsed IR,
projection and generated Lean definitions. A hash binds bytes; it does not prove
that those bytes have the meaning assigned to the generated term.

For one complete example, account for the function/declaration roster, argument
and result types, all operation attributes/properties, ordered operands/results,
SSA definitions/uses, block arguments, captures, nested regions, carries/yields,
return values, unchanged functions and post-transformation observers. Mark each
field as represented in Lean, checked by a trusted host matcher, deliberately
unobservable, or unsupported. No field may simply disappear without explanation.

Read the new shared `src/merlin/verify/lean_graph_input_admission.py`, its capture
entry hooks and fixtures in `merlin/tests/fixtures/graph_input_admission/`.
Its intended interface is `admit_text`, `admit_module`, and `parse_admitted`.
Review both raw-text and parsed-module paths, including supplied-buffer APIs.

Two concrete review findings must remain visible:

| Finding | Why it matters | What to inspect in the fix |
| --- | --- | --- |
| D2-G01: function values/indirect calls and hidden references to defined functions | An unsupported internal computation could become an arbitrary opaque operation. | Recursive symbol/attribute/type inspection; rejection of function-value/indirect-call paths; real external-call positive controls. Shape's narrow `shape-direct` mode must preserve only its already-proved typed direct internal calls. |
| D2-G02: file-level resource data lost by parse/print | Two different resource payloads could have the same captured module and query. | Original text checked before parsing/reprinting, plus structural resource checks; original/supplied-buffer/capture-only routes; comments and quoted strings do not fool detection. Refusal of resources is an acceptable explicit domain restriction. |

The corrective producer also needs its `certificates.py` guard before information
is lost. Check identical helper identity where it is copied into both checkouts,
and inclusion in the **new** source closure. Historical seals stay unchanged.

For shape and memory-SCF, read `GraphLoopsShapeModel.lean` and
`GraphLoopsMemorySCFModel.lean`. The historical model uses a common projected
blueprint with source/target modes. Python establishes the actual final program's
correspondence. Nonempty binding strings or equal digest strings do not move that
semantic obligation into Lean. This is D2-G03.

The W worker is adding a separately projected actual-after term and a Lean
bridge, initially for a supported shape subset. Read its **final** handoff and
new `GraphLoopsShapeAfter*`/`lean_graph_loops_shape_after*` modules when ready.
Require an independently changed, compiler-valid final body/operand to change
the projected term and fail the relation. A new shape bridge does not
automatically upgrade the older memory-SCF or composed adapter claims.

## 5. Read the final theorem and the definitions it mentions

Open the retained generated `.lean` query and exact printed theorem type for
the selected receipt. Start with its final root; follow its relation and model
definitions, then the component lemmas. Do not begin by counting helper lemmas.

For scalar/pipeline work, read `Outcome`, `Model`, `evalProgram`, `loopEval` and
`Preserves` in `src/merlin/verify/lean/Merlin/GraphStateScalarContract.lean`.
The model's effect operation may fail and carries a persistent world. Its local
successful-result-arity law is a premise. Its pure primitive interpretation is
total, deterministic and state-free. Establish how actual operations are
classified into those categories; native behavior does not follow merely from
the word “pure.”

Check quantifier order: are the same interpretation and world used throughout?
Does the theorem cover all admitted input vectors or only a selected constant
example? Is entry initialization derived? Are local typing obligations proved
from explicit contracts, rather than assuming all intermediate values are well
typed? Is source success assumed? Does a premise already assert the entire
source/target equality the theorem is supposed to establish?

For composition, inspect each complete boundary and the **same actual middle**.
Compatible file hashes alone do not connect different semantic relations or
domains. Read the transport/composition theorem and confirm it supplies every
required domain/interpretation/observer obligation. Distinguish a constant
operational witness from a domain whose behavior depends on the input; neither
is an actual-model coverage proof.

`#print axioms` answers whether a root uses axioms, including `sorryAx`; it does
not establish that its type expresses the intended compiler property. Audit the
exact final roots and their transitive dependencies. This project's accepted
standard axioms are `propext`, `Classical.choice` and `Quot.sound`; no new axioms,
`sorry`, `admit`, native decision shortcut or substituted implementation is
permitted. Passing an audit does not discharge ordinary theorem hypotheses.

## 6. Family-by-family code reading map

Paths below are relative to the repository. Lean files are under
`src/merlin/verify/lean/Merlin/`; Python files under `src/merlin/verify/`.
Read frozen query/source copies when reproducing a historical result; current
files may contain newer changes.

| Family | Read first | Exact review focus and current boundary |
| --- | --- | --- |
| Scalar CSE/DCE and LICM | `GraphStateScalarContract.lean`, `GraphStateScalar.lean`, `GraphStateScalarSound.lean`; `lean_graph_state_scalar_capture.py`, `lean_graph_state_scalar.py` | Equal returned vectors and complete success/error terminal world in the common model. Check initialized SSA, retained effects/order, external declarations and unsupported control/calls. Standalone scope does not imply live gate use. |
| Scalar → unroll → segment | `GraphStatePipelineProgram.lean`, `GraphStatePipelineControl.lean`, `GraphStatePipelineScope.lean`, `GraphStatePipelineLoops.lean`; `lean_graph_state_pipeline_capture.py`, `lean_graph_state_pipeline_receipt.py` | Generated final root `Merlin.GraphStatePipelineInstance.complete`. Qualified submode: one changed top-level static nonnegative loop, positive stride, no-tail bulk, factor 2–4, tile 2–16, 1–4 carries, two inputs, region-free original body. Read exact control/range/index-bound bridges and failure absorption; generated nested regions are covered. Empty/dynamic/tail/inline/general nested source loops are not thereby covered. |
| Exact integer normalization and supported chain | `GraphStateIntegerProgram.lean`, `GraphStateIntegerRepresentation.lean`, `GraphStateIntegerTrace.lean`, `GraphStateIntegerCode.lean`, `GraphStateIntegerCountermodels.lean`; `lean_graph_state_integer_capture.py` | Canonical word input domain; unflagged signless scalar wrapping add/sub/mul folds and identity rules. Read `EntryDomain`, `PreservesEntriesCanonical`, `programs_sound` and reached-typing derivation. Numerical floating-point, poison, overflow flags and physical index representation are outside this claim. |
| Shape normalization | `GraphLoopsShapeModel.lean`, `GraphLoopsShapeModule.lean`, `GraphLoopsShapeSound.lean`, `GraphLoopsShapeCountermodels.lean`; `lean_graph_loops_shape_capture.py` | Historical `checked_complete` relates possible complete successful outcomes of two model modes, with actual-after correspondence trusted to Python. Review function/call depth, descriptors and memory interpretation. Read the new actual-after bridge separately when delivered. Do not relabel this as scalar-style error/terminal-world equality. |
| Reuse → memory-SCF segmentation | `GraphLoopsMemorySCFModel.lean`, `GraphLoopsMemorySCFOperational.lean`, `GraphLoopsMemorySCFComposition.lean`, `GraphLoopsReuse*`; `lean_graph_loops_memory_scf_capture.py` | Common interpretation, exact initializer/middle binding, safe lifetime/alias geometry, possible public successful outcomes. Total/private/unobservable allocation is assumed; finite-capacity or allocation-failure preservation is not proved. Preserve the 8→16-byte resource counterexample. |
| Shape → reuse → segment adapter | `GraphLoopsShapeEpochEntry.lean`, `GraphLoopsShapeEpochFunction.lean`, `GraphLoopsShapeEpochChain.lean`, other `GraphLoopsShapeEpoch*`; `lean_graph_loops_shape_epoch.py` and its projection/selection modules | Development composition and complete caller-observation adapter. Existing development tests do not constitute frozen qualification: the freeze stopped before originals/replays. Check domain/descriptor/private-epoch policies and new actual-after coverage explicitly. |
| Declared calls / fusion | `GraphRegionsDeclaredBridge.lean`, `GraphRegionsDeclaredMemory.lean`, `GraphRegionsDeclaredRaw.lean`, `GraphRegionsDeclaredCountermodels.lean`; declared-call projection/receipt modules | `checked_calls_preserve_observers` retains instruction contracts, initialized packing, physical Action frame/locality and ABI publication premises. The native Action's frame is not derived from machine code. The stronger after-binding/captured-source candidates have separate incomplete qualification. |
| Dependency scheduling and wider mechanisms | `GraphStateSchedule.lean`, `GraphStateScheduleCountermodels.lean`; schedule checkpoint/handoff | Development only. Individual compiled lemmas or one scalar example cannot qualify wider scheduling, buffering/prefetch/pipelining, staged dense rewrites or the user's gSIM figure. Put unsupported rows outside the paper's proved set. |

Historical engineering packets report 64 tests/eight originals/eight replays for
the four-pass chain and 75 tests/17 originals/17 replays for corrected integer v2.
Standalone shape has 78 tests per original/replay batch, memory-SCF 71 tests per
batch, and declared-call v1 46 tests per batch. Those are NOT counts of positive
original receipts: declared-call STATUS cites two actual original receipts.
Derive distinct positive receipt/replay counts from the raw fixed rosters;
do not infer them from pytest totals. Consult each roster for exclusions and
positive, negative and protocol-test categories. Counts overlap in dependencies
and **must not be summed as passes proved**.
The newest shape adapter and frontend capture extensions remain development
work; do not inherit their parents' historical counts.

## 7. Walk through one real example end to end

Use `out/artifacts/verification/graph-state-pipeline/paper-evidence/README.md`,
`PIPELINE_WORKED_EXAMPLE.md` and `EVIDENCE.json`. They identify a complete
five-buffer CSE/DCE→LICM→unroll→segment example and retained checker output.
Alternatively use integer v2's `paper-evidence/EVIDENCE.json:worked_example` for
the six-buffer integer-prefix chain. Prefer one example that changes every
claimed stage and retains an observer after the loop.

1. Open every **actual original buffer**, not only the excerpt in the paper.
   Annotate each changed operation, carried value and unchanged continuation.
2. Follow the exact raw pair/chain into its projection and generated program
   definitions. Locate the actual body/SSA/call/observer facts in the query.
3. Read the final root's printed type and exact-root/transitive audit output.
   Explain in plain English what it says for this example and arbitrary admitted
   inputs under its model. Identify each remaining premise.
4. Follow source/query/object/audit associations into the original receipt,
   batch postcheck and one fresh replay. A boolean acceptance field is not this
   chain of evidence.
5. Read a retained compiler-valid wrong-result mutation and its refusal stage.
   Inspect a changed-middle and a coherent query/object forgery too. Record
   whether refusal came from host admission/projection or Lean proof checking.
6. Write the paper's worked example using the actual source cut, a short genuine
   IR excerpt, exact root, bounded checker-output excerpt, relation and premises.
   The panel-pipelining/gSIM measurements demonstrate a different property;
   they cannot stand in for this proof example.

If the referenced originals or raw outputs are absent, mark the packet
**unavailable for independent replay**. Ask for the immutable original bundle;
do not silently regenerate it in a new environment and call that historical
replay. Graham found incomplete delivery despite locally complete packets.
Five [original evidence archives](original-evidence-delivery/README.md) are now
prepared locally, with every indexed member checked against its original pin.
They still need to be transferred to the reviewer; the small Git delivery record
does not deliver those archives or the external runtime/parser dependencies.

## 8. Check qualification and environment evidence

Read these packet entry points and their `delivery.json`, `EVIDENCE.json`,
fixed PLAN/BATCH, source inventories and manifests as available:

- `out/artifacts/verification/graph-state-pipeline/review/sealed-v1/README.md`
- `out/artifacts/verification/graph-state-integer/review/sealed-v2/README.md`
- `out/artifacts/verification/graph-shape-normalization/today-handoff-20261025/README.md`
- `out/artifacts/verification/graph-regions-declared/today-handoff-20261025/README.md`
- `out/artifacts/verification/graph-state-schedule/checkpoint-user-stop/README.md`

Check actual presence and digest of every selected original, source, query,
object, imported proof dependency, audit, statement, process output and replay.
Verify exact Python and loaded xDSL source identities: a package version or
lockfile alone is insufficient. Graham's local parser differed from the
historically selected parser. Absolute-path historical receipts may require
separate delivery support on a laptop. Regeneration under another parser gets
a new identity and a new qualification.

Check that the roster was frozen before measurement; changed and identity
positives are counted separately; raw failures/skips are retained; postchecks
passed; and positive replay batches are not inflated with negative probes or
overlapping suites. A source snapshot or console-only passing test is not a
sealed positive receipt. Earlier diagnostic/failed versions remain unqualified
even if a later corrected version passes.

Current runtime status matters: the installed `libleanshared.so` association
incident remains **UNKNOWN**. Do not run old native controllers or their SDK
dependency guards against it. A new private copy matched 4,527 ORIGINAL SDK
pins, and a separately reviewed basic `Init` kernel probe passed with zero
axioms. That does not explain the old incident, qualify graph code, or release
worker native contexts. New graph builds require their own source/import/object
and runtime context, exact-root audits, fixed qualification and fresh replay.
See `runtime-snapshot-review/README.md` and the retained probe evidence.

## 9. Decide the paper wording

**Candidate narrow wording, subject to the completed human review above:**

> We implement proof-producing translation validators for enumerated Merlin
> graph-transformation boundaries. For accepted boundaries in the stated
> domains, Lean checks preservation in explicit operational models, under the
> listed primitive/effect and IR-correspondence assumptions. Our scalar/SCF
> validators preserve returned values and complete modeled success/error
> terminal states; memory and declared-call results use the separately stated
> outcome/observer relations. We report the checked variants and qualification
> artifacts individually.

Add “ordinary compilation requires acceptance before proceeding” only for
routes with the real corrected compiler-gate positive and fail-closed negatives.
Add an actual-after Lean-binding claim only for the new subsets whose distinct
program terms and bridge have actually been checked and qualified. Specify
trusted Python parsing/projection even after that bridge: kernel checking a
decoded term is not a parser-correctness proof.

Do not write “all graph transformations are verified,” “the shared pipeline is
verified,” “W8A8 is verified,” or “the models/hardware are correct” from this
evidence. Do not replace possible successful-outcome preservation with complete
failure preservation. Do not imply finite-capacity/no-fault, poison, numerical
laws, instruction encoding, native primitive fidelity, runtime delivery or
selected-model coverage have been proved. Keep each open item in a visible
assumptions/coverage row, not buried behind the word “conditional.”

The two paper FIXMEs are closed only when the final text names: (1) exact checked
transformations, semantics, assumptions and trusted IR mapping; (2) a retained
actual before/after example with exact checker output and evidence identity.
A useful, restricted correctness result can satisfy that without claiming every
Merlin graph mechanism.

## 10. Human sign-off worksheet

Copy this once per claim/family into the final review packet:

```text
Proposed exact paper sentence:
Reviewed primary commit + owned patch/source manifest:
Producer revision + corrective patch/commit:
Accepted variants; explicit excluded variants:
Actual original source/middle/final identities:
Final theorem root(s) and full statement file:
Input domain and operational inhabitation witness:
Preserved observation relation; failures included/excluded:
Semantic premises (local contracts/control/allocation/frame/ABI/etc.):
Trusted parsing/projection/host/runtime correspondence:
Each premise: proved / human-reviewed assumption / empirical / open:
Qualification PLAN/BATCH + original receipt + fresh replay:
Exact parser/tool/import/object identity and audit:
Valid wrong-result, stale/missing/forged and domain controls:
Counterexamples/first failures preserved at:
Live compiler enforcement: demonstrated / standalone only / unavailable:
Actual-model coverage: exact captures, or NONE:
Human reviewer/date; accepted / revise / blocked:
Allowed final wording and remaining obligations:
```

Human review should explain the claim and its limits without relying on an
agent's summary. Preserve Graham's independent review and any disagreement;
record the response against the exact new code/evidence cut. An agent's
“reviewed” status does not substitute for Jeremy/Graham/Agustin sign-off.
