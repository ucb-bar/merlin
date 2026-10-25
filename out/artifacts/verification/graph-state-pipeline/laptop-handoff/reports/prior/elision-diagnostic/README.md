# Configuration elision — conditional whole-program verification

Primary: `/home/jeremy/Projects/merlin`, `feat/pass-verification`. This is an additive
owned delivery. No broker/public acceptance, effect approval, native hardware,
numerical, source-definedness or full-model claim is issued.

## Paper alignment and exact relation

Existing roots: `Merlin.GraphState.Configuration.put` and `put_idempotent` in the
preserved GraphStateConfiguration module, and `Merlin.StateEffects.State`.
`Elision.assign_as_put` connects the new overwrite meaning to that existing root.
New root: `Merlin.GraphState.Elision.check_sound`; `context_preserved` also preserves
arbitrary partial surrounding programs. This closes whole-call omission, which is
not an exchange and therefore is not established by the earlier exchange checker.

Actual transformation: `eliminate_redundant_configuration` in read-only
`merlin-graph-proof-intake@4499eebb959c0f92cefd92cc6d80646ba66474ed`, configuration.py.
Both complete actual Kernels are parsed; full interfaces, attributes, retained
calls, arguments, pointers, unit/token/stage/state annotations and order are bound.
Constant integer configuration values are extracted from actual operands using an
independently selected decoder. Unknown instructions and decorated instructions
are retained barriers. No optimizer omission plan or count is used as proof.

Supported domain: bounded, structurally valid, straight-line Kernel programs,
unique typed configuration fields and integer constant configuration operands.
Each invocation starts with unknown knowledge and arbitrary initialized world.
World is a total map of cells to generic values. Interpretation can be a partial
step returning either a complete successful world or an error plus failed world.
The theorem preserves this entire outcome for every initial world and every
admitted common interpretation. It imposes no source-success precondition.
Arbitrary observers of that entire outcome are therefore preserved. Observations
outside this modeled world/outcome require their own admission.

## Proved in Lean

* Empty initial knowledge is sound for arbitrary initial configuration.
* Successful ordered retained steps maintain the known-state invariant, including
  write-over-clobber behavior and frame conditions. Unknown/unordered effects
  clear knowledge. Required reads do not create facts; known contradictions refuse.
* Omission requires the entire nonempty pure write set to be already known equal.
  It is identity on the whole world, using the explicit total exact setter law.
  Partially redundant setters remain whole calls. This is neither partial rewriting
  nor dead-store motion.
* The independently executable complete-list checker has a soundness proof.
  Retained failures preserve the same error and terminal state. Retained success
  derives the next invariant; omitted setters derive identity. No final-equality,
  intermediate-typing or whole-source-success assumption substitutes for that proof.
* Concrete actual-roster Int models prove a nonzero successful execution (initial
  mode -13/output 2, final mode 7/output 5), and a failed required read retaining
  error 17 and terminal mode -13/output 2. Their operational admission is proved.
  All represented Int cells are initialized; the model has no poison constructor.
  This is not actual runtime source-definedness or poison freedom.

## Still conditional / review questions

| Boundary | Proved or measured | Premise remaining |
|---|---|---|
| Local setter | Known equal overwrite is whole-state identity | Actual operation is total and exactly the declared overwrite, with no pulse, exception, hidden alias or observer |
| Other ordered calls | Successful write/frame/read contract maintains knowledge | Actual declared fields/clobbers/frame and common retained-call interpretation are complete |
| Unknown/unordered calls | Arbitrary partial retained call; knowledge reset | Same actual call has the same world/outcome interpretation on both sides |
| IR projection | Complete actual operands/interfaces matched; adversarial tests | Pinned parser, primary Python decoder, original file selection and byte identity binding are trusted |
| Lean | Actual kernel objects and transitive std3 audit | Pinned tool/kernel and separately identified execution support |
| Native | Explicit C fixture outcomes compared on actual projected rosters | C/Lean and device/fixture correspondence are not proved; no hardware qualification |
| Replay | Fresh original-bound projection/check and exact retained object reproduction | Host receipt protocol is trusted code, not a Lean theorem or broker authority |

Review must establish the operational law, not merely approve a `pure_write` flag
or a hash. A hidden-pulse counterexample has the same installed register values
but different complete observations after deletion. The checker also refuses a
production-produced deletion when the explicit local non-failing admission is
absent. Passing tests and provisional contracts do not manufacture approval.

`verify_configuration_state` read establishment is a separate diagnostic. An
unknown required read may fail, as the concrete failure witness demonstrates;
typing, established reads, execution success and absence of poison are distinct.

## Coverage and preserved counterexamples

Changed actual cases cover multiple independent fields, signed/large constants,
partially redundant multi-write setters, clobbers, opaque/unordered barriers,
persistent observers, required-read failures, dependency annotations, caller
memory/interfaces and exact floating observer bits. Identity and disabled cases
are labeled separately. Valid actual-output mutations and independent Lean data
mutations cover wrong values, first setters, clobber/barrier crossing, altered
uses, missing effects/dependencies, retained order, interfaces and outside state.
Malformed input refusals are separate. Strict receipt tests include stale and
consistently rehashed query/object/audit/dependency/original/snapshot forgeries.

The decimal `-0.0` development fixture was rejected by the pinned text parser;
its log is retained. The parser supports hexadecimal floats. A valid hex signed-
zero mutation demonstrates production's Python equality accepts -0 versus +0,
while the independent complete bit-exact token check refuses it. This is a
structural observer check, not a floating numerical equivalence assertion.

A real configuration-hoisting invocation changes a Kernel but retains its loop.
Actual configuration cleanup rejects that same Kernel because it only accepts
straight-line Calls. No optional hoist→cleanup composition is claimed. No xDSL↔
Kernel bridge, frontend/shared-pass numerical bridge, full LeanMLIR manifest,
broker integration or device semantics has been supplied by this work.

Retained models are not claimed to exercise this pass: this packet qualifies
explicit independently chosen Kernel fixtures, not selected captures/full models.
All prior v2/chain sources/capsules, numerical failures/counterexamples, approvals
and lock identities remain preserved and separate.

## Reproduction and delivery

The fixed PLAN records exact sources, profile, tool, cases, tests and prior seals.
A frozen pytest batch retains all passing artifacts. Original actual-pass checks
are followed by one separate fresh-process replay batch. Every accepted instance
retains its generated query, actual object, complete raw audit and exact theorem
statements. The receipt is a closed versioned host transport; saved JSON is never
proof authority. Replay requires externally selected originals and decoder hash,
rederives both complete programs, checks a fresh theorem and reproduces the old
object from its exact original query path.

The final owned patch contains new Elision leaves only. Dependency closure includes
unchanged existing roots, producer source, selected runtime/imports, proof objects
and pinned Lean. The delivery seal inventories the qualification, private profile,
review packet and preserved development failures. Independent byte verification
checks that inventory and all 27 preserved old source bytes. See delivery.json and
MANIFEST.json in sealed-v1 for final measured counts and hashes.
