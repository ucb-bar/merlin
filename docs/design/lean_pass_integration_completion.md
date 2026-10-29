---
title: Completing Merlin pass proofs and compiler integration
kind: design
status: draft
owner: verification
last_verified: 2026-10-29
related: [merlin_lean_verification, verified_compiler_roadmap, lean_compiler_stack, lean_qdq_gate, lean_buffer_reuse_gate]
code_refs:
  - src/merlin/verify/pass_validation.py
  - src/merlin/verify/compiler_transforms.py
  - src/merlin/verify/compiler_loop_intake.py
  - src/merlin/verify/lean_graph_state_pipeline_emit.py
  - src/merlin/verify/compiler_qdq.py
  - src/merlin/verify/compiler_qdq_source.py
  - src/merlin/verify/compiler_macc.py
  - src/merlin/verify/buffer_reuse_intake.py
  - src/merlin/verify/configuration_elision_gate.py
  - src/merlin/verify/configuration_hoist_gate.py
  - merlin/lean/MerlinVerify/Check/TypedModule.lean
  - merlin/lean/MerlinVerify/Check/Bf16F32AccBoundary.lean
  - src/merlin/verify/lean/Merlin/GraphStatePipelineLoops.lean
  - src/merlin/verify/lean/Merlin/GraphLoopsUnrollComposition.lean
---

# Completing Merlin pass proofs and compiler integration

The agreed endpoint covers Merlin pass proofs and their compiler integration
through output IR. It covers the nine current interfaces: tensor DCE, CSE,
ordered unrolling, ordered tiling, private buffer reuse, Q/DQ lowering, MACC
policy lowering, redundant configuration-write elimination and scoped
configuration hoisting. Downstream MLIR/LLVM, linking, executable behavior and
hardware correspondence remain outside this endpoint.

This is a completion design, not a claim that the endpoint has been achieved.
Current invocation examples and qualified domains are in
[the compiler stack guide](../guides/lean_compiler_stack.md); dated evidence and
remaining unavailable checks belong in
[the integration handoff](../../out/artifacts/handoff/pass-verif-integration-20261028/README.md).

## What “full proof” means here

The existing theorems prove their stated relations. A theorem quantified over
arbitrary states or lists can be fully proved even when the compiler adapter
accepts only one loop shape. The outstanding work includes adapter coverage,
semantic bridges, producer correctness, actual consumer hooks and qualification.
These require different remedies.

Completion requires coverage of the cases the actual selected Merlin producers
can transform in the promised IR language. Inventory those branches from the
producer code before choosing examples. Do not redefine coverage around a small
successful corpus or make a transformation a no-op to avoid proving it.
Existing genuine producer no-ops can receive an identity proof and zero positive
rewrite credit. Inputs outside the declared language remain explicit refusals.
Timeout, missing tools and resource exhaustion remain unavailable outcomes;
accepting arbitrarily large IR within a fixed budget is not a proof requirement.

The relation must also match the pass. DCE/CSE and Q/DQ use typed refinement;
the scalar loop route preserves returned values, modeled errors and terminal
world; storage currently preserves possible successful public observations;
MACC conforms to a proposed numerical policy. A refinement theorem need not
prove that every source execution is defined. A separate successful-execution
claim needs checked entry conditions and a definedness theorem. In particular,
an abstract float-to-integer conversion cannot simply be assumed total.

## Work order and acceptance criteria

| Order | Deliverable | Evidence required to close it |
| --- | --- | --- |
| 1 | Complete domain and claim inventory; shared semantic joins | Every producer branch, input/output interface, relation, premise and consumer recorded; compatible interpretations connected by proved embeddings |
| 2 | General compiler loop adapter | Remove incidental arity/factor/size caps; cover actual inline, bulk, tail and surrounding-program cases with kernel proofs |
| 3 | Independent storage proof route | Reuse works without a mandatory tiling scaffold; complete ownership/heap simulation and an explicit observation contract |
| 4 | Complete numerical boundaries | Actual Q/DQ source qualification, broader lowering coverage and MACC producer alignment; all technical joins tested |
| 5 | Configuration consumer integration | Both existing checkers run at their real selected consumer boundaries, with source-bound effects and exact delivery |
| 6 | Qualification and review packet | Current-source proof, mutation, delivery and fresh-replay checks complete for every promised route; only listed semantic approvals remain |

Numerical producer fixes can be prepared while the loop/storage bridges are
developed. Approval of numerical policies remains independent of technical
implementation and testing. No phase closes by changing a status field or
adding a selector to `all`.

## 1. Bind complete programs and compose the right relations

Create a common claim inventory using the current receipt, capture and live
completion infrastructure. For each interface record the actual producer,
changing branches, complete input/output module, theorem root, typed domain,
effect/allocator/numerical premises and the consumer that receives output.
An inventory is descriptive; it cannot issue proof authority.

Inventory the trusted parser/exporter, primitive interpretation and tool
transport as well. Require complete typed projection and bind the representation
consumed by the proof to the delivered IR. Include that correspondence code in
the human review packet: matching byte hashes establishes identity, while the
semantic interpretation needs its own justification.

The tensor evaluator, scalar/SCF state model, reuse heap model and configuration
model currently support distinct statements. Add semantic embedding theorems
where a compiler chain crosses models. Bind those embeddings to the complete
typed IR, including other functions, declarations, SSA captures, metadata and
untouched context. Matching a body key or adjacent byte hashes does not by
itself connect two different interpretations of those bytes.

Define composition separately for refinement, observational equivalence and
policy conformance. A policy-changing MACC step must retain its policy label in
the final chain. It cannot be composed as ordinary equality to half-precision
source accumulation. Keep additive bridges outside frozen semantic files.
If the frozen AST/evaluator needs extension, prepare a versioned interface/spec
change for review with migration and conservation lemmas.

For DCE/CSE, retain the current kernel-checked tensor route and complete-module
binding. Inventory operations that the actual producers can remove or merge
against the typed rulebook and source-support checker. Prove missing rules and
context closure for the promised language; do not infer purity from dialect
names. Unsupported operations cannot disappear from a complete-program claim.

The incoming handoff also identifies held integer-simplification, scalar
redundancy and invariant-motion profile preparations. Their activation needs
the operational provider, semantic joins and real compiler consumers above.
Apply the same acceptance criteria when extending beyond the nine current
interfaces; changing a held eligibility flag does not supply those proofs.

Acceptance requires a theorem about the actual complete boundary and its
declared observable result, plus independent mutations of untouched functions,
types, metadata and output delivery. Preserve the canonical theorem-root,
source-inventory and fresh-completion checks already implemented.

## 2. Generalize unrolling and ordered tiling

The initial implementation removes the compiler adapter's independent
sixteen-argument/carry limits, permits zero carries, makes list-proof emission
iterative and counts arguments/operands/results in preflight resource budgets.
It also removes the scalar compiler's factor 2..4 and tile-size 2..16 upper
restrictions using existing parameterized arithmetic lemmas. Factors and sizes
remain integers at least two, under code/proof-growth and signed-index bounds.
Historical standalone and private-buffer profiles keep their declared domains.
Qualification includes factor seven and size seventeen and is recorded
separately in the handoff. Other construction limits, including the retained
external-call proof expansion bound, still need the same resource/domain audit.
A size cap imposed by a historical fixture is different from an operational
resource limit.

The harder gap is operational coverage. Actual unrolling includes empty and
fully inlined loops, bulk plus a remainder, depth-selected nested loops and
cloneable typed bodies. The current compiler scaffold covers a nonempty bulk
with no unroll remainder and one region-free top-level loop per changed
function. Existing `GraphLoopsUnroll*` arithmetic/fold results are useful, but
another trip-count lemma will not prove the missing SSA execution bridge.

Add structural induction over actual bodies and surrounding blocks: arbitrary
carry/value lists, lane substitutions, tail joins, calls, memory/effect ordering,
failure state and untouched operations. Compose adjacent and nested loop
boundaries, then lift to complete functions/modules. Ordered tiling needs the
same context closure with short final segments. Follow each producer's actual
behavior for dynamic bounds and metadata; prove genuine unchanged branches as
identity instead of inventing a dynamic rewrite.

A reusable typed SSA-renaming theorem should connect each actual cloned body
to the original body with its shifted induction variable and incoming carries.
Keep body operations under a common effectful interpretation so this extension
does not require reproving every operator's arithmetic. Combine that structural
theorem with the existing ordered-iteration/fold lemmas and checked index bounds.

Parameterize index semantics over the widths the compiler promises. Retain the
existing index32 disagreement as a counterexample until the model/producer
correspondence is resolved; an index64 proof cannot waive it. Native witnesses
help diagnose this correspondence but do not prove LLVM execution.

Acceptance: every changing producer branch in the promised language has a
generic operational theorem; tests cover inline/bulk/tail, multiple/nested
loops, zero/many carries, effects and failure. Corrupt tail offsets, lane order,
carry joins and a second changed function independently and require rejection.

## 3. Decouple buffer reuse from tiling

Build a direct allocation/reuse simulation. Current normalization→reuse→tiling
proofs and explicit identity prefixes are reusable components, but users should
not need a tiling transformation to request a storage gate. A fabricated tile
of size one would leave that coupling unresolved.

Track heap cells, allocation epochs, ownership and initialization through each
allocation, read, write and free. Relate private addresses with a bijection and
frame unchanged public memory. Extend the checked lifetime/access conditions to
plain untiled initialization, supported ranks/types, views/aliases and actual
callers. Derive nonescape and disjointness from actual IR and selected contracts;
candidate lifetime hints cannot establish them.

Choose the observable allocation contract explicitly. Reusing storage can
change finite-memory allocation failure and private allocator state. Equality
of all such states/outcomes is generally false. A complete proof can instead
use public observations under checked ownership and a total/private allocator
contract, or prove a suitable capacity/availability premise. Record which
failure observations are preserved before strengthening the current relation.

Acceptance: independently selected reuse works on complete untiled programs,
every allocation epoch and access is bound, stale aliases and uninitialized
reads reject, and both changing and unchanged prefixes have the stated proofs.

## 4. Close Q/DQ source and lowering integration

The current explicit gate proves refinement under the proposed Q/DQ model. It
does not approve that model. Keep it explicit and outside `all` while its source
specification is under review; see [the Q/DQ guide](../guides/lean_qdq_gate.md).

Run the real original-PT2 `SourceCapture` route with the pinned interpreter,
producer checkout, PyTorch wheel and dependency directory. Required checks are
fresh graph reload/capture, graph arguments, producer/wheel bytes, actual IR,
both compiler routes, tamper controls and separate-process replay. Those tools
are currently unavailable in the located environments and this qualification
was deferred by the user. Host fake-frontend controls do not close this task.
Provisioning and this run remain technical prerequisites to the final milestone.

Connect captured IR to the actual lowering boundary when earlier preprocessing
has changed it. Add a checked/proved prefix chain rather than requiring an
unrelated whole-IR equality or trusting a provenance ID. Extend source-site
binding to complete surrounding typed dataflow and unchanged context.

Inventory every actual production lowering: quantization, per-tensor,
per-channel and per-group dequantization, input/output precisions, scale and
zero-point representations. The current signed-i8/f32 quantizer and selected
integer dequantizer proofs do not cover every such form. Generalize the source
specifications, typed scalar/tensor rules and lowering correspondence for the
promised forms. Reject conversion of scalar scales when it changes results.

Acceptance: all promised production forms and source-to-boundary joins run with
real tools and current source identities. The remaining human decision covers
the proposed source meaning: scalar conversion/reciprocal precision, operation
order, ties-to-even, integer widths, clamp metadata, NaNs/signed zero and source
exceptions. If successful execution is also promised, prove its entry-domain
conditions separately from refinement.

## 5. Align MACC production behavior with its policy proof

The existing checker proves conformance to a draft f32-accumulation reference.
Half accumulation and f32 accumulation are generally different. Preserve the
user's instruction to keep draft status until independent numerical review.

Retained counterexamples and source review expose producer legality gaps:
generic recognition checks operation names without fully binding operand wiring;
named-matmul lowering uses fixed maps; the new f32 destination is zero-filled
even when the original initialization is nonzero. The checker rejects these
cases. They cannot be fixed by broadening proof admission. Prepare separately
reviewable producer legality/correctness patches, preserving counterexamples,
under the repository's separation between verification and producer changes.

Then cover the production-supported operand/result formats, including mixed
bf16/f16 cases where applicable, actual maps, initialization and surrounding
SSA/calls. Instantiate existing parametric policy lemmas against the real
emitted boundary. Resolve mixed fused/unfused accumulation with an indexed
policy relation or an explicit checked producer license. Existing uniform
fusion bridges and f16 injectivity premises do not establish arbitrary fusion
or NaN behavior.

Acceptance: every promised producer form either emits policy-conforming IR or
retains its legitimate unchanged behavior; output and call/source joins are
kernel checked. Human review then approves or rejects the numerical reference,
format/fusion laws and admission record. A whole-capture definedness claim, if
requested, remains a separate proof obligation.

## 6. Connect configuration proofs to actual consumers

Configuration elision and scoped hoisting currently expose separate explicit
instruction/effect APIs. Wire their checks into the actual selected
instruction/Kernel consumers before output publication. Reuse the coverage
registry, fresh worker completion and exact-output delivery mechanism instead
of treating a standalone saved receipt as compiler approval.

Derive effects from the original selected ISA/contract and bind the complete
instruction stream, loop scope, callers and terminal state. A total,
idempotent configuration setter can admit repeated-write elimination under its
declared purity conditions. Observable or failing MMIO writes cannot simply
be removed. Unconditional hoisting out of a zero-trip loop is generally false;
prove a guarded transformation or preserve the producer's legitimate no-op.

Acceptance: both real consumers require fresh proofs on changing boundaries;
effects, failure-state preservation, zero-trip behavior and complete output
delivery have independent mutation controls. Human review concerns the chosen
ISA/effect contract, rather than a missing compiler hook.

## 7. The “only human review remains” exit

Freeze a source/tool cut after implementation. For all nine interfaces run
actual producer positives, independent corrupt-output negatives, complete-entry
and complete-stage coverage checks, exact consumer delivery and fresh-process
replay. Audit Lean source/axioms and protected pins; run relevant dependency
regressions. Check installed resources/import origins if installed delivery is
part of the promised interface. Record skips, unavailable tools and failures
separately; none can close a promised route. Sample counts support engineering
qualification and do not replace universally quantified theorems.

The final review packet must contain the claim/domain matrix, actual statements
and premises, source/tool manifests, semantic joins, retained counterexamples,
current qualification and concrete proposed contract changes. The only open
items may then be human decisions on Q/DQ source semantics and purity, the MACC
policy/fusion laws, allocator observability/failure assumptions, configuration
effects and any frozen-interface amendment. Missing proofs, unconnected
consumers, known producer defects and unavailable qualification runs remain
engineering work and must be closed before declaring this milestone.

Upstream formal infrastructure such as lean-mlir remains a reuse option when a
specific typed-SSA/context or dialect-semantics component fills a demonstrated
gap. The current CSE/DCE route need not be ported merely to change libraries.
Any reused component still needs a proved connection to Merlin's actual IR,
observable semantics and compiler consumer.
