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
  - src/merlin/verify/lean_graph_pipeline_inline.py
  - src/merlin/verify/lean_graph_pipeline_tail.py
  - src/merlin/verify/lean_graph_pipeline_gate_profiles.py
  - src/merlin/transforms/planning.py
  - src/merlin/compile/scheduling/configuration_ir.py
  - src/merlin/compile/scheduling/configuration_loops.py
  - src/merlin/compile/scheduling/schedule_pipeline.py
  - src/merlin/llvmlower/device_target_passes.py
  - src/merlin/sched/codegen/__init__.py
  - merlin/lean/MerlinVerify/Check/TypedModule.lean
  - merlin/lean/MerlinVerify/Check/Bf16F32AccBoundary.lean
  - src/merlin/verify/lean/Merlin/GraphStatePipelineLoops.lean
  - src/merlin/verify/lean/Merlin/GraphLoopsUnrollComposition.lean
---

# Completing Merlin pass proofs and compiler integration

The [Q/DQ and MACC review packet](qdq_macc_numerical_contract_review.md) records
additive domain/initialized-accumulator proofs and the remaining source-profile,
SSA, map, native-reference and independent numerical-review acceptance criteria.
These component proofs do not complete the broader compiler profiles.

The agreed endpoint covers Merlin pass proofs and their compiler integration
through output IR. Its original nine interfaces are tensor DCE, CSE,
ordered unrolling, ordered tiling, private buffer reuse, Q/DQ lowering, MACC
policy lowering, redundant configuration-write elimination and scoped
configuration hoisting. Downstream MLIR/LLVM, linking, executable behavior and
hardware correspondence remain outside this endpoint.

The additional `host_graph_fusion` interface strengthens the existing logical
graph proof through actual emitted helpers and every public result. It is an
explicit tenth selector outside `all`, covering one region of static pure integer
pointwise stages on both compiler routes. Fresh source-bound proof, replay,
canonical worker enforcement and failed-product quarantine are implemented.
The next graph extensions are multiple fused regions with exact adjacent joins,
broader indexing/reduction domains, and composition with earlier layout rewrites.
Each needs actual producer-boundary and mutation qualification. A concrete MLIR
arithmetic bridge would strengthen the remaining primitive correspondence premise;
the current common typed total pure-action interpretation remains explicit.
See the [stack guide](../guides/lean_compiler_stack.md#logical-host-graph-fusion)
and [qualification handoff](../../out/artifacts/audits/host-graph-fusion-20261029/README.md).

This is a completion design, not a claim that the endpoint has been achieved.
Current invocation examples and qualified domains are in
[the compiler stack guide](../guides/lean_compiler_stack.md); dated evidence and
remaining unavailable checks belong in
[the integration handoff](../../out/artifacts/handoff/pass-verif-integration-20261028/README.md).

## What “full proof” means here

The 2026-10-29 implementation checkpoint closes current-owner configuration
elision/hoist hooks, empty/inline/tail unrolling, actual sibling/prefix/suffix
context composition, actual unroll→tiling chain composition, and reuse without
mandatory tiling. Buffer qualification now covers the already admitted four
integer widths. Indexed-data hoisting adds a separate conservative host-admitted
profile while retaining the all-index instruction-model premise. Its setter
prefix extension and multiple flat Kernel-loop contexts now have installed
canonical/native qualification. Selected innermost-loop depths also have actual
compiler/native qualification through dynamic ancestors, including retained
deeper bodies with auxiliary stride captures. Explicit original C64 consumers
now bind actual emitted integer domains and admit symbolic positive extents
through 2^63−1, preserving default profiles. Recursive positive counted-Kernel hoisting now has installed original-owner
worker, C64 emission and replay qualification: 78 pytest cases plus five separate
legacy reconstruction checks, with 27 finite native state/failure observations.
All-environment Models and selected effect contracts remain explicit. Broader
memory/source profiles remain engineering work. Rank-one uniform-seed buffers
now pass all 33 installed host/compiler/native/replay/mutation cases through
the complete-entry root, for actual untiled and tiled output. Computed initializer
prefixes and rank-three source bridges still need engineering.
Default-all composition now has installed actual compiler qualification for
untiled extent-three and tile-size-32 extent-257 tail cases. A plan-parametric
same-mode DCE theorem relates complete successful returned-value/world outcomes
after actual dead index-constant deletions. The canonical worker independently
decodes both endpoints and checks their exact frame and SSA joins. Original
live memory ownership binds this stage to its compiler source, and binds the
later native CSE capture to the actual upstream input, pipeline and runner.
Authentic unchanged CSE follows the zero-job, zero-credit identity contract.
Missing/skip-only captures, changed memory CSE and caller allow-skip flags still
refuse. This closes the observed default-all delivery gap for the admitted
memory profiles; it adds no categorized-error, allocator, canonicalization or
downstream native-fidelity proof. Evidence is retained under
`out/build/default-all-memory-20261029/installed-qualification-v3/`.
Private-buffer tile sizes now have a finite 65,536 resource ceiling, separate
from the existing signed-index64 safety obligations.
General LLM operation contexts remain engineering. Existing scalar loop
projection does not admit arbitrary tensor/float data carried through
`linalg.generic` contexts. Actual recapture inventories also exceed its older
2,048 weighted-record budget. A private generic-value/index-carrier extension
has four compiled additive math modules and actual source/target queries under
qualification; it installs no new production eligibility yet. The correction
work retains partial outcomes and one shared external interpretation, without
converting arbitrary tensor/float payloads to mathematical integers.

The effort table below is the earlier scope estimate, not remaining effort.
Broader buffer ranks, producer-language coverage and numerical/source-contract
coverage remain engineering; the human-review-only exit has not been reached.

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

## Engineering scope and effort

Here upstream integration means Merlin's normal compiler and selected scheduling
consumers. The transformation algorithms largely exist. Seven checks are already
registered, including the two explicitly selected numerical checks; configuration
elision and hoisting have separate proof APIs. Most remaining work concerns
coverage, current-source intake and real consumer enforcement.

These are planning estimates from the current implementation, not measured
completion times. One day means an experienced contributor's working day with
Lean/Merlin familiarity. The broader target covers legal changing cases of the
selected current producers under the compiler's index64 contract and the agreed
IR language. Additional dialect semantics, a revised frozen semantic interface,
new allocator policies and index32 support need separate estimates after an
inventory. In particular, the DCE/CSE closure estimates assume the existing
modeled tensor language; they do not budget every possible MLIR operation.

Ranges include implementation, useful mutation controls, exact delivery and
fresh replay for the work item. Human review turnaround, missing pinned tools,
large toolchain builds and unrelated repository gate debt add calendar time.
The first-delivery column is a subset of the broader estimate, not an extra
charge. Broader row estimates exclude the shared foundation and final combined
qualification below.

| Pass | Current compiler status | Next contained delivery | Broader coverage effort | Main uncertainty |
| --- | --- | --- | --- | --- |
| Tensor DCE | Integrated in `all` | Existing route; 1–3 days for domain/source-support closure | 1–3 days within the current modeled domain | Additional operation families found by the inventory |
| Leading tensor CSE | Integrated in `all` | Existing route; 1–3 days for coverage and endpoint audit | 1–3 days within the current modeled domain | Complete context and source-support coverage |
| Ordered unrolling | Integrated for no-tail bulk loops | 5–8 days: activate empty/inline/tail cases through a real audited worker | 10–18 days | Typed cloning/renaming, tail joins, nested and unchanged context |
| Ordered tiling | Integrated for one-dimensional scalar loops and the memory chain | 3–5 days: multiple/unchanged-loop context on the scalar route | 7–12 days | Nested/depth-selected bodies and composition with unrolling |
| Private buffer reuse | Integrated through normalization→reuse→tiling | 5–8 days: independently selected untiled reuse for the current rank-two/i32 fragment | 15–25 days | General heap/epoch framing, current producer ranks/word widths and surrounding code |
| Q/DQ lowering | Explicit compiler gate on both paths | 3–5 days: real pinned SourceCapture qualification and complete existing-route checks | 10–18 days | Capture-to-boundary joins, precisions/granularities and exact scalar conversion |
| bf16/f16 MACC | Explicit draft-contract compiler gate | 3–5 days: separately reviewable producer legality fixes and regression packet | 8–15 days | Initialization/maps/operand wiring, mixed formats and permitted fusion policies |
| Configuration-write elision | Producer has compiler consumers; Lean gate remains separate | 3–5 days: instruction-IR consumer requires fresh proof before returning changed IR | 7–12 days | Current-source intake, nested instruction context and the separate Kernel consumer |
| Scoped configuration hoist | Producer and scoped proof API exist; no ordinary producer invocation found in the inspected source | 5–8 days: one real loop-capable Kernel owner runs and gates the current positive-loop case | 10–18 days | Choosing the owner, multiple prefixes, nesting and scoped effect/frame proofs |

The shared foundation is **10–20 days**, counted once: roughly 5–10 for typed
SSA renaming/context closure and 5–10 for the required semantic joins, common
claim/coverage records and adapters. Reuse the existing owned workers, audits,
capture and delivery machinery. Prove joins where actual routes cross models;
separate routes do not require a universal replacement interpreter.

Reserve **5–10 days** for combined qualification, installed-resource checks where
promised, dependency regressions and the final review packet. The broader nine
rows total 69–124 days; with shared work and final qualification this is
**84–154 contributor-days, approximately 17–31 engineer-weeks before contingency**.
For planning, reserve another 25–35 percent until the high-risk spikes below
close. Three experienced contributors could target roughly 10–16 calendar weeks
with that reserve, subject to shared-proof dependencies and review availability.
This is a staffing scenario, not a committed delivery date.

A narrower delivery that wires one real owner per existing configuration proof
domain, prepares the known MACC producer fixes, qualifies the existing source
capture route and runs focused qualification is approximately **4–6
engineer-weeks**, assuming the pinned tools are available. It leaves the broader
loop/storage/numerical coverage work open and does not satisfy the
human-review-only milestone.

### Concrete integration work per family

**DCE/CSE:** keep the existing `pass_validation.observe` and leading-CSE route.
Compare actual producer eligibility with the typed rulebook and source-support
checker, inventory retained baseline refusals, and check complete module/context
binding. Resolve genuine missing rules in the promised language. No additional
selector or duplicate CSE implementation is needed.

**Unrolling:** the inline and tail source preparations already exist in
`lean_graph_pipeline_inline.py` and `lean_graph_pipeline_tail.py`. They explicitly
remain held and issue no compiler authority. Review and kernel-qualify their
actual generated statements, then add current-buffer intake, canonical dispatch,
every adjacent boundary root and live delivery. Extend the structural theorem
for multiple/nested loops, arbitrary typed cloned bodies, untouched context and
effect/failure preservation. The first spike should compile and corrupt one real
inline and one real nonempty-tail boundary using this route.

**Tiling:** reuse the ordered-iteration theorems and the same context/SSA
foundation. Extend actual projection and adjacent roots to multiple and nested
loops, broader typed bodies and unroll-plus-tail outputs. A standalone scalar
tiling proof and a storage-chain proof retain their respective relations.
Dynamic/negative controls that the current producer leaves unchanged need
identity handling, rather than a new dynamic tiling algorithm.

**Buffer reuse:** separate the reuse boundary from the four-buffer requirement
in `buffer_reuse_bridge.reconstruct_chain`. Build direct untiled intake and a
heap/epoch simulation rooted in actual alloc/init/access/free operations, then
generalize ranks, byte-addressable integer widths and surrounding contexts to
match `transforms.planning.plan_storage`. The current producer already refuses
escapes, unknown aliases, dynamic layouts and asynchronous lifetimes. Inventing
new analyses to optimize those cases is an additional optimization project;
the present gate must account for their unchanged context correctly. Existing
normalization/identity proofs and lifetime lemmas remain reusable.

**Q/DQ:** the worker and both compiler hooks already exist. Provision/select the
exact source capture environment, run real original-PT2 reconstruction and fresh
replay, then bind every earlier preprocessing change through a checked prefix.
Close the production granularity/precision inventory, especially per-group and
non-current output formats. Preserve reciprocal/conversion/rounding/clamp order.
The source model's approval is a separate human decision. A new successful-source
execution claim would add definedness work beyond the current refinement claim.

**MACC:** first prepare independent fixes for generic-body operand recognition,
map handling and initialization, preserving the counterexamples. A legitimate
producer legality fix must follow the intended transform contract; suppressing
rewrites merely to match the existing proof is insufficient. Then extend source
site qualification/reference binding to all promised formats, maps and initial
values, and account for the fusion choices licensed by emitted IR. Keep the
claim as f32-policy conformance with numerical admission pending.

**Configuration elision:** use `ConfigurationIRSelection.apply` and the existing
selected instruction-IR transport for the first gated boundary. Introduce a
current-compiler intake bound to the actual selected ISA/effect contracts;
preserve the historical pinned-producer API. Hold changed IR until the live
completion accepts. `schedule_pipeline` also calls
`eliminate_redundant_configuration` on Kernel programs. Cover that distinct
consumer with a Kernel intake or a proved representation bridge, then add the
nested instruction-block context proof required by the production traversal.

**Configuration hoisting:** choose the actual owner of the counted Kernel
before emission, supply the original effect selection, and invoke the existing
producer under explicit transformation/proof controls. The first proof can reuse
the current positive-loop constant-setter domain. The straight-line
`schedule_pipeline` is not a loop-capable insertion point. Existing structured
Kernel owners that call `sched.codegen.emit_c_function` are candidate sites to
audit. Extend proof coverage to the producer's complete movable prefixes and
recursive scopes, with exact error/terminal-state preservation. Preserve
zero-trip no-op behavior; any new guarded hoist is a separately proved change.

### Additional prepared passes

These are separate from the nine-interface total. Their transformations exist,
but the larger scalar/integer profile's `public_check` refuses and
`compiler_eligible` returns false. Budget **3–5 shared days** to build and review
the operational provider using the common proof/delivery infrastructure, then:

| Additional pass | Incremental effort after shared prerequisites | Required work |
| --- | --- | --- |
| Exact integer simplification | 3–6 days | Standalone actual-boundary intake and dispatch; prove/check canonical word entry/result conditions and wrapping-width cases |
| Scalar redundancy elimination | 3–6 days | Integrate the existing pure scalar CSE/DCE producer with complete scopes, dominance/captures and canonical boundary roots |
| Scalar LICM | 5–10 days | Prove actual motion/context correspondence, invariant SSA operands and producer-checked total speculation; handle dynamic/zero-trip cases of this pure domain |
| Standalone shape/view normalization | 3–5 days | Factor the existing normalization proof out of the storage chain; expose a real selected compiler boundary with complete-module binding |

The scalar LICM producer moves total, state-free expressions, so zero-trip
speculation can be legal. Configuration setters change state and have a
different zero-trip obligation. These passes must keep separate semantics and
legality checks even if they share the worker infrastructure.

### First investigations and delivery order

Start with a **2–3 day inventory/proof spike**, charged to the shared budget:
enumerate every actual producer branch and consuming route; compile the held
inline/tail examples; identify the loop-capable configuration owner; and prove
one reusable SSA/context lemma against the existing models. Verify the exact
Q/DQ tool inputs and enumerate formats/granularities. Use the results to revise
the broad ranges before committing calendar dates.

The smallest production milestone is configuration elision at the instruction-IR
boundary. In parallel work streams, prepare the numerical producer fixes/tool
environment and develop the shared loop/context theorem. Next deliver inline/
tail unrolling and scalar tiling context closure; then direct untiled reuse and
its rank/width generalization. Finish the numerical and configuration coverage
extensions and run a frozen combined qualification cut. Review only remains
when every promised consumer and producer branch has the evidence defined below.

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
