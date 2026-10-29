# Assumption ledger

Everything a verified claim depends on without proving it (design §10). Each entry says what is
assumed, why, and what evidence stands behind it. A claim that relies on an entry must cite it.

## Integration status — 2026-10-29

The current collaboration branch is `feat/pass-verif-integration`. Its five
default compiler checks concern tensor DCE/CSE, ordered unrolling/segmentation
and scoped private-buffer reuse. An additional explicitly selected
`bf16_f32acc_draft_contract` gate checks candidate numerical-reference
conformance and remains outside `all`. The user has retained draft status until
independent review. Configuration-write elision now also has an explicit
`configuration_write_elision` compiler selector and original-owner hooks for
instruction IR and straight-line Kernel scheduling. Installed consumer/worker
qualification includes actual bare instruction and ordinary schedule/native
paths. `configuration_loop_hoisting` also guards the public producer before its
ordinary Kernel consumer. Both remain outside `all` and require original selected
ISA/effect inputs. Hoisting retains its all-index `Models` premise and scoped
positive-loop profile; finite host bounds/native examples do not discharge it.
The additive V3 prefix checker derives sequential last-write-wins assignments
and preserves complete partial outcomes for multiple/overlapping pure setters,
including identical retained writes and prefix-only loops. Its canonical
ten-module source closure and actual original-owner compiler paths are qualified;
the existing nine-module V1/V2 closures remain intact. This adds no ISA effect,
totality, state-completeness or hardware assumption approval.
The additive V4 Focus checker composes actual prefix motions across multiple
flat top-level loops with full command contexts, frame/atom rosters and an exact
terminal join. Its eleven-module canonical closure, live delivery, fresh replay,
four closed kernel refusals and V1/V2/V3 compatibility pass independent audit.
Ninety distinct current-source cases pass, with 153 finite native observations;
these do not discharge all-index `Models` or approve the selected ISA effects.
The explicit original C64 consumer adds a current-only V5 profile for positive
static extents through 2^63−1. Its versioned request binds the actual builtin
C emitter, both endpoint domains and original selected contracts, retaining
live source/proof-product checks through controlled emission. V1–V4 defaults
and legacy schemas remain intact. All 129 current-cut cases pass, including
symbolic large-count proofs, fresh controlled MAX63 replay, an actual Lean
changed-count refusal and 15 small native state/failure comparisons. Huge
loops were not compiled or executed. This binds the declared C integer domain;
it does not prove C/encoding/native equivalence or discharge all-index `Models`.
Their state relations
must not be substituted for a numerical-policy proof or for one another.

The explicit `lower_quant_ext` compiler selector now checks actual xDSL and
textual Q/DQ lowering boundaries with typed refinement, source support and
chain composition under the existing proposed Q/DQ semantics. It stays outside
`all`; source-specification approval and framework/IEEE/native fidelity are
not established. Optional `SourceCapture` selection freshly reconstructs the
retained original PT2 graph, checks source correspondence and reopens the pinned
producer/Torch payload before and after the compiler proof. This is trusted
operator source binding, not execution attestation or a closed dependency proof.
Quantizer definedness remains separate from lowering refinement. No source
specification, frozen interface, software pin or semantic assumption is changed.

Source quantize/dequantize expansion proofs and compiler corrections are
inherited. W8A8 source-domain/calibration/bounds work and the bf16/f16
f32-accumulation contract remain candidates with the independent-review and
coverage limits recorded below. This integration adds no numerical admission,
float law, approved source contract or changed A9 assumption. Approved numerical
policy gates remain unavailable; `all` does not request draft MACC conformance.

The additive `Check/Bf16F32AccBoundary.lean` packages the existing candidate
reference equality and strict local-site qualification behind a nonempty-site
check. Its theorem compiled with only the standard three axioms. It is not
independent numerical approval or a new float law. The explicit compiler check
captures the actual pass output and retains its distinct draft claim.
Canonical governance ownership now rejects caller-selected review trees. Loop
objects are frozen before audit and revalidated with tree membership at live
delivery. Formal compiler selections cannot use custom handler aliases or
additional worker import roots. These controls add no semantic assumption.

Current scalar compiler intake supports multiple changed definitions and
argument/carry lists without a separate fixed arity cap, including zero carries.
Scalar compiler unroll factors and tile sizes are integers at least two, without
the historical factor-four/tile-sixteen cap. Private-buffer and standalone
profiles retain their separate declared parameter domains.
Complete syntax, symbolic-expression, query-size and runtime resource budgets
still apply; they do not supply semantic premises. Existing narrower
standalone source-profile declarations retain their original limits. The new
`GraphLoopsShape.Identity` checker bridges unchanged normalization with exact
complete syntax equality and zero rewrite sites, preserving the original
changed-normalization checker. One-to-one private reuse plans retain every
allocation/free binding. New additive untiled initializer simulation lemmas
compile through the complete memory-entry chain. The actual three-buffer gate,
independent-process fresh replay and ordinary compiler delivery now pass, as
does normalization-only change with identity reuse. This does not broaden the
allocator premise or the existing rank-two i8/i16/i32/i64 domain, and supplies no native
allocator or full error/world equivalence.

The additive rank-one source and actual tiled/untiled target bridges derive
one-dimensional Cell/SCF execution, exact source/configuration alpha binding,
private-slot/world partition and local simulation through the complete-entry
root. They introduce no whole-run equality or local-simulation premise. The
installed actual gates pass 250 roots and a 2,545-theorem STD3 audit. All 33
distinct qualification cases pass: 12 host and 21 SDK/native, including both
public compiler routes, two independent-process replays, ten actual target
mutations and three initializer-link component refusals. Independent review checks the
exact source/tool/product joins and finite native wrap observations.
Only captured-uniform-seed initializer bodies are in this additive profile;
computed initializer prefixes need a typed pure-execution bridge. Allocator,
possible-success, typed-input and native/error review boundaries remain intact.

Current outer-loop unrolling supports empty, full-inline and bulk-plus-tail
forms, retaining arbitrary supported nested bodies with dynamic inner bounds
and modeled external effects. Thirty distinct loop-lane tests passed across
three recorded source cuts (14 mode, 6 mixed-entry and 10 nested-mode cases).
Actual sibling-loop changes and surrounding partial-effect contexts now have
checked whole-entry joins, including the actual unroll→tiling intermediate.
Control facts in local wrappers are discharged from the actual prefix; final
entry roots retain no source-success premise. Explicit one-based depth selection
now checks innermost loops through arbitrary supported dynamic ancestors, with
complete actual-entry roots and public worker/native controls at depths two and
three. Independent public inner tiling and combined three-buffer checks also
pass. Selected inner loops can now retain supported deeper bodies: an actual
combined gate, native boundary/wrap checks, fresh independent replay and genuine
callee/stride Lean refusals qualify the new route. Only controlled auxiliary
wrappers may capture unchanged stride block arguments; actual source projection
still requires known positive strides. The roots add no control or success
premise. Passing test
logs are not live completion authority; retained final-cut replay remains
separate. No frozen semantics or new axiom is introduced.

The user-approved MACC producer correction now explicitly rejects unsupported
half contraction policy requests before mutation and preserves named maps.
The corrected producer passed 49 structural/checker cases and two native
regressions; three retained-artifact cases were skipped. This legality approval
does not approve the draft numerical reference or original half-source
equivalence. Shared/init operand binding is separately checked by the wrapper.

The draft MACC worker audits exact consumed roots transitively against the
standard three axioms and scans all imported project declarations for banned
constructs. Its report explicitly identifies this narrower audit scope. It
revalidates canonical project objects and process evidence through delivery;
it does not claim a private object snapshot or a proven runtime import map.
The original full-project auditor remains available and unchanged in default
mode. Qualification results and retained source cuts are in the handoffs.

Read the [quantization handoff](../../out/artifacts/handoff/quantization-lean-20261028/README.md)
and [current integration checkpoint](../../out/artifacts/handoff/pass-verif-integration-20261028/README.md)
before resuming numerical work. Completed compiler tests and fresh replay retain
their recorded source cuts. Four real native SourceCapture examples on both
compiler routes, checked-prefix composition and source-freshness refusals passed
the 14-case cohort; seven observer-byte mutation controls also passed. The exact
pinned CPU wheel and producer are now available and verified in owned scratch.
Final retained replay after the latest integration source changes and broader
source/format campaigns remain separate from these completed tests.

## Modelling choices in the op semantics

| Id | Assumption | Where | Evidence / justification |
|---|---|---|---|
| A1 | A failing `cf.assert` (or poison condition) is UB: claims hold for inputs on which the source's assertions pass. | `Sem/Ops/Cf.lean` | MLIR aborts; abort is not modelled as an outcome. |
| A2 | `arith.remsi INT_MIN, -1 = 0` (defined). | `Sem/Ops/Arith.lean` | The Arith spec only names division by zero as UB for `remsi`; LLVM `srem` would make it UB. |
| A3 | A poison dividend with divisor `-1` in a signed division is UB; a poison divisor is UB. | `Sem/Ops/Arith.lean` | Poison may take the triggering value (`INT_MIN`, `0`); matches LLVM/Alive2 practice. |
| A4 | `tensor.extract`/`tensor.insert` out of bounds, or with a poison index, is UB. | `Sem/Ops/Tensor.lean` | MLIR leaves out-of-bounds access undefined. |
| A5 | A structured op's reductions run in lexicographic order of its iteration space. | `Sem/Ops/Linalg.lean` | The order `convert-linalg-to-loops` materializes; integer results are order-independent. The host differential corpus now exercises order-sensitive f32/f64 matmul without fusion or reassociation; other reductions and lowering configurations remain assumptions. See `docs/guides/lean_differential_tests.md`. |
| A6 | Named linalg ops mean their structured generic form. Map bodies receive inputs followed by the destination element. | `Sem/Ops/LinalgNamed.lean` | Reviewed correction from the legacy inputs-only convention to LLVM [f5e175f](https://github.com/llvm/llvm-project/commit/f5e175f06dacf6751f27f7a4bd9e9a3489e95a5e); executable compiler probes and host comparisons qualify explicit cases. Map progress/definedness and map-to-generic rewriting on typed inputs are now proved. Runtime ranks are established through checked SSA execution; typed receipts/chains preserve that domain. Adjacent closed-expression motion has a separate universal proof, including captured-constant hoisting across static tensor.empty. Adjacent region-free independent operand-taking motion now has a separate source-success proof, with symmetric SSA dependency checks. General non-adjacent and nested motion remain outside this rule. These transformation proofs do not prove general fidelity to MLIR; see `docs/guides/lean_differential_tests.md`. |
| A7 | `tensor.empty` is a tensor of poison. | `Sem/Ops/Tensor.lean` | MLIR leaves its contents unspecified. As a source, poison is the reading every value refines, so a target may read anything there; as a target, a read of it refines only a source that is also poison there. Treating it as poison is therefore never more permissive than "unspecified" towards the target. |
| A8 | Integer division overflow (`INT_MIN / -1`) is UB at every width, i1 included. | `Sem/Ops/Arith.lean` | The Arith spec makes signed division overflow UB; for i1, `(-1) / (-1) = 1` is not representable. |
| A9 | `quant_ext.dequantize_per_tensor`, `quant_ext.dequantize_per_channel` and `quant_ext.quantize_per_tensor` (region-free, value results) only compute their results, so DCE may erase them and CSE may reuse their executed values. | `Rules/Dce.lean` (`dequantOp`, `quantizeOp`), `Check/ExactCse.lean` | Purity is a premise of the three modeled Q/DQ operations; it is not inferred from the dialect name. Current production pruning conservatively retains unregistered Q/DQ until typed lowering. Only these three, whose meanings `Sem/Ops/{Dequant,Quantize}.lean` model, are admitted by the formal rules; other `quant_ext` operations stay ineligible. This wording correction does not change A9. |

## Trusted components (design §10, D11)

| Id | Component |
|---|---|
| T1 | Lean kernel; Lean compiler and runtime for the compiled checker (including the compiler-generated `_unsafe_rec` twins of recursive definitions); `Lean.Json`. |
| T2 | The pipeline’s `torch_mlir` parser/printer and Python bridge: export/round trip, receipt byte binding/replay, authored roster selection and gate orchestration. These glue components are regression tested, not formally proved. Within one gate job, a process-local receipt session may reuse actual successful checks instead of executing them twice. The host records the exact result bytes when a live check completes; caller-supplied hashes and saved receipts grant no reuse authority. Schema, provenance, axiom, export and byte bindings are still checked. Edited receipts, other jobs and external composition always replay. |
| T3 | Fidelity of the MerlinSem op semantics to MLIR and the captured source producer. Not proved. In addition to hand-built `#guard` cases, `merlin/tests/ir/test_lean_differential.py` compares the actual Lean denotation with native upstream MLIR on a deterministic integer/tensor/f32/f64 corpus and checks named-op generalization. The map signature mismatch is corrected and its corpus must pass. Probes bind the installed contract to compiler fingerprints; package versions do not establish LLVM revisions. Coverage limits and required-tool instructions are in `docs/guides/lean_differential_tests.md`; untested forms and targets remain trusted. New bounded integer-folder/named-op and dequantization tests use hash-pinned MLIR. A separate hash-pinned PyTorch 2.10.0+cpu oracle checks both dequantization source operations against NumPy and selected actual MLIR-folded compiler bodies. NaN payloads, complete IEEE-instance qualification, human specification review, and capture-producer binding remain outstanding. |
| T4 | Upstream MLIR lowering below the verified tensor stage, LLVM, the linker, the C runtime. The qualified scalar/parallel defaults and native oracle generalize maps before bufferization. A regression checks the required initializer copy survives; [LLVM issue 213241](https://github.com/llvm/llvm-project/issues/213241) documents the separate direct-map bufferization bug. This is not a proof of downstream lowering. |
| T5 | Accelerator kernels meeting their stated contracts (the OOT hole, design D8). |
| T6 | Floating-point accuracy (measured, never proved; floats are parametric in every theorem). |

## Known model and native qualification limits

The input-dependent cast, typed mixed-reshape, nested integer-body, comparison/select and
combined-pass suites extend T3's bounded evidence. Cast widths, SSA provenance, capture scopes,
and transformation soundness are proved against the locked tensor model; native comparisons
exercise selected initialized integer and bit-preserving tensor cases with independent references.
These extensions introduce no new float laws or assumptions and do not prove general model,
capture-producer or downstream-lowering fidelity. Cast composition now extends into supported
generic bodies, preserving escaping intermediates and deleting only proved-total unused casts.
Local reshape provenance extends transitively and into generic bodies. Captured outer provenance,
unsupported cast directions and bare-input reshape cancellation remain outside coverage.

Value-dependent extraction definedness derives signed index bounds from actual checked constant
producers and initialized unflagged index arithmetic at the claimed width, including wraparound. Facts follow SSA bindings and block
argument shadowing. Initialized tensors and proved in-bounds coordinates are required; unknown
input indices and out-of-bounds/poison sources refuse. Native qualification includes independent
row-major references and nonzero multidimensional captures. Scalar floating totality additionally
covers the finite unary vocabulary and FMA without adding float laws. Only the existing f32/f64
negate/abs/sqrt/rounding/FMA interpretation gains native evidence here; other unary functions retain
the parametric model-fidelity assumption. No complete IEEE instance is proved. Scalar format conversion and fixed-width integer-to-float
totality likewise add no numerical laws; selected f32/f64 native cases use exact single-rounding
references. Rank-zero from-elements/extract alias proofs preserve all scalar bits and poison.
Rank-one constructor selection additionally has a checked bounds bridge, including actual
constant-index arithmetic. Higher-rank construction remains outside this alias family.

The shared checker additionally proves exact scalar widening/truncation round trips, signed and
unsigned integer min/max commutation, total unused scalar-constant deletion, and adjacent pure
instruction motion under symmetric SSA independence checks. Index literals are normalized by
replaying actual decoded values at the claimed width on both sides; opposite-side hints cannot
supply facts. Initialized unsigned minimum clamps and positive-divisor unsigned remainders can
establish safe tensor-read bounds. These are bounded checker families, not blanket proofs of
canonicalization or downstream lowering. Their regression and native qualification artifacts
identify the exact source and frozen checker; drafted or interrupted tests establish no coverage.

The current `tensor.reassocValid` predicate rejects all-unit nonzero-rank to rank-zero reshape
transitions, and their inverse, with empty reassociation. Upstream MLIR accepts and executes
these forms. This is a conservative support gap: source support and coverage-chain checks
abstain, even when a conditional composition boundary accepts because the source is stuck.
These transitions are outside qualified coverage; no semantics change or new assumption is
introduced by the composition rules.

An extended source-definedness receipt must compose through the matching extended chain
theorem. A cross-review found that the initial Python integration still labelled these chains
with the legacy theorem, whose Boolean premise was narrower. Additive composition theorems,
exact schema identity checks and bounds-only typed/untyped regressions correct that receipt
mismatch. The old theorem statements and the semantics remain unchanged; old affected chain
receipts must be regenerated.

Unsigned widening integer-to-index casts derive bounds from actual typed runtime bits, and
nonwrapping interval addition retains them only below the signed limit at the claimed width.
Unknown direct index inputs remain unbounded. Rank-zero dense extraction compares actual decoded
values against checked scalar materialization candidates; cross-region constant hoisting remains
outside its local theorem. Legal float-to-float bitcasts are another conservative model support
gap: the current float-result branch accepts integer bits only. Native support-gap observations
are kept separate from matching raw-bit integer/float reinterpretation evidence.

The opt-in runtime handoff gate encloses preprocessing. An explicit preparation capture can
extend the chain to its original source, and `verify_native_prefix=True` additionally holds the
actual ordinary native Module/Context through D1 until strict replay accepts. Bufferization and
the remaining native suffix stay in T4. The initial route excludes optional features, custom
pipelines and callbacks, and refuses index-typed 32-bit programs because of S6. Whole-model and
OOT consumers require their own integration. These bounded routes do not establish complete
production-path coverage or a full-compiler correctness claim. Python roster selection, source
binding and finite process supervision remain in T2; OS filesystem/process-creation behavior and
pending kernel cleanup are host assumptions.

Rank-zero `linalg.transpose` cannot currently be qualified: the installed MLIR binding crashes
in verification, and assertion-enabled upstream `mlir-opt` reproduces an empty-permutation-map
assertion in `TransposeOp::getIndexingMaps`. Passing Lean proofs do not establish native behavior
for that form. Reproduction evidence is retained with shared-pass verification artifacts.

S4: the checksum-pinned torch-mlir 20260531.828 native pipeline can mutate a captured initialized
destination across nested `linalg.generic` iterations. A minimal no-splat pure-tensor source expects
`[2,3]`; native execution returns `[2,4]`. Standalone one-shot-bufferization introduces the aliasing
before any oracle wrapper or LLVM lowering. A reduction variant also fails, while an explicit
`scf.for` control copies the destination each iteration and matches the reference. This conflicts
with the documented [immutable tensor contract](https://mlir.llvm.org/docs/Bufferization/#destination-passing-style).
The exact wheel's implementation root cause is not proved. These captured-destination forms are
outside native-fidelity qualification; known-gap regressions are counted separately from matched
cases. Tensor-model refinement and definedness proofs do not establish this downstream behavior.
Fresh per-iteration destinations are qualified separately. No semantics workaround is introduced.

S5: one private checker build crashed in the bundled Clang backend while compiling generated
`DefinedControl.c` (exit 139). The Lean proof kernels compiled. One isolated replay of the
preprocessed reproducer and one isolated replay of the original build both succeeded with the
same compiler/options and unchanged generated C. This intermittent tool failure remains
unresolved; it is distinct from S1's historical evaluator exit 139 and S4's native tensor behavior.
Tests accidentally launched against the stale executable were retained as invalid qualification;
only the subsequent frozen successful build was qualified. Compiler fingerprints, commands,
backtrace and both diagnostic replays are retained with shared-pass verification artifacts.

Closed index addition, subtraction, multiplication and bitwise operations can now be normalized
from actual successful prefix steps at the claimed index width. Target constants remain untrusted
materialization hints and must decode to the exact computed value. Rank-one `from_elements`
selection joins actual constructor point facts with proved signed in-bounds constant indices;
aliases validate the full continuation, including captures and shadowing. Neither family assumes
producer facts at block entry. Cross-region provenance remains a separate coverage obligation.

S6: a retained index32 example adds `2147483647` and `1`, then compares the result signed-less-than
zero. Lean evaluates the original comparison to true, while the pinned native pipeline evaluates
it to false. Canonicalization produces false, which agrees with native execution but differs from
the original Lean result at index32; index64 results agree throughout. An explicit source DLTI
index32 layout does not remove the discrepancy. This is an unresolved model/native or compiler
configuration discrepancy, not an established upstream bug. Package version `20260531.828` is
not a proven LLVM source revision. Exact sources, exports, tool fingerprints and pipelines are
retained; the mismatch is counted separately from matching native qualification. No semantics,
compiler or oracle workaround is introduced.

The preprocessing gate resolves an immutable authored boundary roster before execution and
checks its ordered occurrences, pass identities and unchanged boundaries before and after strict
semantic replay. Independent roster replay takes the caller's expected plan; recorded metadata
cannot define its own coverage denominator. Generic captures without an expected plan explicitly
claim recorded-boundary coverage only. Explicit preparation/native composition is a separate
selection; an OOT consumer is not covered by this host gate.

An explicit `preparation.capture` prefix can now extend the chain to the original source for
ordinary preparation with no selected optional mutations, integer quantization, routing or
provider callbacks. Capture alone is unverified. Consumption independently reconstructs the
authored preparation roster, validates the retained prefix and exact actual preprocessing input,
then checks original-source support/definedness and one concatenated strict chain. Mandatory
frontend-intent and numerical repairs must actually prove or fail; capture does not grant them a
new specification. Existing whole-model builders remain ungated by default; D1 requires the
additional explicit native-prefix option, and OOT integration remains separate.

Exact scalar constant aliases compare decoded runtime values, including raw floating bits,
at the claimed index width; full-continuation renaming preserves captures and shadowing. This
closes the previously recorded duplicate-value alignment gap in closed index folding. Index
comparisons separately replay the actual predicate over proved local index values; the matching
S6 index64 boundary verifies while the index32 mismatch refuses. Constructor selection can learn
computed closed index values from successful original source steps without target intermediates.
Independent adjacent constant motion across region-free operand-using instructions and exact
widening/truncation cancellation now have registered proofs. A final operand-order traversal
follows all aliases, including captured-parent renames. Local neutral integer identities preserve
poison and validate current scalar literals. Region-bearing motion, inner-body operation motion,
captured numeric facts and poison-erasing absorbing identities remain separate coverage gaps;
native agreement alone is not a proof. The differential adapter's wider integer intermediates
leave the locked semantics unchanged and retain explicit input/output and host float limits.

Partial integer cast chains retain typed runtime origins and preserve poison and escaped
intermediates. Insert/extract selection uses actual matching decoded coordinates; source
initialization requires an initialized destination or a proved complete rank-zero overwrite.
Unknown coordinates and positive-rank partial writes into dirty destinations remain refused.
Integer comparison ordering covers all ten predicates with exact outcome equality, using actual
property/attribute precedence. Its internal canonical headers and integer commutation share one
final traversal through all original regions, preserving captures and raw failure outcomes.
This closes the named matmul parse operand-order gap without changing emitted MLIR snapshots.
Actual changing matmul/transpose generalization has held-gate/native qualification; changing
a bounded adjacent identity-map integer fusion family now proves actual changing two-loop-to-one
boundaries, including its fresh empty destination. Captured-scalar fusion and general layouts remain
explicit proof gaps and stop before acknowledgment. Successful sampled executions do not turn a
refused boundary into a verified compilation.

One leading closed scalar constant can be hoisted from a generic body with an exact typed proof;
index/tensor producers and general region motion remain outside this family. An alpha-validated
lexical renaming precedes final operand ordering, preserving complete outcomes and resolving
internal SSA numbering changes after hoisting. Supported integer/float bitcast roundtrips preserve
raw bits, poison and escaping intermediates. Unused static tensor.empty deletion requires a total
producer and complete continuation liveness checks. No semantics or assumption is changed.

A separate typed return-boundary comparator proves strict refinement when one actual identity
parallel generic and return become a return-only function. It relates the source canonical tensor
pointwise to the actual target input; raw tensor equality and arbitrary call continuations are not
assumed. A retained live prefix still abstains. This comparison composes through the same exact
normalizers and preserves the old script-checker interface; no frozen semantics is changed.

## Proposed dequantization source contract

`Spec/Dequant.lean` specifies the distinct per-tensor and per-channel arithmetic orders from
PyTorch v2.10.0 (`_decomposed.py`: per-tensor line 289, per-channel lines 734–738) for signed-i8
input and f32 scale/result, with an i32 or i64 zero point per-tensor (PyTorch converts the int zero
point to f32 once, as captured models carry it) and an i32 zero point per-channel, including tensor
coordinates and poison propagation. `quant_min`/`quant_max` follow PyTorch: per-tensor ignores them;
per-channel only checks them against the i8 range, so [-127, 127] is admitted and nothing is
clamped. f64 scales are excluded: PyTorch then multiplies per-channel in f64 and rounds the product
to f32, a double rounding the f32 model does not express. Per-channel i64 zero points are excluded
(they remain outside the proved per-channel profile; the integration compiler now subtracts at the promoted
integer width on both lowering routes). Its theorems add no axioms or float algebra laws.
`Sem/Ops/Dequant.lean` dispatches that proposed source contract only for checked tensor types,
shapes and metadata. The source support rule proves non-stuck evaluation on typed inputs, and
the registered C1 site rule connects accepted target loops to it. A separate definedness rule
proves successful, nonpoison results on typed, initialized inputs, with an explicit parameter-axis
check. Definedness receipts compose with the refinement chain to retain that input domain.
Such receipts prove refinement relative to this model; they do not establish that the model is
the captured producer's meaning.
Human specification review and exact capture-producer/version binding remain required before
paper qualification (T3). Unsupported granularities, types and attributes remain refused.

## Merlin W8A8 integer component

`Check/PolicyW8A8Body.lean` proves that acceptance of an actual integer region establishes
signed i8 extension, multiplication and i32 accumulation, including poison propagation. Repeated
execution agrees with a modular dot product. `Rules/W8A8Accumulation.lean` proves exact signed
results for every prefix from zero when `K * 16384 < 2^31`; this covers the full i8 range,
including -128. The length condition is an explicit theorem premise, not yet a fact extracted
from an enclosing contraction by a checker. No float law or new assumption is introduced.

`test_lean_w8a8_body.py` exercises Merlin-generated bodies, valid MLIR mutations, SSA renaming,
alias refusal, and bounded comparisons of Lean execution with the MLIR integer folder.
These are component results. Tensor maps, iteration bounds, zero initialization, the quantization
recipe, requantization, source definedness and whole-boundary correspondence are not established
by this checker. No full W8A8 policy receipt is issued. Independent policy-specification review
and chosen-model qualification remain required; upstream LLVM/MLIR passes stay trusted.

## Merlin W8A8 scalar components

`Rules/W8A8Scalar.lean` proves the ordered abs/max, guarded scale, reciprocal, quantization and
requantization scalar execution for all current float models and poison inputs. Alpha checkers
in `Check/PolicyW8A8Scalar.lean` establish these contracts from actual region syntax. Scale and
quantization captures are learned from the actual executed prefix with SSA invalidation, using
`Check/PolicyW8A8Captures.lean`; a hint cannot assert their values. No assumption is added.

The user approved the explicit zero-row rule: select the f32 representation of 0.00001 when the
row maximum compares equal to zero. F9 retains an axiom-free countermodel separating this recipe
from the unguarded formula under abstract floats; it is not an observed IEEE runtime defect.
Conversion can still yield poison in the abstract model, so these results alone do not prove
IEEE clamp-before-convert definedness or whole-source definedness.

Targeted tests cover real-pass bodies, captured producers, valid MLIR mutations and SSA handling.
An actual TinyLlama boundary probe matches 155 integer and 155 requantization bodies; SmolVLA's
quantization boundary is unchanged under captured-scope selection. These are diagnostics, not
whole-boundary receipts. Tensor maps/axes, source correspondence, reviewed policy semantics,
site/chain integration and full qualification remain open.

## Merlin W8A8 tensor contraction component

`Rules/W8A8TensorTrace.lean` lifts scalar steps through ordered mapped tensor updates, including
repeated output coordinates. `Rules/W8A8IntegerTensor.lean` applies the accepted MAC body to every
read/write using independently checked access bounds, with arbitrary poison and initial i32 values.
`Rules/W8A8Matmul.lean` proves that a standard matmul output cell selects exactly K reduction
points in order. No new semantic definition, axiom or float law is introduced.

`Check/PolicyW8A8Matmul.lean` checks the actual maps/iterators, runtime access bounds and scalar
body, and enforces K < 131072. `check_zero_sound` establishes initialized output cells with exact
signed integer dot products for initialized byte inputs and a zero-initialized destination. The
zero destination is an explicit theorem input; recognizing its actual prefix producer and
connecting actual source operands/quantization/scales remain site obligations. The test adapter
reconstructs shapes from actual SSA types but does not issue a whole-boundary receipt.

Fourteen tensor tests pass, including compiler-verified wrong-map/axis/i16 mutations, the strict
K-bound threshold, zero extents, and three native differential cases with four samples each.
Native MLIR, actual Lean generic execution and an independent integer reference agree, including
-128 and modular overflow with nonzero initial accumulators. A TinyLlama diagnostic accepts all
155 hinted integer tensor components; this count is not complete source-to-output coverage.

## Merlin W8A8 initializer component

`Rules/W8A8Zero.lean` and `Check/PolicyW8A8Zero.lean` prove the actual closed matrix
empty/constant/fill initializer and its selected post-execution SSA binding. The alpha matcher
checks literal zero, region wiring, result types and fresh names. The theorem quantifies over
arbitrary surrounding environments; it assumes no caller-provided zero fact.
`Check/PolicyW8A8InitializedMatmul.lean` connects the canonical zero tensor to the tensor
contraction result. Complete source operand/scale correspondence, policy semantics and C5 chain
integration remain open. All 155 hinted TinyLlama integer initializers match; no full receipt
is issued. No assumption is added.

## Merlin W8A8 floating tensor stage components

`Rules/W8A8TensorOperands.lean` and `Rules/W8A8Pointwise.lean` prove checked mapped reads and
complete tensor overwrites. `Rules/W8A8FloatTensor.lean` lifts ordered scale, reciprocal,
quantization and zero/one/two-scale requantization contracts. `Rules/W8A8AbsMax.lean` proves the
ordered reduction trace with explicit initial cells. These results preserve poison and quantify
over the existing parametric float model; no float law or assumption is added.

`Check/PolicyW8A8Geometry.lean`, `PolicyW8A8Pointwise.lean` and `PolicyW8A8AbsMax.lean` check
maps, bounds, iterator kinds, types/shapes and region syntax. Scale and clamp literals use
SSA-invalidated constant facts about the executed environment. Expected parameter dimensions
remain contract inputs; source-derived grouping and operand/scale correspondence still need a
site proof. The actual floating zero initializer is not discharged by the reduction theorem.

Tests cover actual Merlin stages, zero extents, compiler-verified layout/body mutations,
rank-zero through rank-three templates and native requantization comparisons with Lean and an
independent f32 reference. Zero/one-scale test templates are synthetic components. A TinyLlama
probe accepts 155 hinted tensor requantization stages with per-tensor scales; this is not a
whole-pass receipt. Policy review, source definedness and C5 site/chain integration remain open.

## Shared W8A8 initialization and initialized abs/max

`Rules/W8A8Fill.lean` and `Check/PolicyW8A8Fill.lean` prove the closed empty/literal/fill producer
and its actual post-execution destination binding for all static ranks. The earlier integer
matrix initializer now delegates to these shared definitions, preserving its public contracts.
`Check/PolicyW8A8InitializedAbsMax.lean` checks exact positive-f32-zero bits and connects that
producer to the ordered abs/max tensor theorem. This removes the caller's zero-destination
premise for the recognized producer; it assumes no floating identity or new float law.

Qualification includes actual left/right Merlin reductions and valid MLIR nonzero/negative-zero,
yield-old and bypass-fill mutations, static ranks and empty extents, SSA handling and four native
fill comparisons. This closes the initializer component gap, not source-to-output policy
correspondence. No complete C5 receipt, independent policy review or new assumption is added.

The native recipe companion executes complete dynamic Merlin contractions against a separate
ordered NumPy reference on finite inputs. Four shapes and twenty positive samples agree bit
for bit; four compiler-verified corruptions are detected. This is numerical evidence about the
declared recipe, not a whole-pass Lean theorem, direct execution of Lean's host max/min model,
or eager TorchAO agreement. General source/site correspondence and independent policy review
remain required.

## Whole-module policy validation and suffix composition

`Check/PolicyModule.lean` checks the actual target against a reference obtained solely from a
fixed source-to-reference transformer. Expansion refusal is a checker refusal. Its soundness
theorem proves conditional refinement of every defined reference function; the strict theorem
requires independently established reference support on the claimed domain. The suffix theorem
composes universal or typed downstream boundaries and does not assume that a syntactic numerical
policy preserves refinement of earlier programs. No numerical policy is approved by this module.

`Rules/W8A8Reference.lean` now supplies a source-derived draft transformer with canonical
contraction eligibility, zero initialization, dynamic groups, and calibrated byte/scale reuse
through supported source layouts. `Check/PolicyW8A8Module.lean` proves its conditional module
match and a stronger qualification theorem requiring original-source definedness and reference
support on typed, nonpoison inputs. It does not establish independent review of this draft.
The compiled draft command always reports the unresolved review requirement. TinyLlama's entire
function matches the reference, while its source/support obligations still refuse captured
quantization semantics; this is not a whole-pass verified receipt. F11's noncanonical source
body is refused. No frozen semantics, policy spec, statement pin or production pass is changed.

The byte-bound draft API snapshots and round-trips the actual before/after programs and binds
the compiled checker, theorem audit, lock and detailed obligations. Draft matching is never
promoted to reviewed policy status. Receipt tests reject attempted promotion and prevent a
forged policy verdict from entering the ordinary refinement chain. The source quantization
contract gaps and independent specification review remain explicit requirements.

`Rules/W8A8DequantWidth.lean` is an additive source-contract review candidate for i32/i64 zero
points. Its i32 specialization is proved equal to the existing contract, and its tensor output
is typed and nonpoison for initialized, valid inputs under every current FloatModel. It is not
installed in the frozen dispatcher. Test-only evaluation compares it with pinned PyTorch and
executed standard MLIR; F12 records a genuine i64 per-channel production discrepancy. Neither
these finite comparisons nor the compatibility theorem add an IEEE law or review approval.

`Check/PolicyW8A8Chain.lean` composes a qualified draft policy boundary with actual downstream
boundaries on each function's original typed, nonpoison input domain. Universal C1 or typed
normalization may establish a suffix step; the typed path checks current source membership and
the exact common signature. Its final target refines `P_W8A8(original)`, not the original floating
program. No monotonicity through earlier rewrites is assumed. The byte-bound API rechecks actual
programs and every joining file; stored receipt verdicts are not proof premises. Detailed reports
and intermediate exports are bound, and report freshness is checked during publication.

The public `w8a8-draft` and `w8a8-chain-draft` commands preserve the independent review refusal,
including when all operational checks pass. A checked suffix does not approve the policy,
establish source quantization semantics, or cover an unbound downstream compiler handoff.

Policy chains may now bind the final program to a recorded `handoff.json`, checking the actual
handoff file and all live witnesses. Checker timeouts produce unmeasured attempts and invalidate
old receipts. An unqualified policy prefix leaves downstream diagnostics explicitly unmeasured;
they are never reported as passing. `Check/ExactSyntax.lean` proves a complete structural equality
check used to avoid SSA alignment for identical policy references. Source support and definedness
remain separate even on this exact-identity path.

`Rules/W8A8QuantizeDomain.lean` characterizes nonpoison byte quantization by success of its exact
ordered conversion. A countermodel proves that clamping and initialized data do not imply
conversion success under every unrestricted FloatModel; the actual proved region may return
`.ok` containing poison. This adds no float law and does not narrow a frozen claim. A concrete
floating interpretation or explicitly justified numerical domain remains a review obligation.

`Rules/W8A8ByteFloat.lean` supplies a pure integer-field binary32 min/max and signed-byte
conversion candidate. Its clamp theorem proves conversion success for every non-NaN value,
including infinities after clipping. `W8A8ByteModel`, `W8A8ByteTensor` and
`Check/PolicyW8A8ByteDefined.lean` connect this result to the actual scalar region, tensor
execution and existing untrusted-input quantization checker. Their explicit domain requires
initialized data and a non-NaN rounded product at every accessed coordinate. They neither assume
this condition for arbitrary inputs nor install an approved global floating interpretation.

The candidate test adapter executes complete dynamic W8A8 programs against native MLIR and an
independent NumPy recipe: twenty samples across four shapes agree bit for bit. Scalar/min/max
qualification brings the total to 801 sample executions; arithmetic NaNs are compared by
classification, and poison conversions are never executed natively. The 31-test regression and
7601-theorem audit pass without new assumptions or locked-file changes. Captured-source
quantization semantics and independent policy/domain review remain required for full receipts.

`Rules/W8A8SourceQuantize.lean` adds an uninstalled captured per-tensor source-contract
candidate from versioned PyTorch references. It retains binary64 scalar reciprocal followed
by f32 conversion, zero-point addition and lower-then-upper clipping to -128/127. It proves
output typing and initialized execution under its explicit post-addition non-NaN domain;
`W8A8SourceByte.lean` proves the corresponding full-range byte-conversion obligation. Finite,
positive scalar calibration scales and scalar i32/i64 zero points are checked at runtime.

Lean, pinned PyTorch and separate native MLIR agree on 64 candidate samples; all 89 retained
TinyLlama quantization parameter sets additionally agree on 1424 synthetic scalar observations.
The combined 50-test regression and 7646-theorem audit pass. This candidate neither changes
frozen quant_ext dispatch nor proves that a complete model satisfies its numerical domain.
Source-contract review, checked model domains and whole-model suffix qualification remain open.

`Rules/W8A8ReferenceBounds.lean` proves that the actual draft source recognizer's accepted K
implies exact i32 accumulation at every prefix, including the coordinate-derived tensor cell
trace and the full signed-byte range. All 155 selected TinyLlama source contractions satisfy
the bound (133 with K=2048, 22 with K=5632). The 22 module tests cover the strict threshold;
the 7666-theorem audit passes. These are source arithmetic side conditions, not policy approval.
The retained SmolVLA capture has no selected W8A8 sites; its first support refusal is a dynamic
`tensor.empty` producing `tensor<?xi64>`, rather than evidence of an active W8A8 failure.

`Rules/W8A8SourceContract.lean` recomputes per-tensor Q/DQ eligibility from actual SSA types,
static shapes, signed storage, i32/i64 zero-point widths and source metadata. It refuses unknown
semantic attributes and duplicate keys; only the reviewed annotation contract permits ignored
attributes. All 399 retained TinyLlama source operations are recognized (310 DQ, 89 Q). This
additive adapter is not installed in frozen source dispatch.

`Check/PolicyW8A8SourceDomain.lean` checks actual initialized quantizer inputs and the exact
post-addition non-NaN domain. `Check/PolicyW8A8Source.lean` combines syntax and runtime checks;
acceptance proves successful, typed, nonpoison candidate execution of that invocation. Its
dequantizer check distinguishes initialized float bits from finite values. No caller-provided
definedness flag or new floating law is used. These are instance theorems, not universal
whole-function input-domain proofs.

The combined regression passes 97 tests with zero skipped; the full guard audits 7709 theorems
with zero violations and unchanged locks/pins. All 89 retained quantization parameter sets pass
the new domain checker on their synthetic inputs, with unchanged retained output observations.
Independent review, approved semantic integration and whole-model qualification remain open.

`Check/PolicyW8A8SourceFacts.lean` derives exact calibration bindings from scalar constants and
rank-zero splats, forgetting facts on every SSA overwrite. Its generic transfer theorem requires
an explicit proof that the chosen semantics preserves those producer meanings; the frozen
semantics satisfies that agreement. No numerical law or source quantization assumption is added.
Checked parameters imply typed, non-stuck source quantization for arbitrary float models and
typed data of the recognized shape. Poison outputs remain possible without the separate domain
check, so this is source support rather than whole-function definedness.

`Check/PolicyW8A8SourcePrefix.lean` now proves calibration-fact validity across complete
successfully executed prefixes under producer-compatible semantics. Its one-pass segment check
supplies the calibration check at every selected source site; unsupported quant_ext operations
refuse. This does not establish execution or source definedness from a static scan.

`Rules/W8A8SourceSemantics.lean` integrates the additive per-tensor Q/DQ contracts into an explicit
provisional program interpretation, preserving ordinary frozen dispatch and nested evaluation.
Lean proves agreement with the checked source adapter and producer compatibility. Complete
functions and internal calls are executable in a separately labeled differential mode; frozen
dispatch, the public receipt checker and approved semantics remain unchanged.

`Check/PolicyExact.lean` proves complete-AST policy matching for arbitrary program semantics.
Its strict theorem retains independent source-definedness and reference-support proofs on the
same context/input domain. This route requires identical SSA identifiers. Raw pass exports can
renumber identifiers: the existing C1 renaming theorem is frozen-semantics-specific and cannot
substitute for the explicit generic semantic laws below. Exact matching or finite execution supplies no
whole-function numerical-domain proof. Jeremy authorized using the existing recipe provisionally;
the assumptions and later correction boundaries are recorded in the W8A8 review packet.

All 399 retained TinyLlama Q/DQ parameter pairs pass the calibration check. The 44-test source
regression and full 7759-theorem audit pass with zero skips/violations. Tests refuse invalid or
unknown scales, stale scalar/tensor facts and altered literals; nonzero initialized zero points
are not mislabeled as symmetric-reuse eligibility. Frozen files and statement pins are unchanged.

`Check/ParametricAlpha.lean` and `ParametricModule.lean` now prove whole-module equality under
checked SSA renaming for an interpretation satisfying explicit operation/call laws. Operand
arity and region emptiness remain visible to those laws. A common fuel induction covers
internal calls, declarations and missing symbols. The checker compares outer metadata and
non-function operations, checks every function body and refuses unrecognized function views
or extra function regions before erasing bodies from the outer comparison.

`Rules/W8A8SourceAlpha.lean` proves these laws for the provisional source interpretation.
`Check/PolicyParametric.lean` connects complete raw before/after modules to the source-derived
policy, with no source-definedness or reference-support premise hidden in the match theorem.
Strict qualification retains both independent obligations on the same domain. The retained
TinyLlama boundary matches under this theorem, including all 155 changed contractions. This
closes its SSA bridge; it does not establish whole-function numerical definedness, approve a
source specification, qualify suffixes or issue a verified receipt.

`Check/PolicyW8A8SourceProgress.lean` proves fact-dependent typed support of actual source Q/DQ
steps, using the current SSA bindings. `Check/PolicyW8A8Support.lean` threads these facts through
whole-function progress and reuses ordinary typing rules under the provisional interpretation.
Acceptance proves `FuncNotStuck` for every declared typed input, including poison. Source
operations inside nested regions and internal-call support currently refuse; matching and
finite execution support for internal calls are separate results.

`Check/PolicyW8A8Supported.lean` independently checks source/reference support and composes it
with the generic policy match. Acceptance proves strict typed `FuncRefines` against the draft
policy reference, as well as source modelling. The retained TinyLlama boundary passes this
combined check for all 155 changed contractions. Support permits UB and poison; it establishes
neither successful initialized execution nor the whole-function numerical domain. Dedicated
poison and false-assertion regressions keep that distinction explicit. Frozen dispatch and
the public full-qualification gate remain unchanged.

## f32-accumulation contract for `lower_bf16_matmul_f32acc`

Additive numerical-contract strengthening (2026-10-29):
`Rules/Bf16F32AccInitialized.lean` proves the existing zero contract is defined on
initialized operands with checked mapped accesses, without float laws. It also
defines an arbitrary per-cell f32 initializer, proves its exact ordered trace,
empty-trace behavior and definedness, and connects the actual accumulating generic
to that contract for independent operand formats. `zero_specialization` preserves
the existing contract exactly. These component results do not enable mixed-format
or nonzero-initializer source acceptance. Extending a half zero is not identified
with literal f32 zero. The compiler still requires its current exact source body,
positive-zero initializer and checked supported maps.

The producer/report classifier now leaves ordinary sum/product reductions alone;
half named matmuls and multiply-add reductions, including malformed ones, retain
atomic refusal when unsupported. Complete output matching remains mandatory.
Independent numerical review and concrete fusion/IEEE fidelity remain open. See
the [separate proposals and theorem/domain map](../../docs/design/qdq_macc_numerical_contract_review.md).

`Rules/Bf16F32AccContract.lean` states the contract in six clauses: per-operand `extf` to f32, the
source's own indexing maps, one accumulator per cell folded in lexicographic order (A5), a `+0.0`
start admitted only for a `+0.0` source destination, one attribute-free `truncf`, and multiply/add
fast-math exactly `none` or `contract`. It is a review candidate written after reading the pass,
not a barrier-authored specification; receipts stay `abstracted` until independent review.

`Check/Bf16F32Acc.lean` and `Check/Bf16F32AccChain.lean` prove that accepted outputs (and checked
downstream refinement suffixes) refine the source-derived reference; strict forms retain separate
source-definedness and reference-support obligations. `Rules/Bf16F32AccTensor.lean` and
`Rules/Bf16F32AccSite.lean` prove the reference's emitted operations compute the contract for any
checked geometry. Recovered hints choose only `Flag` values and `Spec.Annotations.semanticsFree`
attributes; every other attribute is compared with the canonical one.

Fusion: the semantics gives `contract`-flagged operations their own uninterpreted functions.
`Rules/Bf16F32AccFusion.fused_step` proves a fused execution is an instance of the model when the
target fuses every contract step or none, the operand `extf` is injective (bf16 is a shift; f16 on
non-NaN inputs) and the source has no other licensed operation (reported as `fusion_premises`).
Mixed per-site fusion is not covered. Native evidence: `test_lean_bf16_f32acc_native.py` compares
JIT execution, including two retained SmolVLA sites at full shape, with an independent NumPy
reference of the contract, and extends T3's bounded evidence to bf16/f16 `extf`/`truncf`.

Pass-local qualification (2026-10-21): `Check/Bf16F32AccExact.check_sound` establishes equality
of the complete actual output and the source-derived reference under any common operation
semantics satisfying SSA renaming and call locality, including unsuccessful executions. This
does not supply a successful execution of the capture. `Check/Bf16F32AccSites.check_sound`
additionally requires strict qualification of reconstructed fragments containing each actual
source contraction, its actual initialization producers and the actual six target operations.
`collectOps_coverage` preserves every recognized source contraction and its position, while
`Rules/Bf16F32AccNonvacuity.operand_domain_nonempty` supplies initialized operands for every
admitted local shape, including zero extents. Global output matching is independently required;
local success cannot excuse another change to the capture.

This permits a modular bf16 pass claim without claiming whole-capture definedness or correctness
of another shared pass. SmolVLA's dynamic `tensor<?xi64>` and source-definedness gaps still block
its separate whole-capture qualification. The review packet under
`out/artifacts/verification/merlin-passes/frontend/bf16-f32acc/review/` contains the exact proposed
claim, executable evidence publisher and an unapplied specification/pinning patch. The publisher
replays the checker and permits `verified` only after the installed specification, policy files
and all current theorem statements match `SPEC_LOCK`; it cannot approve the policy itself.

## Ordered per-tensor quantizer components

`Rules/QuantizeDefinedness.lean` adds an explicit model-dependent conversion-safe
domain. `scalar_defined_iff` characterizes nonpoison scalar output by the exact
conversion; `result_defined` and `run_defined` prove nonpoison output/successful
source execution on that domain for i32/i64 zero points. `target_result_defined`
transfers the domain through the actual six-operation checked lowering using the
existing site theorem. `initialized_counterexample` preserves typed initialized
inputs with scale +1 in a legal model whose conversions all fail. Thus source
typing/initialization and clamp names cannot justify unrestricted definedness.
No new compile-time definedness verdict, float law, source approval, format or
granularity is introduced. The separate IEEE-instance and frozen-specification
review proposals are linked above.

`Spec/Quantize.lean` is a proposed signed-i8 per-tensor quantizer contract,
following the scalar extraction and operation order in PyTorch v2.10.0. It
retains binary64 reciprocal formation, conversion to f32, multiply, round-even,
zero-point conversion/addition, lower-then-upper clamp, and signed conversion.
The contract is parametric in the existing FloatModel and propagates poison;
it assumes no floating algebra or concrete conversion-success law.

`Rules/QuantizeScalar.lean` proves the actual reciprocal and quantizer bodies
and their composition. `Rules/QuantizeTensor.lean` transports those results
through checked SSA renaming, captured clamp constants, mapped reads and full
tensor overwrites, including rank-zero and empty shapes. Access bounds and
geometry remain independently checked premises.

`Sem/Ops/Quantize.lean` installs the source operation for static f32 data, rank-zero
f32 scales, rank-zero i32/i64 zero points and full-range signed-i8 output.
`Check/RulesQuantize.lean` proves typed source support. `QuantizeRuntime` and
`QuantizeLowering` connect the actual input tensors and checked geometry to the
independent specification. `QuantizeSite.target_sound` proves execution of all six
actual target operations, including their temporary SSA values and clamp captures.
`QuantizeRule.check_sound` binds this to complete source/target fragments; the C1
synthesizer proposes sites and independently rechecks their actual ASTs.
`Check/QuantizeCollapse.lean` additionally proves exact typed normalization of the
actual expansion, checking the entire continuation for temporary escapes. Existing
region traversal lifts this equality into nested generic bodies. Canonicalization
of the three supported dtype spellings is separately proved by source execution.

In the compiler integration checkout, the strict gate accepts all 35 forms of the 230 retained source
quantizers after the real lowering. Rank-zero and empty shapes, both zero-point
widths, Q/DQ composition and altered arithmetic/rounding/reciprocal/clamp refusals
are regression tested. Source support permits poison; no unrestricted conversion
success or whole-model numerical definedness is claimed. Specification/capture
review and native numerical fidelity remain separate obligations.
