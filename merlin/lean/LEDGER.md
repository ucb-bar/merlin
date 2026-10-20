# Assumption ledger

Everything a verified claim depends on without proving it (design §10). Each entry says what is
assumed, why, and what evidence stands behind it. A claim that relies on an entry must cite it.

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

## Trusted components (design §10, D11)

| Id | Component |
|---|---|
| T1 | Lean kernel; Lean compiler and runtime for the compiled checker (including the compiler-generated `_unsafe_rec` twins of recursive definitions); `Lean.Json`. |
| T2 | The pipeline’s `torch_mlir` parser/printer and Python bridge: export/round trip, receipt byte binding/replay, authored roster selection and gate orchestration. These glue components are regression tested, not formally proved. |
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
PyTorch v2.10.0 for signed-i8 input, i32 zero point and f32 scale/result, including tensor
coordinates and poison propagation. Its theorems add no axioms or float algebra laws.
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
