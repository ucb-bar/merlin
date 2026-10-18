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
| A6 | Named linalg ops mean their structured generic form. Map bodies receive inputs followed by the destination element. | `Sem/Ops/LinalgNamed.lean` | Reviewed correction from the legacy inputs-only convention to LLVM [f5e175f](https://github.com/llvm/llvm-project/commit/f5e175f06dacf6751f27f7a4bd9e9a3489e95a5e); executable compiler probes and host comparisons qualify explicit cases. Map progress/definedness and map-to-generic rewriting on typed inputs are now proved. Runtime ranks are established through checked SSA execution; typed receipts/chains preserve that domain. Adjacent closed-expression motion has a separate universal proof, including captured-constant hoisting across static tensor.empty. Operand-taking motion, non-adjacent hoisting and nested motion remain unsupported. These transformation proofs do not prove general fidelity to MLIR; see `docs/guides/lean_differential_tests.md`. |
| A7 | `tensor.empty` is a tensor of poison. | `Sem/Ops/Tensor.lean` | MLIR leaves its contents unspecified. As a source, poison is the reading every value refines, so a target may read anything there; as a target, a read of it refines only a source that is also poison there. Treating it as poison is therefore never more permissive than "unspecified" towards the target. |
| A8 | Integer division overflow (`INT_MIN / -1`) is UB at every width, i1 included. | `Sem/Ops/Arith.lean` | The Arith spec makes signed division overflow UB; for i1, `(-1) / (-1) = 1` is not representable. |

## Trusted components (design §10, D11)

| Id | Component |
|---|---|
| T1 | Lean kernel; Lean compiler and runtime for the compiled checker (including the compiler-generated `_unsafe_rec` twins of recursive definitions); `Lean.Json`. |
| T2 | The MLIR parser/printer bundled with `torch_mlir` (the pipeline's own), used for export and round trip. |
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
references. Rank-zero from-elements/extract alias proofs preserve all scalar bits and poison;
positive-rank aliases still require a separate bounds bridge.

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

The opt-in runtime handoff gate encloses preprocessing only. Earlier model preparation and the D1
native tensor passes are not automatically included. Available proved/tested routes for those
passes do not establish complete production-path coverage or a full-compiler correctness claim.

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
claim recorded-boundary coverage only. This does not extend the automatic gate to preparation,
D1 or the OOT consumer.

An explicit `preparation.capture` prefix can now extend the chain to the original source for
ordinary preparation with no selected optional mutations, integer quantization, routing or
provider callbacks. Capture alone is unverified. Consumption independently reconstructs the
authored preparation roster, validates the retained prefix and exact actual preprocessing input,
then checks original-source support/definedness and one concatenated strict chain. Mandatory
frontend-intent and numerical repairs must actually prove or fail; capture does not grant them a
new specification. Existing whole-model builders remain ungated by default, and D1/OOT remain
outside this automatic route.

Exact scalar constant aliases compare decoded runtime values, including raw floating bits,
at the claimed index width; full-continuation renaming preserves captures and shadowing. This
closes the previously recorded duplicate-value alignment gap in closed index folding. Index
comparisons separately replay the actual predicate over proved local index values; the matching
S6 index64 boundary verifies while the index32 mismatch refuses. Constructor selection can learn
computed closed index values from successful original source steps without target intermediates.
Independent constant motion across operand-using instructions and widening/truncation cancellation
remain explicit mixed-family coverage gaps in this batch; native agreement alone is not a proof.

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
