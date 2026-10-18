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
| A6 | Named linalg ops mean their structured generic form. Map bodies receive inputs followed by the destination element. | `Sem/Ops/LinalgNamed.lean` | Reviewed correction from the legacy inputs-only convention to LLVM [f5e175f](https://github.com/llvm/llvm-project/commit/f5e175f06dacf6751f27f7a4bd9e9a3489e95a5e); executable compiler probes and host comparisons qualify explicit cases. Map progress/definedness and map-to-generic rewriting on typed inputs are now proved. Runtime ranks are established through checked SSA execution; typed receipts/chains preserve that domain. Adjacent closed-expression motion has a separate universal proof, including captured-constant hoisting across static tensor.empty. Operand-taking motion, non-adjacent hoisting and nested rewrites remain unsupported. These transformation proofs do not prove general fidelity to MLIR; see `docs/guides/lean_differential_tests.md`. |
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

The current `tensor.reassocValid` predicate rejects all-unit nonzero-rank to rank-zero reshape
transitions, and their inverse, with empty reassociation. Upstream MLIR accepts and executes
these forms. This is a conservative support gap: source support and coverage-chain checks
abstain, even when a conditional composition boundary accepts because the source is stuck.
These transitions are outside qualified coverage; no semantics change or new assumption is
introduced by the composition rules.

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
