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
| A5 | A structured op's reductions run in lexicographic order of its iteration space. | `Sem/Ops/Linalg.lean` | The order `convert-linalg-to-loops` materializes; integer results are order-independent. For floats this is a modelling choice: the differential suite of design §9.4 that would test it is not implemented yet. |
| A6 | Named linalg ops mean their OpDSL generic form. | `Sem/Ops/LinalgNamed.lean` | Transcribed from the OpDSL definitions; the cross-check against `mlir-opt --linalg-generalize-named-ops` (design §4.3) is not implemented yet. |
| A7 | `tensor.empty` is a tensor of poison. | `Sem/Ops/Tensor.lean` | MLIR leaves its contents unspecified. As a source, poison is the reading every value refines, so a target may read anything there; as a target, a read of it refines only a source that is also poison there. Treating it as poison is therefore never more permissive than "unspecified" towards the target. |
| A8 | Integer division overflow (`INT_MIN / -1`) is UB at every width, i1 included. | `Sem/Ops/Arith.lean` | The Arith spec makes signed division overflow UB; for i1, `(-1) / (-1) = 1` is not representable. |

## Trusted components (design §10, D11)

| Id | Component |
|---|---|
| T1 | Lean kernel; Lean compiler and runtime for the compiled checker (including the compiler-generated `_unsafe_rec` twins of recursive definitions); `Lean.Json`. |
| T2 | The MLIR parser/printer bundled with `torch_mlir` (the pipeline's own), used for export and round trip. |
| T3 | Fidelity of the MerlinSem op semantics to MLIR. Not proved; tested only by hand-built `#guard` cases (`Tests/Semantics.lean`) with an arbitrary float model. A differential suite against MLIR's own execution (design §9.4) is not implemented yet. |
| T4 | Upstream MLIR lowering below the verified tensor stage, LLVM, the linker, the C runtime. |
| T5 | Accelerator kernels meeting their stated contracts (the OOT hole, design D8). |
| T6 | Floating-point accuracy (measured, never proved; floats are parametric in every theorem). |
