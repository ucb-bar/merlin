/- Actual matrix initialization for W8A8: empty, literal i32 zero, fill. The destination is
proved initialized from operation syntax, never from a certificate or its tensor type. -/
import MerlinVerify.Check.Alpha
import MerlinVerify.Check.Step
import MerlinVerify.Sem.Facts
import MerlinVerify.Sem.Ops

namespace MerlinVerify.Rules.W8A8Zero

open MerlinVerify MerlinVerify.Sem.Ops.Linalg

def zero : Val := .some (.bits 32 0#32)

theorem zero_eq : zero = .some (.bits 32 0) := by decide

def matrixTy (m n : Nat) : Ty := .tensor [some m, some n] (.int 32) none

def fillRegion : Region := .mk [.mk [{ id := 3, ty := .int 32 }, { id := 4, ty := .int 32 }]
  [.mk "linalg.yield" [] [3] [] [] [] []]]

def operations (m n : Nat) : List Op :=
  [.mk "tensor.empty" [{ id := 0, ty := matrixTy m n }] [] [] [] [] [],
   .mk "arith.constant" [{ id := 1, ty := .int 32 }] [] [("value", .int 0 (.int 32))] [] [] [],
   .mk "linalg.fill" [{ id := 2, ty := matrixTy m n }] [1, 0]
     [("operandSegmentSizes", .denseArray (.int 32) [1, 1])] [] [fillRegion] []]

def returned (id : Nat) : Op := .mk "func.return" [] [id] [] [] [] []

theorem fillRegion_sound (ctx : Ctx) (call : CallSem) (env : Env) (value old : Val) :
    evalRegion Sem.Ops.opSem ctx call env fillRegion [.scalar value, .scalar old] =
      .ok [.scalar value] := by
  simp [evalRegion, evalBlocks, evalBlock, fillRegion, Env.bindAll, evalOps,
    Op.isTerminator, terminatorNames, Op.name, Env.getAll, Outcome.mapM, Env.get?, List.lookup,
    Outcome.ofOption]

theorem fill_sound (ctx : Ctx) (call : CallSem) (env : Env) (m n : Nat) :
    genericSem ctx [{ numDims := 2, numSyms := 0, results := [] }, identityMap 2]
      [.parallel, .parallel] [.scalar zero] [.tensor (Tensor.full [m, n] (.int 32) .poison)]
      (fun c => evalRegion Sem.Ops.opSem c call env fillRegion) =
      .ok [.tensor (Tensor.full [m, n] (.int 32) zero)] := by
  apply genericSem_parallel1 ctx _ _ _ _ _ [m, n] (fun _ => zero)
    (by rfl) (by decide) (by simp [shapeOf, Tensor.full, Tensor.canon, identityMap])
    (by rfl) rfl (by rfl)
  intro p hp
  have hr : readAt (.tensor (Tensor.full [m, n] (.int 32) .poison)) (identityMap 2) p =
      .ok ((Tensor.full [m, n] (.int 32) .poison).get p) := readAt_identity hp
  have hz : readAt (.scalar zero) { numDims := 2, numSyms := 0, results := [] } p = .ok zero := rfl
  simpa [Tensor.full, Tensor.canon] using
    (show bodyAt ctx [.scalar zero] [{ numDims := 2, numSyms := 0, results := [] }]
      [identityMap 2] [Tensor.full [m, n] (.int 32) .poison]
      (fun c => evalRegion Sem.Ops.opSem c call env fillRegion) p = .ok [zero] from by
        simp [bodyAt, hr, hz, fillRegion_sound])

private theorem dialect_empty : Sem.Ops.dialectOf "tensor.empty" = "tensor" := by cbv
private theorem dialect_constant : Sem.Ops.dialectOf "arith.constant" = "arith" := by cbv
private theorem dialect_fill : Sem.Ops.dialectOf "linalg.fill" = "linalg" := by cbv
private theorem full_shape (shape : List Nat) (elt : Ty) (value : Val) :
    (Tensor.full shape elt value).shape = shape := rfl

set_option maxRecDepth 4096 in
theorem operations_sound (ctx : Ctx) (call : CallSem) (env : Env) (m n : Nat) :
    evalOps Sem.Ops.opSem ctx call env (operations m n ++ [returned 2]) =
      .ok [.tensor (Tensor.full [m, n] (.int 32) zero)] := by
  simp [operations, returned, evalOps, matrixTy, evalRegions,
    Env.getAll, Env.get?, Env.bindAll, List.lookup, Outcome.mapM, Outcome.ofOption,
    Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
    Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
    dialect_empty, dialect_constant, dialect_fill, Sem.Ops.Arith.sem,
    Sem.Ops.Tensor.sem, Sem.Ops.Tensor.resultTensorTy, Ty.staticTensor?,
    Sem.Ops.Linalg.sem, splitInsOuts, attrDenseArray?, Op.attr?, scalarOfAttr,
    Op.isTerminator, terminatorNames, full_shape]
  have hf := fill_sound ctx call
    ((1, .scalar zero) :: (0, .tensor (Tensor.full [m, n] (.int 32) .poison)) :: env) m n
  simp only [zero] at hf
  rw [hf]
  simp [List.lookup, zero]

#print axioms fill_sound
#print axioms operations_sound

end MerlinVerify.Rules.W8A8Zero
