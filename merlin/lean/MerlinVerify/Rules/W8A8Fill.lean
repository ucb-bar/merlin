/- Closed literal initialization for W8A8 tensors of any static rank. The fill's actual
region is checked separately from its name; no numerical property of the literal is used. -/
import MerlinVerify.Rules.ParallelFusion
import MerlinVerify.Check.Alpha
import MerlinVerify.Check.Step
import MerlinVerify.Sem.Ops

namespace MerlinVerify.Rules.W8A8Fill

open MerlinVerify MerlinVerify.Sem.Ops.Linalg

def tensorTy (shape : List Nat) (elt : Ty) : Ty := .tensor (shape.map some) elt none

def fillRegion (elt : Ty) : Region := .mk [.mk [{ id := 3, ty := elt }, { id := 4, ty := elt }]
  [.mk "linalg.yield" [] [3] [] [] [] []]]

def constant (elt : Ty) (literal : Attr) : Op :=
  .mk "arith.constant" [{ id := 1, ty := elt }] [] [("value", literal)] [] [] []

def operations (shape : List Nat) (elt : Ty) (literal : Attr) : List Op :=
  [.mk "tensor.empty" [{ id := 0, ty := tensorTy shape elt }] [] [] [] [] [],
   constant elt literal,
   .mk "linalg.fill" [{ id := 2, ty := tensorTy shape elt }] [1, 0]
     [("operandSegmentSizes", .denseArray (.int 32) [1, 1])] [] [fillRegion elt] []]

def returned (id : Nat) : Op := .mk "func.return" [] [id] [] [] [] []

theorem fillRegion_sound (ctx : Ctx) (call : CallSem) (env : Env) (elt : Ty) (value old : Val) :
    evalRegion Sem.Ops.opSem ctx call env (fillRegion elt) [.scalar value, .scalar old] =
      .ok [.scalar value] := by
  simp [evalRegion, evalBlocks, evalBlock, fillRegion, Env.bindAll, evalOps,
    Op.isTerminator, terminatorNames, Op.name, Env.getAll, Outcome.mapM, Env.get?, List.lookup,
    Outcome.ofOption]

def scalarMap (rank : Nat) : AffineMap := { numDims := rank, numSyms := 0, results := [] }

theorem fill_ranges (shape : List Nat) :
    loopRanges [scalarMap shape.length, identityMap shape.length] [[], shape] shape.length =
      .ok shape := by
  simpa [loopRanges, loopRange, scalarMap] using ParallelFusion.identity_ranges shape 0

/-- Filling overwrites every in-bounds cell, including rank-zero tensors. Empty shapes with a
zero extent stay empty. The initial tensor may have arbitrary contents. -/
theorem fill_sound (ctx : Ctx) (call : CallSem) (env : Env) (shape : List Nat) (elt : Ty)
    (value : Val) (initial : Idx → Val) :
    genericSem ctx [scalarMap shape.length, identityMap shape.length]
      (List.replicate shape.length .parallel) [.scalar value] [.tensor ⟨shape, elt, initial⟩]
      (fun c => evalRegion Sem.Ops.opSem c call env (fillRegion elt)) =
      .ok [.tensor (Tensor.full shape elt value)] := by
  apply genericSem_parallel1 ctx _ _ _ _ _ shape (fun _ => value)
    (by rfl) (by simp [scalarMap, identityMap])
    (by simp [shapeOf, scalarMap, identityMap])
    (by simpa [shapeOf] using fill_ranges shape) rfl (by rfl)
  intro point hp
  have old : readAt (.tensor ⟨shape, elt, initial⟩) (identityMap shape.length) point =
      .ok (initial point) := readAt_identity hp
  have scalar : readAt (.scalar value) (scalarMap shape.length) point = .ok value := rfl
  simp [bodyAt, old, scalar, fillRegion_sound]

private theorem static_tensor (shape : List Nat) (elt : Ty) :
    (tensorTy shape elt).staticTensor? = some (shape, elt) := by
  have hs : (shape.map some).mapM id = some shape := by
    induction shape with
    | nil => rfl
    | cons d ds ih =>
      simp only [List.map_cons, List.mapM_cons]
      rw [ih]
      rfl
  simp [tensorTy, Ty.staticTensor?, hs]

private theorem dialect_empty : Sem.Ops.dialectOf "tensor.empty" = "tensor" := by cbv
private theorem dialect_constant : Sem.Ops.dialectOf "arith.constant" = "arith" := by cbv
private theorem dialect_fill : Sem.Ops.dialectOf "linalg.fill" = "linalg" := by cbv
private theorem full_shape (shape : List Nat) (elt : Ty) (value : Val) :
    (Tensor.full shape elt value).shape = shape := rfl

set_option maxRecDepth 4096 in
/-- The constant attribute must actually decode to the specified scalar. -/
theorem operations_sound (ctx : Ctx) (call : CallSem) (env : Env) (shape : List Nat)
    (elt : Ty) (literal : Attr) (value : Scalar)
    (decoded : scalarOfAttr ctx.indexBits literal = some value) :
    evalOps Sem.Ops.opSem ctx call env (operations shape elt literal ++ [returned 2]) =
      .ok [.tensor (Tensor.full shape elt (.some value))] := by
  have constantSem (anyCall : CallSem) :
      Sem.Ops.Arith.sem ctx anyCall (constant elt literal).blind [] [] =
      .ok [.scalar (.some value)] := by
    cases literal <;> simp_all [constant, Op.blind, Sem.Ops.Arith.sem, Op.name, Op.attr?, Op.attrs,
      Op.props, Op.results, scalarOfAttr]
  simp only [constant, Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
    List.map_cons, List.map_nil] at constantSem
  simp [operations, constant, returned, evalOps, evalRegions,
    Env.getAll, Env.get?, Env.bindAll, Outcome.mapM, Outcome.ofOption,
    Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
    Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
    dialect_empty, dialect_constant, dialect_fill,
    Sem.Ops.Tensor.sem, Sem.Ops.Tensor.resultTensorTy, static_tensor,
    Sem.Ops.Linalg.sem, splitInsOuts, attrDenseArray?, Op.attr?,
    Op.isTerminator, terminatorNames]
  rw [constantSem]
  simp [Outcome.bind_ok, List.lookup, full_shape]
  have hf := fill_sound ctx call
    ((1, .scalar (.some value)) :: (0, .tensor (Tensor.full shape elt .poison)) :: env)
    shape elt (.some value) (Tensor.full shape elt .poison).get
  rw [show genericSem ctx [{ numDims := shape.length, numSyms := 0, results := [] },
      identityMap shape.length] (List.replicate shape.length .parallel)
      [.scalar (.some value)] [.tensor (Tensor.full shape elt .poison)]
      (fun c => evalRegion Sem.Ops.opSem c call
        ((1, .scalar (.some value)) :: (0, .tensor (Tensor.full shape elt .poison)) :: env)
        (fillRegion elt)) = .ok [.tensor (Tensor.full shape elt (.some value))] from hf]
  simp [List.lookup]

#print axioms fill_sound
#print axioms operations_sound

end MerlinVerify.Rules.W8A8Fill
