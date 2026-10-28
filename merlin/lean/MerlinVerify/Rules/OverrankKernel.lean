/- Complete initialized kernel, including SSA, literal zero, fill, reduction and returned result.
This is a valid intent reference. It assigns no semantics to an invalid MLIR operation. -/
import MerlinVerify.Rules.OverrankDefined
import MerlinVerify.Rules.W8A8Fill
namespace MerlinVerify.Rules.OverrankKernel
open MerlinVerify Sem.Ops.Linalg OverrankIntent

def iterAttr (i : IterKind) : Attr := .enum "linalg" "iterator_type"
  (match i with | .parallel => "parallel" | .reduction => "reduction")
def generic (kind : FKind) (s : Shape) : Op :=
  .mk "linalg.generic" [⟨5,W8A8Fill.tensorTy s.out (.float kind)⟩] [100,101,2]
    [("indexing_maps",.array ((maps s).map Attr.affineMap)),
     ("iterator_types",.array ((iters s).map iterAttr)),
     ("operandSegmentSizes",.denseArray (.int 32) [2,1]),
     ("prov.transforms",.str "collapse_overrank_matmul")] [] [OverrankBody.region kind] []
def operations (kind : FKind) (s : Shape) : List Op :=
  W8A8Fill.operations s.out (.float kind) (.float 0 (.float kind)) ++
    [generic kind s,W8A8Fill.returned 5]

private theorem static_tensor (shape : List Nat) (elt : Ty) :
    (W8A8Fill.tensorTy shape elt).staticTensor? = some (shape,elt) := by
  have hs : (shape.map some).mapM id = some shape := by
    induction shape with
    | nil => rfl
    | cons d ds ih => simp [ih]
  simp [W8A8Fill.tensorTy,Ty.staticTensor?,hs]
private theorem dialect_empty : Sem.Ops.dialectOf "tensor.empty" = "tensor" := by cbv
private theorem dialect_constant : Sem.Ops.dialectOf "arith.constant" = "arith" := by cbv
private theorem dialect_fill : Sem.Ops.dialectOf "linalg.fill" = "linalg" := by cbv
private theorem dialect_generic : Sem.Ops.dialectOf "linalg.generic" = "linalg" := by cbv
private theorem prov_free : Spec.Annotations.semanticsFree "prov.transforms" = true := by cbv

set_option maxRecDepth 4096 in
set_option maxHeartbeats 2000000 in
/-- Actual full evaluator execution on arbitrary initialized A/B, with the previous destination
ignored. This proves success even though the newly allocated tensor initially contains poison. -/
theorem operations_sound (ctx : Ctx) (call : CallSem) (env : Env) (kind : FKind) (s : Shape)
    (layout : layoutCheck kind s = true) (A B : Idx → Option (FBits kind)) :
    evalOps Sem.Ops.opSem ctx call
      ((100,.tensor (OverrankTensor.floats kind s.left A)) ::
       (101,.tensor (OverrankTensor.floats kind s.right B)) :: env) (operations kind s) =
      .ok [.tensor (tensor ctx.M kind s A B)] := by
  have hf := W8A8Fill.fill_sound ctx call
    ((1,.scalar (value kind (some (zero kind)))) ::
     (0,.tensor (Tensor.full s.out (.float kind) .poison)) ::
     (100,.tensor (OverrankTensor.floats kind s.left A)) ::
     (101,.tensor (OverrankTensor.floats kind s.right B)) :: env)
    s.out (.float kind) (value kind (some (zero kind)))
    (Tensor.full s.out (.float kind) .poison).get
  have hg := OverrankTensor.generic_sound ctx call
    ((2,.tensor (Tensor.full s.out (.float kind) (value kind (some (zero kind))))) ::
     (1,.scalar (value kind (some (zero kind)))) ::
     (0,.tensor (Tensor.full s.out (.float kind) .poison)) ::
     (100,.tensor (OverrankTensor.floats kind s.left A)) ::
     (101,.tensor (OverrankTensor.floats kind s.right B)) :: env)
    kind s layout (actual := OverrankBody.region kind) (by
      simp [Check.alphaRegion,Check.alphaRegions,Check.alphaOps,Check.alphaOp,Check.opHeadEq,
        Check.freshIds,Check.Rho.find,OverrankBody.region,OverrankBody.primitive,
        List.lookup,Op.name,Op.results,Op.operands,Op.attrs,Op.props,Op.regions,Op.successors]) A B
  simp [operations,W8A8Fill.operations,W8A8Fill.constant,W8A8Fill.returned,generic,
    evalOps,evalRegions,Env.getAll,Env.get?,Env.bindAll,Outcome.mapM,Outcome.ofOption,
    prov_free,Sem.Ops.opSem,Sem.Ops.unmodelledAttr?,Sem.Ops.interpretedAttrs,Sem.Ops.dispatch,
    Op.blind,Op.name,Op.results,Op.attrs,Op.props,Op.successors,
    dialect_empty,dialect_constant,dialect_fill,dialect_generic,
    Sem.Ops.Tensor.sem,Sem.Ops.Tensor.resultTensorTy,static_tensor,
    Sem.Ops.Arith.sem,Sem.Ops.Linalg.sem,splitInsOuts,attrDenseArray?,attrAffineMaps?,
    iterKinds?,Op.attr?,Op.isTerminator,terminatorNames,
    scalarOfAttr,List.lookup,Tensor.full,Tensor.canon,value,zero,W8A8Fill.scalarMap] at hf hg ⊢
  simp [hf, Outcome.bind_ok, List.lookup]
  simp [maps, iters, iterAttr] at hg ⊢
  conv_lhs =>
    pattern List.mapM _ (List.replicate (s.batch.length+1) IterKind.parallel)
    rw [mapM_eq_some_map _ id _ (by intro i hi; cases i <;> rfl)]
  simp only [List.map_id, Option.bind_some]
  rw [hg]
  simp [List.lookup]
/-- Valid complete function implementing the independent intent; the old destination is an
explicit, arbitrary third input and has no influence on the new zero-initialized reduction. -/
def program (kind : FKind) (s : Shape) : Module := { ops := [
  .mk "func.func" [] [] [("function_type",.type (.function
    [W8A8Fill.tensorTy s.left (.float kind),W8A8Fill.tensorTy s.right (.float kind),
      W8A8Fill.tensorTy s.out (.float kind)] [W8A8Fill.tensorTy s.out (.float kind)])),
    ("sym_name",.str "subject")] [] [.mk [.mk
    [⟨100,W8A8Fill.tensorTy s.left (.float kind)⟩,
     ⟨101,W8A8Fill.tensorTy s.right (.float kind)⟩,
     ⟨102,W8A8Fill.tensorTy s.out (.float kind)⟩] (operations kind s)]] []] }

/-- Complete function success and absence of poison, including allocation/fill/SSA, for all
contexts and initialized input tensors. No external-call or numerical-law assumptions. -/
theorem program_defined (ctx : Ctx) (kind : FKind) (s : Shape)
    (layout : layoutCheck kind s = true) (A B : Idx → Option (FBits kind))
    (domain : Domain kind s A B) (old : RVal) :
    denote Sem.Ops.opSem ctx (program kind s) "subject"
      [.tensor (OverrankTensor.floats kind s.left A),.tensor (OverrankTensor.floats kind s.right B),old] =
      .ok [.tensor (tensor ctx.M kind s A B)] ∧ ResultsDefined [.tensor (tensor ctx.M kind s A B)] := by
  constructor
  · simpa [program,denote,Module.funcs,Module.func?,Op.asFunc?,Op.name,Op.attr?,Op.props,Op.attrs,
      evalFunc,evalRegion,evalBlocks,evalBlock,Env.bindAll,Op.regions,List.lookup,Region.blocks] using
      operations_sound ctx (evalFunc Sem.Ops.opSem ctx (program kind s) 1)
        [(102,old)] kind s layout A B
  · intro v hv
    simp only [List.mem_singleton] at hv
    subst v
    exact OverrankDefined.intent_clean ctx.M kind s layout A B domain

end MerlinVerify.Rules.OverrankKernel
