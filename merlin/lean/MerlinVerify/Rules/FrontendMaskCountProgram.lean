/- Actual decoded count-provider checker; provenance is checked as semantically inert. -/
import MerlinVerify.Rules.FrontendMaskCount

namespace MerlinVerify.Rules.FrontendMaskCountProgram
open MerlinVerify.Check
open MerlinVerify FrontendMaskRuntime FrontendMaskBounds FrontendMaskProgram FrontendMaskExecution FrontendMaskCount
open Sem.Ops Sem.Ops.Linalg

abbrev pairs := List (String × Attr)
def genericOp (n : Nat) (attrs props : pairs) : Op :=
  .mk "linalg.generic" [{id := 2, ty := .tensor [some n] (.int 64) none}] [0, 1]
    attrs props [extBody] []

def GenericFacts (n : Nat) (attrs props : pairs) : Prop :=
  let op := genericOp n attrs props
  unmodelledAttr? op.blind = none ∧
  attrAffineMaps? op.blind "indexing_maps" = some [identityMap 1, identityMap 1] ∧
  iterKinds? op.blind = some [.parallel] ∧
  attrDenseArray? op.blind "operandSegmentSizes" = some [1, 1]
instance (n : Nat) (attrs props : pairs) : Decidable (GenericFacts n attrs props) :=
  inferInstanceAs (Decidable (_ ∧ _))

theorem generic_exec (n limit : Nat) (attrs props : pairs) (h : GenericFacts n attrs props)
    (ctx : Ctx) (call : CallSem) (mask init : MerlinVerify.Tensor) (body : Ctx → RegionSem) :
    FrontendMaskRuntime.sem limit ctx call (genericOp n attrs props) [.tensor mask, .tensor init] [body] =
      (genericSem ctx [identityMap 1, identityMap 1] [.parallel] [.tensor mask] [.tensor init] body).bind
        (fun rs => if rs.length = 1 then .ok rs else .stuck "linalg.generic: result count ≠ output count (memref form unsupported)") := by
  obtain ⟨hm, hmaps, hit, hseg⟩ := h
  have hd : dialectOf (genericOp n attrs props).name = "linalg" := by cbv
  rw [linalg_eq limit ctx call _ hd]
  simp only [opSem, hm]
  have hn : (genericOp n attrs props).blind.name = "linalg.generic" := rfl
  have hd' : dialectOf "linalg.generic" = "linalg" := by cbv
  simp only [dispatch, hd', hn, beq_self_eq_true, Bool.true_or, ↓reduceIte]
  change Linalg.sem ctx call (genericOp n attrs props).blind [.tensor mask, .tensor init] [body] = _
  simp only [Linalg.sem, hn, splitInsOuts, hseg, hmaps, hit]
  rfl

def reduceOp : Op :=
  .mk "linalg.reduce" [{id := 5, ty := .tensor [] (.int 64) none}] [2, 4]
    [("dimensions", .denseArray (.int 64) [0])] [] [sumBody] []

theorem reduce_exec (limit : Nat) (ctx : Ctx) (call : CallSem) (env : Env)
    (bits : Nat → Bool) (n : Nat) :
    FrontendMaskRuntime.sem limit ctx call reduceOp [.tensor (extended bits n), .tensor (scalarTensor 0)]
      [fun inner => evalRegion (FrontendMaskRuntime.sem limit) inner call env sumBody] =
      .ok [.tensor (scalarTensor (countFrom bits n 0))] := by
  have hd : dialectOf "linalg.reduce" = "linalg" := by cbv
  have hred := reduce_generic limit ctx call env (extended bits n) bits n rfl (extended_get bits n)
  simp [identityMap, outMap, extended, scalarTensor, Tensor.full, Tensor.canon] at hred
  simp [FrontendMaskRuntime.sem, reduceOp, opSem, dispatch, hd, unmodelledAttr?, interpretedAttrs,
    Op.blind, Op.name, Op.results, Op.props, Op.attrs, Op.successors, Op.attr?,
    LinalgNamed.sem, LinalgNamed.natArray?, attrDenseArray?, LinalgNamed.run, splitInsOuts,
    shapeOf, extended, scalarTensor, Tensor.full, Tensor.canon, LinalgNamed.mapOf,
    List.lookup, identityMap, hred, Outcome.bind]

structure Metadata where
  zero : pairs := [("value", .int 0 (.int 64))]
  splat : pairs := []
  cast : pairs := []

def zeroOp (attrs : pairs) : Op :=
  .mk "arith.constant" [{id := 3, ty := .int 64}] [] attrs [] [] []
def splatOp (attrs : pairs) : Op :=
  .mk "tensor.splat" [{id := 4, ty := .tensor [] (.int 64) none}] [3] attrs [] [] []
def castOp (attrs : pairs) : Op :=
  .mk "arith.index_cast" [{id := 7, ty := .index}] [6] attrs [] [] []

def MetadataFacts (m : Metadata) : Prop :=
  unmodelledAttr? (zeroOp m.zero).blind = none ∧ m.zero.lookup "value" = some (.int 0 (.int 64)) ∧
  unmodelledAttr? (splatOp m.splat).blind = none ∧ unmodelledAttr? (castOp m.cast).blind = none
instance (m : Metadata) : Decidable (MetadataFacts m) := inferInstanceAs (Decidable (_ ∧ _))

theorem zero_exec (limit : Nat) (ctx : Ctx) (call : CallSem) (attrs : pairs)
    (hm : unmodelledAttr? (zeroOp attrs).blind = none)
    (hv : attrs.lookup "value" = some (.int 0 (.int 64))) :
    FrontendMaskRuntime.sem limit ctx call (zeroOp attrs) [] [] = .ok [.scalar (bitval 0)] := by
  have hd : dialectOf (zeroOp attrs).name = "arith" := by cbv
  rw [arith_eq limit ctx call _ hd]
  simp only [opSem, hm]
  have hd' : dialectOf "arith.constant" = "arith" := by cbv
  simp [dispatch, zeroOp, Op.blind, Op.name, Op.results, Op.attrs, Op.props, hd',
    Arith.sem, Op.attr?, hv, scalarOfAttr, bitval]

theorem splat_exec (limit : Nat) (ctx : Ctx) (call : CallSem) (attrs : pairs)
    (hm : unmodelledAttr? (splatOp attrs).blind = none) :
    FrontendMaskRuntime.sem limit ctx call (splatOp attrs) [.scalar (bitval 0)] [] =
      .ok [.tensor (scalarTensor 0)] := by
  have hd : dialectOf "tensor.splat" = "tensor" := by cbv
  have hdo : dialectOf (splatOp attrs).name = "tensor" := hd
  rw [tensor_eq limit ctx call _ hdo (by change "tensor.splat" ≠ "tensor.empty"; decide)]
  simp only [opSem, hm]
  simp [splatOp, Op.blind, Op.name, Op.results, Op.attrs, Op.props, hd, opSem, hm,
    dispatch, Tensor.sem, Tensor.resultTensorTy, Ty.staticTensor?, scalarTensor]

theorem cast_exec (limit : Nat) (ctx : Ctx) (call : CallSem) (attrs : pairs) (n : Nat)
    (hm : unmodelledAttr? (castOp attrs).blind = none) (hib : ctx.indexBits = 64) :
    FrontendMaskRuntime.sem limit ctx call (castOp attrs) [.scalar (bitval n)] [] = .ok [index n] := by
  have hd : dialectOf (castOp attrs).name = "arith" := by cbv
  rw [arith_eq limit ctx call _ hd]
  simp only [opSem, hm]
  have hd' : dialectOf "arith.index_cast" = "arith" := by cbv
  simp [dispatch, castOp, Op.blind, Op.name, Op.results, Op.attrs, Op.props, hd', Arith.sem,
    Arith.widthOf, Arith.unOp, Arith.pureN, Arith.resultElt, Arith.intUn, Arith.bv,
    Ty.bitWidth?, liftEager, List.mapM_cons, List.mapM_nil, bitval, index, hib, Outcome.ofOption, Outcome.bind]

def countOps (n : Nat) (attrs props : pairs) (m : Metadata) : List Op := [
  .mk "tensor.empty" [{id := 1, ty := .tensor [some n] (.int 64) none}] [] [] [] [] [],
  genericOp n attrs props,
  zeroOp m.zero, splatOp m.splat,
  reduceOp,
  .mk "tensor.extract" [{id := 6, ty := .int 64}] [5] [] [] [] [],
  castOp m.cast]

def countRegion (n : Nat) (attrs props : pairs) (m : Metadata) : Region :=
  .mk [.mk [] (countOps n attrs props m ++ [.mk "func.return" [] [7] [] [] [] []])]

def check (n mask : Nat) (attrs props : pairs) (m : Metadata) (actual : Region) : Bool :=
  decide (n < 2^63 ∧ GenericFacts n attrs props ∧ MetadataFacts m) &&
    FrontendBoolSimulation.region [] [(0, mask)] (countRegion n attrs props m) actual

local macro "count_steps" : tactic =>
  `(tactic| simp [*, evalRegion, evalBlocks, evalBlock, evalOps, evalRegions, countRegion, countOps,
    Op.isTerminator, terminatorNames, Env.bindAll, Env.getAll, Env.get?, List.lookup,
    Outcome.mapM, Outcome.ofOption, reduce_exec,
    FrontendMaskRuntime.sem, opSem, dispatch, unmodelledAttr?, interpretedAttrs,
    Op.blind, Op.name, Op.results, Op.props, Op.attrs, Op.successors, Op.attr?,
     Tensor.sem, Tensor.resultTensorTy,
    Ty.staticTensor?, Arith.sem, Arith.widthOf, Arith.unOp, Arith.pureN, Arith.resultElt,
    Arith.intUn, Arith.bv, Ty.bitWidth?, liftEager, scalarOfAttr, hasUnit,
    List.mapM_cons, List.mapM_nil, Outcome.bind, scalarTensor, Tensor.full, Tensor.canon,
    Idx.inBounds_nil_iff, index, bitval])

def countEnv (n : Nat) (mask : MerlinVerify.Tensor) (bits : Nat → Bool) (outer : Env) : Env :=
  [(7, index (countFrom bits n 0)), (6, .scalar (bitval (countFrom bits n 0))),
   (5, .tensor (scalarTensor (countFrom bits n 0))), (4, .tensor (scalarTensor 0)),
   (3, .scalar (bitval 0)), (2, .tensor (extended bits n)),
   (1, .tensor (Tensor.full [n] (.int 64) .poison)), (0, .tensor mask)] ++ outer

set_option maxHeartbeats 1500000 in
theorem count_prefix_exec (n limit : Nat) (attrs props : pairs) (m : Metadata)
    (hh : GenericFacts n attrs props) (hmeta : MetadataFacts m) (ctx : Ctx) (call : CallSem)
    (mask : MerlinVerify.Tensor) (bits : Nat → Bool) (hm : MaskMatches mask bits n)
    (hib : ctx.indexBits = 64) (outer : Env) (tail : List Op) :
    evalOps (FrontendMaskRuntime.sem limit) ctx call ((0, .tensor mask) :: outer)
      (countOps n attrs props m ++ tail) =
    evalOps (FrontendMaskRuntime.sem limit) ctx call (countEnv n mask bits outer) tail := by
  simp only [countEnv, List.cons_append, List.nil_append]
  have hg : ∀ env : Env,
      FrontendMaskRuntime.sem limit ctx call (genericOp n attrs props)
        [.tensor mask, .tensor (Tensor.full [n] (.int 64) .poison)]
        [fun inner => evalRegion (FrontendMaskRuntime.sem limit) inner call env extBody] =
      .ok [.tensor (extended bits n)] := by
    intro env
    rw [generic_exec n limit attrs props hh]
    rw [ext_generic limit ctx call env mask bits n hm]
    rfl
  have hz := zero_exec limit ctx call m.zero hmeta.1 hmeta.2.1
  have hs := splat_exec limit ctx call m.splat hmeta.2.2.1
  have hc := cast_exec limit ctx call m.cast (countFrom bits n 0) hmeta.2.2.2 hib
  simp only [zeroOp, bitval] at hz
  simp only [splatOp, scalarTensor, Tensor.full, Tensor.canon, Idx.inBounds_nil_iff, bitval] at hs
  simp only [castOp, bitval] at hc
  simp only [genericOp, Tensor.full, Tensor.canon, ite_self] at hg
  have hr := fun env : Env => reduce_exec limit ctx call env bits n
  simp only [reduceOp, scalarTensor, Tensor.full, Tensor.canon, Idx.inBounds_nil_iff, bitval] at hr
  have hdEmpty : dialectOf "tensor.empty" = "tensor" := by cbv
  have hdSplat : dialectOf "tensor.splat" = "tensor" := by cbv
  have hdExtract : dialectOf "tensor.extract" = "tensor" := by cbv
  have hdConstant : dialectOf "arith.constant" = "arith" := by cbv
  have hdCast : dialectOf "arith.index_cast" = "arith" := by cbv
  count_steps
  rw [evalOps_cons_step _ _ _ _ _ _ (by simp [Op.isTerminator, genericOp, Op.name, terminatorNames])]
  simp [stepOp, genericOp, Env.getAll, Env.get?, Env.bindAll, List.lookup, Outcome.mapM,
    Outcome.ofOption, evalRegions, hg, Op.operands, Op.results, Op.regions]
  rw [evalOps_cons_step _ _ _ _ _ _ (by simp [Op.isTerminator, zeroOp, Op.name, terminatorNames])]
  simp [stepOp, zeroOp, Env.getAll, Env.get?, Env.bindAll, List.lookup, Outcome.mapM,
    Outcome.ofOption, evalRegions, hz, Op.operands, Op.results, Op.regions]
  rw [evalOps_cons_step _ _ _ _ _ _ (by simp [Op.isTerminator, splatOp, Op.name, terminatorNames])]
  simp [stepOp, splatOp, Env.getAll, Env.get?, Env.bindAll, List.lookup, Outcome.mapM,
    Outcome.ofOption, evalRegions, hs, Op.operands, Op.results, Op.regions]
  rw [evalOps_cons_step _ _ _ _ _ _ (by simp [Op.isTerminator, reduceOp, Op.name, terminatorNames])]
  simp [stepOp, reduce_exec, Env.getAll, Env.get?, Env.bindAll, List.lookup, Outcome.mapM,
    Outcome.ofOption, evalRegions, reduceOp, Op.operands, Op.results, Op.regions, hr]
  count_steps
  rw [evalOps_cons_step _ _ _ _ _ _ (by simp [Op.isTerminator, castOp, Op.name, terminatorNames])]
  simp [stepOp, castOp, Env.getAll, Env.get?, Env.bindAll, List.lookup, Outcome.mapM,
    Outcome.ofOption, evalRegions, hc, Op.operands, Op.results, Op.regions]


set_option maxHeartbeats 1500000 in
theorem count_region_exec (n limit : Nat) (attrs props : pairs) (m : Metadata)
    (hh : GenericFacts n attrs props) (hmeta : MetadataFacts m) (ctx : Ctx) (call : CallSem)
    (mask : MerlinVerify.Tensor) (bits : Nat → Bool) (hm : MaskMatches mask bits n)
    (hib : ctx.indexBits = 64) :
    evalRegion (FrontendMaskRuntime.sem limit) ctx call [(0, .tensor mask)]
      (countRegion n attrs props m) [] = .ok [index (countFrom bits n 0)] := by
  simp only [countRegion, evalRegion, evalBlocks, evalBlock, Env.bindAll, List.length_nil,
    ↓reduceIte, List.map_nil, List.zip_nil_left, List.nil_append, Outcome.bind_ok]
  rw [count_prefix_exec n limit attrs props m hh hmeta ctx call mask bits hm hib []]
  rfl


/-- Acceptance of a decoded provider region yields its exact initialized mask count. The caller
supplies only the cut input's initialized mask meaning and its SSA environment correspondence. -/
theorem check_sound {n maskId : Nat} {attrs props : pairs} {m : Metadata} {actual : Region}
    (h : check n maskId attrs props m actual = true) (limit : Nat) (ctx : Ctx)
    (hu : FrontendBoolFptosi.UneCorrect ctx.M) (call : CallSem)
    (mask : MerlinVerify.Tensor) (bits : Nat → Bool) (hm : MaskMatches mask bits n)
    (hib : ctx.indexBits = 64) (env : Env)
    (he : Check.Corr [(0, maskId)] [(0, .tensor mask)] env) :
    evalRegion (FrontendMaskRuntime.sem limit) ctx call env actual [] =
      .ok [index (countFrom bits n 0)] := by
  simp only [check, Bool.and_eq_true, decide_eq_true_eq] at h
  have hr := FrontendBoolSimulationWith.region_sound (FrontendMaskRuntime.sem limit)
    (FrontendBoolSimulationWith.candidateLaws limit) [] call _ _ _ ctx _ env h.2 hu he
  rw [FrontendBoolSimulationWith.policy_nil] at hr
  rw [← congrFun hr [], count_region_exec n limit attrs props m h.1.2.1 h.1.2.2 ctx call mask bits hm hib]

#print axioms check_sound
end MerlinVerify.Rules.FrontendMaskCountProgram
