/- Operation-level glue for the checked six-operation per-tensor quantizer lowering. -/
import MerlinVerify.Rules.QuantizeLowering
import MerlinVerify.Rules.DequantSite

namespace MerlinVerify.Rules.QuantizeSiteLemmas

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg

theorem empty_sem (ctx : Ctx) (call : CallSem) (op : Op) (shape : List Nat) (elt : Ty)
    (hm : unmodelledAttr? op.blind = none) (hd : dialectOf op.name = "tensor")
    (hn : op.name = "tensor.empty") (ht : Tensor.resultTensorTy op = some (shape, elt)) :
    opSem ctx call op [] [] = .ok [.tensor (MerlinVerify.Tensor.full shape elt .poison)] := by
  rw [tensor_sem_of hm hd]
  simp [Tensor.sem, hn, resultTensorTy_blind, ht]

theorem generic_one (ctx : Ctx) (call : CallSem) (op : Op) (value : RVal) (initial out : MerlinVerify.Tensor)
    (body : Ctx → RegionSem)
    (hm : unmodelledAttr? op.blind = none) (hn : op.name = "linalg.generic")
    (hseg : attrDenseArray? op "operandSegmentSizes" = some [1, 1])
    (hmaps : attrAffineMaps? op "indexing_maps" = some [identityMap 0, identityMap 0])
    (hit : iterKinds? op = some []) (hr : op.results.length = 1)
    (loop : genericSem ctx [identityMap 0, identityMap 0] [] [value] [.tensor initial] body =
      .ok [.tensor out]) :
    opSem ctx call op [value, .tensor initial] [body] = .ok [.tensor out] := by
  have hd : dialectOf op.name = "linalg" := by rw [hn]; cbv
  have splitInputs : splitInsOuts op.blind [value, .tensor initial] = some ([value], [.tensor initial]) := by
    rw [splitInsOuts_blind]
    simp [splitInsOuts, hseg]
  rw [opSem_modelled hm]
  simp only [dispatch, Op.blind_name]
  rw [hd]
  simp only [hn, beq_self_eq_true, Bool.true_or, ↓reduceIte]
  simp only [Linalg.sem, Op.blind_name, hn, splitInputs, attrAffineMaps?_blind, hmaps,
    iterKinds?_blind, hit, loop, Outcome.bind_ok]
  simp [Op.blind_results, hr]

theorem constant_step (ctx : Ctx) (call : CallSem) (env : Env) (op : Op) (r : Value) (bits : FBits .f32)
    (hm : unmodelledAttr? op.blind = none) (hn : op.name = "arith.constant")
    (hr : op.results = [r]) (ha : op.operands = []) (hg : op.regions = [])
    (hv : op.attr? "value" = some (.float bits.toNat (.float .f32))) :
    stepOp opSem ctx call env op = .ok ((r.id, .scalar (.some (.flt .f32 bits))) :: env) := by
  have hd : dialectOf "arith.constant" = "arith" := by cbv
  have operation : opSem ctx call op [] [] = .ok [.scalar (.some (.flt .f32 bits))] := by
    rw [opSem_modelled hm]
    simp [dispatch, hd, Arith.sem, hn, Op.blind_attr?, hv, Op.blind_results, hr,
      scalarOfAttr, FKind.width]
    exact congrArg (Scalar.flt .f32)
      ((BitVec.ofNat_toNat 32 bits).trans (BitVec.setWidth_eq bits))
  unfold stepOp
  rw [ha, hg]
  simp [Env.getAll, Outcome.mapM, evalRegions, operation, Env.bindAll, hr]

theorem getAll_three {env : Env} {a b c : ValueId} {x y z : RVal}
    (h : env.getAll [a, b, c] = .ok [x, y, z]) :
    env.get? a = some x ∧ env.get? b = some y ∧ env.get? c = some z := by
  unfold Env.getAll at h
  simp only [Outcome.mapM, Outcome.ofOption] at h
  cases ha : env.get? a <;> simp [ha] at h
  cases hb : env.get? b <;> simp [hb] at h
  cases hc : env.get? c <;> simp [hc] at h
  rcases h with ⟨rfl, rfl, rfl⟩
  exact ⟨rfl, rfl, rfl⟩

end MerlinVerify.Rules.QuantizeSiteLemmas
