/- Count-provider execution under the unchanged integer and linalg semantics. -/
import MerlinVerify.Rules.FrontendMaskExecution

namespace MerlinVerify.Rules.FrontendMaskCount
open MerlinVerify FrontendMaskRuntime FrontendMaskBounds FrontendMaskProgram FrontendMaskExecution
open Sem.Ops Sem.Ops.Linalg

def bitval (n : Nat) : Val := .some (.bits 64 (BitVec.ofNat 64 n))
def scalarTensor (n : Nat) : MerlinVerify.Tensor := Tensor.full [] (.int 64) (bitval n)
def outMap : AffineMap := {numDims := 1, numSyms := 0, results := []}
def extBody : Region := .mk [.mk [{id := 300, ty := .int 1}, {id := 301, ty := .int 64}]
  [.mk "arith.extui" [{id := 302, ty := .int 64}] [300] [] [] [] [],
   .mk "linalg.yield" [] [302] [] [] [] []]]
def sumBody : Region := .mk [.mk [{id := 303, ty := .int 64}, {id := 304, ty := .int 64}]
  [.mk "arith.addi" [{id := 305, ty := .int 64}] [303, 304]
     [("overflowFlags", .enum "arith" "overflow" "none")] [] [] [],
   .mk "linalg.yield" [] [305] [] [] [] []]]

theorem ext_body_sem (limit : Nat) (ctx : Ctx) (call : CallSem) (env : Env) (b : Bool) (old : Val) :
    evalRegion (FrontendMaskRuntime.sem limit) ctx call env extBody [.scalar (Arith.boolVal b), .scalar old] =
      .ok [.scalar (bitval (if b then 1 else 0))] := by
  have hd : dialectOf "arith.extui" = "arith" := by cbv
  cases b <;> simp [evalRegion, evalBlocks, evalBlock, evalOps, extBody,
    Op.isTerminator, terminatorNames, Env.bindAll, Env.getAll, Env.get?, List.lookup,
    Outcome.mapM, Outcome.ofOption, FrontendMaskRuntime.sem, opSem, dispatch, unmodelledAttr?, interpretedAttrs, hd,
    Op.blind, Op.name, Op.results, Op.props, Op.attrs, Op.successors, Op.attr?,
    Arith.sem, Arith.widthOf, Arith.boolVal, Arith.unOp, Arith.pureN, Arith.resultElt,
    Arith.intUn, Arith.bv, hasUnit, Ty.bitWidth?, liftEager, List.mapM_cons, List.mapM_nil,
    Outcome.bind, evalRegions, bitval]

theorem sum_body_sem (limit : Nat) (ctx : Ctx) (call : CallSem) (env : Env) (a b : Nat) :
    evalRegion (FrontendMaskRuntime.sem limit) ctx call env sumBody [.scalar (bitval a), .scalar (bitval b)] =
      .ok [.scalar (bitval (a+b))] := by
  have hd : dialectOf "arith.addi" = "arith" := by cbv
  have hf : flagList "none" = ["none"] := by cbv
  simp [evalRegion, evalBlocks, evalBlock, evalOps, sumBody,
    Op.isTerminator, terminatorNames, Env.bindAll, Env.getAll, Env.get?, List.lookup,
    Outcome.mapM, Outcome.ofOption, FrontendMaskRuntime.sem, opSem, dispatch, unmodelledAttr?, interpretedAttrs, hd,
    Op.blind, Op.name, Op.results, Op.props, Op.attrs, Op.successors, Op.attr?,
    Arith.sem, Arith.binOp, Arith.pureN, Arith.resultElt, Arith.intBin,
    Arith.arithWithFlags, Arith.bv, overflowFlags, attrEnum?, hf, liftEager,
    List.mapM_cons, List.mapM_nil, Outcome.bind, evalRegions, bitval, BitVec.ofNat_add]

theorem scalar_set (a b : Nat) : (scalarTensor a).set [] (bitval b) = scalarTensor b := by
  unfold scalarTensor Tensor.full Tensor.canon Tensor.set
  congr 1
  funext p
  by_cases hp : p = [] <;> simp [Idx.inBounds_nil_iff, hp]

theorem scalar_canon (a : Nat) : (scalarTensor a).canonize = scalarTensor a := by
  unfold scalarTensor Tensor.full Tensor.canonize Tensor.canon
  congr 1
  funext p
  simp only [Idx.inBounds_nil_iff]
  split <;> simp_all

/-- One genuine generic-reduction iteration adds an initialized 0/1 input to its accumulator. -/
theorem reduce_step (limit : Nat) (ctx : Ctx) (call : CallSem) (env : Env) (ints : MerlinVerify.Tensor)
    (n i acc v : Nat) (hs : ints.shape = [n]) (hi : i < n) (hget : ints.get [i] = bitval v) :
    loopStep ctx [.tensor ints] [identityMap 1] [outMap]
      (fun inner => evalRegion (FrontendMaskRuntime.sem limit) inner call env sumBody) [scalarTensor acc] [i] =
        .ok [scalarTensor (v+acc)] := by
  have hr := readAt_identity (t := ints) (pt := [i]) (by rw [hs]; exact one_inBounds.mpr hi)
  have hm : readAt (.tensor ints) (identityMap 1) [i] = .ok (bitval v) := by
    simpa [hs, hget] using hr
  have hout : readAt (.tensor (scalarTensor acc)) outMap [i] = .ok (bitval acc) := by
    simp [readAt, outMap, evalMap, toIdx, scalarTensor, Tensor.full, Tensor.canon,
      Idx.inBounds_nil_iff, List.mapM_nil]
  have hw : writeAt (scalarTensor acc) outMap [i] (bitval (v+acc)) =
      .ok (scalarTensor (v+acc)) := by
    simp only [writeAt, outMap, evalMap, toIdx, List.mapM_nil]
    change Outcome.ok ((scalarTensor acc).set [] (bitval (v+acc))) = _
    exact congrArg Outcome.ok (scalar_set acc (v+acc))
  simp [loopStep, bodyAt, List.mapM_cons, List.mapM_nil, hm, hout, sum_body_sem, hw]

/-- Reduction executes exactly the finite mask interval, using bitvector addition with no flags. -/
theorem reduce_fold (limit : Nat) (ctx : Ctx) (call : CallSem) (env : Env)
    (ints : MerlinVerify.Tensor) (bits : Nat → Bool) (total : Nat)
    (hs : ints.shape = [total])
    (hv : ∀ i, i < total → ints.get [i] = bitval (if bits i then 1 else 0)) :
    ∀ n start acc, start+n ≤ total →
    ((List.range' start n).map fun i => [i]).foldlM
      (loopStep ctx [.tensor ints] [identityMap 1] [outMap]
        (fun inner => evalRegion (FrontendMaskRuntime.sem limit) inner call env sumBody))
      [scalarTensor acc] = .ok [scalarTensor (acc + countFrom bits n start)]
  | 0, _, _, _ => by simp [countFrom]
  | n+1, start, acc, hn => by
    have hi : start < total := by omega
    rw [List.range'_succ, List.map_cons, List.foldlM_cons,
      reduce_step limit ctx call env ints total start acc _ hs hi (hv start hi)]
    simpa [countFrom, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
      reduce_fold limit ctx call env ints bits total hs hv n (start+1)
        ((if bits start then 1 else 0)+acc) (by omega)

theorem points_one (n : Nat) : allPoints [n] = (List.range n).map (fun i => [i]) := by
  have h : ∀ l : List Nat, l.flatMap (fun i => [[i]]) = l.map (fun i => [i]) := by
    intro l
    induction l with
    | nil => rfl
    | cons a l ih => simp [ih]
  simpa [allPoints] using h (List.range n)

theorem reduce_generic (limit : Nat) (ctx : Ctx) (call : CallSem) (env : Env)
    (ints : MerlinVerify.Tensor) (bits : Nat → Bool) (n : Nat)
    (hs : ints.shape = [n])
    (hv : ∀ i, i < n → ints.get [i] = bitval (if bits i then 1 else 0)) :
    genericSem ctx [identityMap 1, outMap] [.reduction] [.tensor ints] [.tensor (scalarTensor 0)]
      (fun inner => evalRegion (FrontendMaskRuntime.sem limit) inner call env sumBody) =
      .ok [.tensor (scalarTensor (countFrom bits n 0))] := by
  have hfold := reduce_fold limit ctx call env ints bits n hs hv n 0 0 (by omega)
  rw [← List.range_eq_range', ← points_one] at hfold
  unfold genericSem
  simp [shapeOf, hs, scalarTensor, Tensor.full, Tensor.canon, identityMap, outMap, loopRanges, loopRange,
    List.mapM_cons, List.mapM_nil, Outcome.bind, Outcome.ofOption]
  simp [identityMap, outMap, scalarTensor, Tensor.full, Tensor.canon] at hfold
  rw [hfold]
  change Outcome.ok [RVal.tensor (scalarTensor (countFrom bits n 0)).canonize] = _
  simp only [scalar_canon]
  rfl

def extended (bits : Nat → Bool) (n : Nat) : MerlinVerify.Tensor :=
  Tensor.canon [n] (.int 64) fun p => bitval (if bits (p.headD 0) then 1 else 0)

theorem extended_get (bits : Nat → Bool) (n i : Nat) (hi : i < n) :
    (extended bits n).get [i] = bitval (if bits i then 1 else 0) := by
  simp [extended, Tensor.canon, one_inBounds.mpr hi]
  rfl

/-- The count's 0/1 tensor is produced by the actual extui region, including poison outs. -/
theorem ext_generic (limit : Nat) (ctx : Ctx) (call : CallSem) (env : Env) (mask : MerlinVerify.Tensor)
    (bits : Nat → Bool) (n : Nat) (hm : MaskMatches mask bits n) :
    genericSem ctx [identityMap 1, identityMap 1] [.parallel] [.tensor mask]
      [.tensor (Tensor.full [n] (.int 64) .poison)]
      (fun inner => evalRegion (FrontendMaskRuntime.sem limit) inner call env extBody) =
        .ok [.tensor (extended bits n)] := by
  apply genericSem_parallel1 ctx [identityMap 1, identityMap 1] [.parallel] [.tensor mask]
    (Tensor.full [n] (.int 64) .poison)
    (fun inner => evalRegion (FrontendMaskRuntime.sem limit) inner call env extBody) [n]
    (fun p => bitval (if bits (p.headD 0) then 1 else 0)) (by rfl) (by cbv)
  · simp [shapeOf, hm.1, identityMap, Tensor.full, Tensor.canon]
  · simp [loopRanges, loopRange, identityMap, shapeOf, hm.1, Tensor.full, Tensor.canon,
      List.mapM_cons, List.mapM_nil, Outcome.ofOption, Outcome.bind]
  · rfl
  · rfl
  · intro p hp
    obtain ⟨i, rfl, hi⟩ : ∃ i, p = [i] ∧ i < n := by
      cases p with
      | nil => exact (Idx.inBounds_cons_nil n [] hp).elim
      | cons i rest =>
        obtain ⟨hi, hr⟩ := (Idx.inBounds_cons_cons n [] i rest).mp hp
        have he := (Idx.inBounds_nil_iff rest).mp hr
        subst rest
        exact ⟨i, rfl, hi⟩
    have hr := readAt_identity (t := mask) (pt := [i]) (by rw [hm.1]; exact one_inBounds.mpr hi)
    have hr' : readAt (.tensor mask) (identityMap 1) [i] = .ok (Arith.boolVal (bits i)) := by
      simpa [hm.1, hm.2 i hi] using hr
    have hz := readAt_identity (t := Tensor.full [n] (.int 64) .poison) (pt := [i])
      (one_inBounds.mpr hi)
    have hz' : readAt (.tensor (Tensor.full [n] (.int 64) .poison)) (identityMap 1) [i] =
        .ok .poison := by
      simpa [Tensor.full, Tensor.canon, one_inBounds.mpr hi] using hz
    simp [bodyAt, List.mapM_cons, List.mapM_nil, hr', hz', ext_body_sem]

/-- The two upstream structured operations compute the exact initialized mathematical count.
No universal floating-point comparison law is obtained from native samples. -/
theorem mask_count (limit : Nat) (ctx : Ctx) (call : CallSem) (env : Env) (mask : MerlinVerify.Tensor)
    (bits : Nat → Bool) (n : Nat) (hm : MaskMatches mask bits n) :
    genericSem ctx [identityMap 1, identityMap 1] [.parallel] [.tensor mask]
      [.tensor (Tensor.full [n] (.int 64) .poison)]
      (fun inner => evalRegion (FrontendMaskRuntime.sem limit) inner call env extBody) =
        .ok [.tensor (extended bits n)] ∧
    genericSem ctx [identityMap 1, outMap] [.reduction] [.tensor (extended bits n)]
      [.tensor (scalarTensor 0)]
      (fun inner => evalRegion (FrontendMaskRuntime.sem limit) inner call env sumBody) =
        .ok [.tensor (scalarTensor (countFrom bits n 0))] :=
  ⟨ext_generic limit ctx call env mask bits n hm,
    reduce_generic limit ctx call env (extended bits n) bits n rfl (extended_get bits n)⟩

#print axioms mask_count
end MerlinVerify.Rules.FrontendMaskCount
