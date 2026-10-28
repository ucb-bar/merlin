/- Exact replacement of a structured body on its checked scalar argument domain. The loop
invariant is inherited from actual source execution, including poison and exceptional outcomes. -/
import MerlinVerify.Check.StructuredRuntime

namespace MerlinVerify.Rules.StructuredBody

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops.Linalg

def Equal (ib : Nat) (sig : RegionSig) (source target : Ctx → RegionSem) : Prop :=
  ∀ ctx, ctx.indexBits = ib → sig.depth ≤ ctx.loopIndex.length →
    ∀ values, HasTys ib values sig.args → source ctx values = target ctx values

theorem bodyAt_equal {ib : Nat} {ranges pt : List Nat} (hp : Idx.InBounds ranges pt)
    {ctx : Ctx} (hib : ctx.indexBits = ib) {ins : List RVal} {inMaps outMaps : List AffineMap}
    {cur : List Tensor} {source target : Ctx → RegionSem} {inTys outTys : List VTy}
    (hi : HasTys ib ins inTys) (ho : HasTys ib (cur.map RVal.tensor) outTys)
    (hmi : MapsInBounds ranges inMaps inTys) (hmo : MapsInBounds ranges outMaps outTys)
    (he : Equal ib ⟨(inTys ++ outTys).map VTy.eltScalar,
      outTys.map VTy.eltScalar, ranges.length⟩ source target) :
    bodyAt ctx ins inMaps outMaps cur source pt = bodyAt ctx ins inMaps outMaps cur target pt := by
  have hreadI := mapM_read_safe hp id (fun (v, m) => readAt v m pt) (by intro _ _; rfl) ins inMaps inTys
    (by simpa using hi) hmi
  have hreadO := mapM_read_safe hp RVal.tensor (fun (t, m) => readAt (.tensor t) m pt) (by intro _ _; rfl) cur outMaps outTys ho hmo
  unfold bodyAt
  simp only [Outcome.bindOp_eq]
  cases hx : (ins.zip inMaps).mapM (fun (v, m) => readAt v m pt) with
  | ub => rfl
  | stuck _ => rfl
  | ok xs =>
    have hxs := hreadI.2 xs hx
    simp only [Outcome.bind_ok]
    cases hy : (cur.zip outMaps).mapM (fun (t, m) => readAt (.tensor t) m pt) with
    | ub => rfl
    | stuck _ => rfl
    | ok ys =>
      have hys := hreadO.2 ys hy
      simp only [Outcome.bind_ok]
      have ht : HasTys ib ((xs ++ ys).map RVal.scalar)
          ((inTys ++ outTys).map VTy.eltScalar) := by
        rw [List.map_append, List.map_append]
        exact List.rel_append (valsTyped_scalars hxs) (valsTyped_scalars hys)
      rw [he {ctx with loopIndex := pt} hib (Nat.le_of_eq hp.1.symm) _ ht]

theorem foldlM_equal {α β : Type} (f g : β → α → Outcome β) (I : β → Prop) (P : α → Prop)
    (hf : ∀ state point, I state → P point → Safe I (f state point))
    (he : ∀ state point, I state → P point → f state point = g state point) :
    ∀ (points : List α), (∀ point ∈ points, P point) → ∀ state, I state →
      points.foldlM f state = points.foldlM g state
  | [], _, _, _ => rfl
  | point :: points, hp, state, hs => by
    rw [List.foldlM_cons, List.foldlM_cons, ← he state point hs (hp point (by simp))]
    cases hx : f state point with
    | ub => rfl
    | stuck _ => rfl
    | ok next =>
      exact foldlM_equal f g I P hf he points (fun p hm => hp p (by simp [hm])) next
        ((hf state point hs (hp point (by simp))).2 next hx)

/-- Equality is needed only at arguments reached by a checked loop. A source-body safety proof
maintains the output types between iterations; target exceptional behavior cannot be hidden. -/
theorem genericSem_equal {ib : Nat} {ctx : Ctx} (hib : ctx.indexBits = ib)
    {maps : List AffineMap} {iters : List IterKind} {inTys outTys : List VTy}
    {ranges : List Nat} {ins : List RVal} {outs : List Tensor} {source target : Ctx → RegionSem}
    (hs : StructuredOk maps iters.length (inTys ++ outTys) ranges)
    (hi : HasTys ib ins inTys) (ho : HasTys ib (outs.map RVal.tensor) outTys)
    (hb : RegionSafe ib source
      ⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, iters.length⟩)
    (he : Equal ib
      ⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, iters.length⟩ source target) :
    genericSem ctx maps iters ins (outs.map RVal.tensor) source =
      genericSem ctx maps iters ins (outs.map RVal.tensor) target := by
  have hlen : ranges.length = iters.length := by
    rw [optionMapM_length _ _ _ hs.loops, List.length_range]
  have hk : ins.length = inTys.length := HasTys.length hi
  have hmi : MapsInBounds ranges (maps.take ins.length) inTys := by
    have h := List.forall₂_take inTys.length hs.inBounds
    rwa [List.take_left, ← hk] at h
  have hmo : MapsInBounds ranges (maps.drop ins.length) outTys := by
    have h := List.forall₂_drop inTys.length hs.inBounds
    rwa [List.drop_left, ← hk] at h
  have hb' : RegionSafe ib source
      ⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, ranges.length⟩ := by
    rw [hlen]; exact hb
  have he' : Equal ib
      ⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, ranges.length⟩ source target := by
    rw [hlen]; exact he
  rw [genericSem_execute hs hi ho, genericSem_execute hs hi ho]
  congr 1
  apply foldlM_equal _ _ (fun cur => HasTys ib (cur.map RVal.tensor) outTys) (Idx.InBounds ranges)
  · intro cur pt hcur hp
    exact loopStep_safe hp hib hi hcur hmi hmo hb'
  · intro cur pt hcur hp
    unfold loopStep
    rw [bodyAt_equal hp hib hi hcur hmi hmo he']
  · intro pt hp
    exact (mem_allPoints ranges pt).1 hp
  · exact ho

end MerlinVerify.Rules.StructuredBody
