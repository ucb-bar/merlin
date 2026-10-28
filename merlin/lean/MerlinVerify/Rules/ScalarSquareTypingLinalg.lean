/- Successful-result typing and typed callback congruence. Stuck and UB are retained,
not discharged. No numerical identity is needed. -/
import MerlinVerify.Check.Rulebook
import MerlinVerify.Rules.ScalarSquareClaims

namespace MerlinVerify.Rules.ScalarSquareTyping
open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open MerlinVerify.Sem.Ops.Linalg

/-- Equal outcomes, with typed successful results. Neither progress nor poison freedom. -/
def Related {α : Type} (P : α → Prop) (left right : Outcome α) : Prop :=
  left = right ∧ ∀ value, left = .ok value → P value

theorem Related.refl_safe {α : Type} {P : α → Prop} {o : Outcome α}
    (h : Safe P o) : Related P o o := ⟨rfl, h.2⟩
theorem Related.ok {α : Type} {P : α → Prop} {x : α} (h : P x) :
    Related P (.ok x) (.ok x) := ⟨rfl, fun _ e => by cases e; exact h⟩
theorem Related.bind {α β : Type} {P : α → Prop} {Q : β → Prop}
    {s t : Outcome α} {f g : α → Outcome β} (h : Related P s t)
    (step : ∀ x, P x → Related Q (f x) (g x)) : Related Q (s.bind f) (t.bind g) := by
  rw [← h.1]
  cases s with
  | ub => exact ⟨rfl, fun _ e => nomatch e⟩
  | stuck _ => exact ⟨rfl, fun _ e => nomatch e⟩
  | ok x => exact step x (h.2 x rfl)

def RegionRelated (M : FloatModel) (ib : Nat)
    (bodies : (Ctx → RegionSem) × (Ctx → RegionSem)) (sig : RegionSig) : Prop :=
  ∀ ctx, ctx.M = M → ctx.indexBits = ib → sig.depth ≤ ctx.loopIndex.length →
    ∀ args, HasTys ib args sig.args →
      Related (HasTys ib · sig.yields) (bodies.1 ctx args) (bodies.2 ctx args)

theorem fold_related {α β : Type} (P : α → Prop) (Q : β → Prop)
    (f g : α → β → Outcome α)
    (step : ∀ a b, P a → Q b → Related P (f a b) (g a b)) :
    ∀ (xs : List β) (init : α), (∀ x ∈ xs, Q x) → P init →
      Related P (xs.foldlM f init) (xs.foldlM g init)
  | [], init, _, hi => Related.ok hi
  | x :: xs, init, hx, hi => by
    simp only [List.foldlM_cons, Outcome.bindOp_eq]
    exact Related.bind (step init x hi (hx x (by simp))) fun next hn =>
      fold_related P Q f g step xs next (fun y hy => hx y (by simp [hy])) hn

theorem bodyAt_related {M : FloatModel} {ib : Nat} {ranges pt : List Nat}
    (hpt : Idx.InBounds ranges pt) {ctx : Ctx} (hm : ctx.M = M) (hib : ctx.indexBits = ib)
    {ins : List RVal} {inMaps outMaps : List AffineMap} {cur : List Tensor}
    {bodyS bodyT : Ctx → RegionSem} {inTys outTys : List VTy}
    (hins : HasTys ib ins inTys) (hcur : HasTys ib (cur.map RVal.tensor) outTys)
    (hinM : MapsInBounds ranges inMaps inTys) (houtM : MapsInBounds ranges outMaps outTys)
    (hbody : RegionRelated M ib (bodyS, bodyT)
      ⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, ranges.length⟩) :
    Related (ValsTyped ib · (outTys.map eltOfVTy))
      (bodyAt ctx ins inMaps outMaps cur bodyS pt)
      (bodyAt ctx ins inMaps outMaps cur bodyT pt) := by
  unfold bodyAt
  simp only [Outcome.bindOp_eq]
  refine Related.bind (.refl_safe (mapM_read_safe hpt id _ (by intro _ _; rfl)
    ins inMaps inTys (by simpa using hins) hinM)) fun xs hxs => ?_
  refine Related.bind (.refl_safe (mapM_read_safe hpt RVal.tensor _ (by intro _ _; rfl)
    cur outMaps outTys hcur houtM)) fun ys hys => ?_
  have hvals : HasTys ib ((xs ++ ys).map RVal.scalar) ((inTys ++ outTys).map VTy.eltScalar) := by
    rw [List.map_append, List.map_append]
    exact List.rel_append (valsTyped_scalars hxs) (valsTyped_scalars hys)
  refine Related.bind (hbody { ctx with loopIndex := pt } hm hib
    (Nat.le_of_eq hpt.1.symm) _ hvals) fun zs hzs => ?_
  have hlen : zs.length = cur.length := by
    rw [HasTys.length hzs, List.length_map, ← HasTys.length hcur, List.length_map]
  simp only [hlen, ne_eq, not_true_eq_false, ↓reduceIte]
  obtain ⟨vs, rfl, hvs⟩ := scalars_of_typed hzs
  exact .refl_safe (mapM_ofScalars_safe _ (by intro _; rfl) hvs)

theorem loopStep_related {M : FloatModel} {ib : Nat} {ranges pt : List Nat}
    (hpt : Idx.InBounds ranges pt) {ctx : Ctx} (hm : ctx.M = M) (hib : ctx.indexBits = ib)
    {ins : List RVal} {inMaps outMaps : List AffineMap} {cur : List Tensor}
    {bodyS bodyT : Ctx → RegionSem} {inTys outTys : List VTy}
    (hins : HasTys ib ins inTys) (hcur : HasTys ib (cur.map RVal.tensor) outTys)
    (hinM : MapsInBounds ranges inMaps inTys) (houtM : MapsInBounds ranges outMaps outTys)
    (hbody : RegionRelated M ib (bodyS, bodyT)
      ⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, ranges.length⟩) :
    Related (fun cur' => HasTys ib (cur'.map RVal.tensor) outTys)
      (loopStep ctx ins inMaps outMaps bodyS cur pt)
      (loopStep ctx ins inMaps outMaps bodyT cur pt) := by
  unfold loopStep
  simp only [Outcome.bindOp_eq]
  exact Related.bind (bodyAt_related hpt hm hib hins hcur hinM houtM hbody) fun vs hvs =>
    .refl_safe (mapM_write_safe hpt _ (by intro _ _ _; rfl) vs cur outMaps outTys hvs hcur houtM)

theorem genericSem_related {M : FloatModel} {ib : Nat} {ctx : Ctx} (hm : ctx.M = M) (hib : ctx.indexBits = ib) {maps : List AffineMap}
    {iters : List IterKind} {ins outs : List RVal} {bodyS bodyT : Ctx → RegionSem} {inTys outTys : List VTy}
    {ranges : List Nat} (hs : StructuredOk maps iters.length (inTys ++ outTys) ranges)
    (hout : ∀ t ∈ outTys, t.isTensor = true) (hins : HasTys ib ins inTys)
    (houts : HasTys ib outs outTys)
    (hbody : RegionRelated M ib (bodyS, bodyT)
      ⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, iters.length⟩) :
    Related (HasTys ib · outTys) (genericSem ctx maps iters ins outs bodyS)
      (genericSem ctx maps iters ins outs bodyT) := by
  obtain ⟨ts, rfl⟩ := tensors_of_typed houts hout
  have hall : HasTys ib (ins ++ ts.map RVal.tensor) (inTys ++ outTys) := List.rel_append hins houts
  have hshapes := shapes_typed hall
  have hlenR : ranges.length = iters.length := by
    rw [optionMapM_length _ _ _ hs.loops, List.length_range]
  have hL : maps.length = (ins ++ ts.map RVal.tensor).length := by rw [hs.len, HasTys.length hall]
  have hD : maps.any (fun m => m.numDims ≠ iters.length || m.numSyms ≠ 0) = false := by
    rw [List.any_eq_false]
    intro m hm
    simp [(hs.dims m hm).1, (hs.dims m hm).2]
  have hR : (maps.zip ((ins ++ ts.map RVal.tensor).map shapeOf)).any
      (fun p => p.1.results.length ≠ p.2.length) = false := by
    rw [hshapes, List.zip_map_right, List.any_map, List.any_eq_false]
    intro p hp
    have hin := List.forall₂_iff_zip.1 hs.inBounds |>.2 hp
    simp only [mapInBounds, Bool.and_eq_true, beq_iff_eq] at hin
    simp [Function.comp, hin.1]
  have hLR : loopRanges maps ((ins ++ ts.map RVal.tensor).map shapeOf) iters.length = .ok ranges := by
    rw [hshapes]
    exact mapM_ofOption _ _ _ _ hs.loops
  have hk : ins.length = inTys.length := HasTys.length hins
  have hinM : MapsInBounds ranges (maps.take ins.length) inTys := by
    have := List.forall₂_take inTys.length hs.inBounds
    rwa [List.take_left, ← hk] at this
  have houtM : MapsInBounds ranges (maps.drop ins.length) outTys := by
    have := List.forall₂_drop inTys.length hs.inBounds
    rwa [List.drop_left, ← hk] at this
  have hbody' : RegionRelated M ib (bodyS, bodyT)
      ⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, ranges.length⟩ := by
    rw [hlenR]; exact hbody
  unfold genericSem
  simp only [hL, ne_eq, not_true_eq_false, ↓reduceIte, hD, Bool.false_eq_true, hR, Outcome.bindOp_eq]
  refine Related.bind (.refl_safe (mapM_ofTensors_safe _ (by intro _; rfl) ts)) fun ts' hts' => ?_
  subst hts'
  simp only [hLR, Outcome.bind_ok]
  refine Related.bind (fold_related (fun cur => HasTys ib (cur.map RVal.tensor) outTys)
    (Idx.InBounds ranges) _ _ (fun cur pt hcur hpt => loopStep_related hpt hm hib hins hcur hinM houtM hbody')
    _ _ (fun pt hp => (mem_allPoints _ _).1 hp) houts) fun final hfinal => ?_
  exact Related.ok (hasTys_canonize hfinal)


end MerlinVerify.Rules.ScalarSquareTyping
