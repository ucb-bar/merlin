/-
Elementwise ops (scalars, or tensors of one shape): typing and the generic safety lemma.

`elementwise?` reads the element types and common shape off an op's operand/result types;
`pureN_safe` proves an elementwise op safe from a scalar-level fact about its element function.
-/
import MerlinVerify.Check.RulesBase
import MerlinVerify.Sem.Ops.Arith
import MerlinVerify.Sem.Facts

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops

/-- A scalar type, or a tensor of it with shape `sh`. -/
def liftTy : Option (List Nat) → Ty → VTy
  | none, t => .scalar t
  | some sh, t => .tensor sh t

def shapeOfVTy : VTy → Option (List Nat)
  | .scalar _ => none
  | .tensor sh _ => some sh

def eltOfVTy : VTy → Ty
  | .scalar t => t
  | .tensor _ t => t

/-- Elementwise typing: every operand and the single result share one shape (or are all scalars);
returns the shape, the operand element types and the result element type. -/
def elementwise? (argTys resTys : List VTy) : Option (Option (List Nat) × List Ty × Ty) :=
  match argTys, resTys with
  | a :: as, [r] =>
    let sh := shapeOfVTy a
    let elts := (a :: as).map eltOfVTy
    if (a :: as) = elts.map (liftTy sh) ∧ r = liftTy sh (eltOfVTy r) then some (sh, elts, eltOfVTy r)
    else none
  | _, _ => none

/-- Scalar values have the given element types, position by position. -/
def ValsTyped (ib : Nat) (vs : List Val) (ts : List Ty) : Prop :=
  List.Forall₂ (fun v t => Val.HasTy ib t v) vs ts

theorem elementwise?_spec {argTys resTys : List VTy} {sh : Option (List Nat)} {elts : List Ty} {re : Ty}
    (h : elementwise? argTys resTys = some (sh, elts, re)) :
    argTys = elts.map (liftTy sh) ∧ resTys = [liftTy sh re] ∧ elts ≠ [] := by
  unfold elementwise? at h
  simp only at h
  split at h
  · next a as r =>
    split at h
    · next hc =>
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      exact ⟨hc.1, by rw [← hc.2], by simp⟩
    · cases h
  · cases h

/-- Typed values of element types `elts` lifted to one shape. -/
theorem hasTys_lift_scalar {ib : Nat} {args : List RVal} {elts : List Ty}
    (h : HasTys ib args (elts.map (liftTy none))) :
    ∃ vs, args = vs.map RVal.scalar ∧ ValsTyped ib vs elts := by
  induction elts generalizing args with
  | nil => cases h; exact ⟨[], rfl, List.Forall₂.nil⟩
  | cons e es ih =>
    obtain _ | ⟨hv, hrest⟩ := h
    rename_i a args'
    obtain ⟨vs, rfl, hvs⟩ := ih hrest
    cases a with
    | scalar v => exact ⟨v :: vs, rfl, List.Forall₂.cons hv hvs⟩
    | tensor t => simp [RVal.HasTy, liftTy] at hv

theorem hasTys_lift_tensor {ib : Nat} {args : List RVal} {elts : List Ty} {sh : List Nat}
    (h : HasTys ib args (elts.map (liftTy (some sh)))) :
    ∃ ts, args = ts.map RVal.tensor ∧ List.Forall₂ (fun t e => RVal.HasTy ib (.tensor t) (.tensor sh e)) ts elts := by
  induction elts generalizing args with
  | nil => cases h; exact ⟨[], rfl, List.Forall₂.nil⟩
  | cons e es ih =>
    obtain _ | ⟨hv, hrest⟩ := h
    rename_i a args'
    obtain ⟨ts, rfl, hts⟩ := ih hrest
    cases a with
    | tensor t => exact ⟨t :: ts, rfl, List.Forall₂.cons hv hts⟩
    | scalar v => simp [RVal.HasTy, liftTy] at hv

/-- Elements of typed tensors at an in-bounds index are typed. -/
theorem forall₂_get {ib : Nat} {sh : List Nat} {i : Idx} (hin : Idx.InBounds sh i) :
    ∀ (ts : List Tensor) (elts : List Ty),
      List.Forall₂ (fun t e => RVal.HasTy ib (.tensor t) (.tensor sh e)) ts elts →
      ValsTyped ib (ts.map (·.get i)) elts
  | [], [], _ => List.Forall₂.nil
  | _ :: ts, _ :: es, .cons h hrest => List.Forall₂.cons (h.2.2 i hin) (forall₂_get hin ts es hrest)

theorem toArray_getD_mem {α : Type} (l : List α) (k : Nat) (d : α) :
    l.toArray[k]?.getD d = d ∨ l.toArray[k]?.getD d ∈ l := by
  by_cases hk : k < l.length
  · right
    simp [List.getElem?_eq_getElem hk, List.getElem_mem]
  · left
    simp [List.getElem?_eq_none (Nat.le_of_not_lt hk)]

/-- **Elementwise safety.** If the element function is never stuck on typed elements and yields
typed elements, the lifted op is never stuck on typed operands and yields a typed result. -/
theorem pureN_safe (ib : Nat) (op : Op) (args : List RVal) (g : List Val → Outcome Val)
    (sh : Option (List Nat)) (elts : List Ty) (re : Ty) (hne : elts ≠ [])
    (hres : Arith.resultElt op = some re)
    (hargs : HasTys ib args (elts.map (liftTy sh)))
    (hg : ∀ vs, ValsTyped ib vs elts →
      (g vs).NotStuck ∧ ∀ v, g vs = .ok v → Val.HasTy ib re v) :
    (Arith.pureN op args g).NotStuck ∧
      ∀ rs, Arith.pureN op args g = .ok rs → HasTys ib rs [liftTy sh re] := by
  unfold Arith.pureN
  rw [hres]
  simp only [Outcome.bindOp_eq, Outcome.pure_eq]
  cases sh with
  | none =>
    obtain ⟨vs, rfl, hvs⟩ := hasTys_lift_scalar hargs
    rw [liftEager_scalars]
    obtain ⟨hns, hok⟩ := hg vs hvs
    cases hgv : g vs with
    | ub => exact ⟨trivial, fun rs hrs => by cases hrs⟩
    | stuck r => rw [hgv] at hns; exact absurd hns (by simp [Outcome.NotStuck])
    | ok v =>
      refine ⟨trivial, fun rs hrs => ?_⟩
      cases hrs
      exact List.Forall₂.cons (hok v hgv) List.Forall₂.nil
  | some sh =>
    obtain ⟨ts, rfl, hts⟩ := hasTys_lift_tensor hargs
    cases ts with
    | nil => cases elts with
      | nil => exact absurd rfl hne
      | cons _ _ => cases hts
    | cons t ts =>
      have hshape : ∀ u ∈ ts, u.shape = t.shape := by
        intro u hu
        obtain _ | ⟨ht, hrest⟩ := hts
        obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hu
        have hu' := List.Forall₂.get hrest (by simpa using hk) (by rw [← List.Forall₂.length_eq hrest]; simpa using hk)
        rw [ht.1]
        exact (hu' : RVal.HasTy ib (.tensor ts[k]) (.tensor sh _)).1
      rw [liftEager_tensors _ _ t ts hshape]
      have hpt : ∀ i ∈ allPoints t.shape,
          (g ((t :: ts).map (·.get i))).NotStuck ∧ ∀ v, g ((t :: ts).map (·.get i)) = .ok v → Val.HasTy ib re v := by
        intro i hi
        apply hg
        have hin : Idx.InBounds sh i := by
          obtain _ | ⟨ht, _⟩ := hts
          rw [← ht.1]; exact (mem_allPoints _ _).1 hi
        exact forall₂_get hin _ _ hts
      obtain ⟨hns, hall⟩ := listMapM_safe _ (Val.HasTy ib re) _ hpt
      cases hm : (allPoints t.shape).mapM fun i => g ((t :: ts).map (·.get i)) with
      | ub => exact ⟨trivial, fun rs hrs => by cases hrs⟩
      | stuck r => rw [hm] at hns; exact absurd hns (by simp [Outcome.NotStuck])
      | ok vals =>
        refine ⟨trivial, fun rs hrs => ?_⟩
        cases hrs
        have hts' := hts
        obtain _ | ⟨ht, _⟩ := hts'
        refine List.Forall₂.cons ⟨by simp [Tensor.canon, ht.1], rfl, fun i hi => ?_⟩ List.Forall₂.nil
        simp only [Tensor.canon, ht.1, hi, ↓reduceIte]
        rcases toArray_getD_mem vals (linearIndex sh i) .poison with h | h
        · rw [h]; trivial
        · exact hall vals hm _ h

end MerlinVerify.Check
