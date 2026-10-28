/-
Shared lemmas for the per-family safety proofs: elementwise lifting (`liftEager`) on typed operands.
-/
import MerlinVerify.Check.Support
import MerlinVerify.Sem.OutcomeLaws

namespace MerlinVerify.Check

open MerlinVerify

/-- An outcome is safe for `P`: it is not stuck, and every value it yields satisfies `P`. -/
abbrev Safe {α : Type} (P : α → Prop) (o : Outcome α) : Prop := o.NotStuck ∧ ∀ a, o = .ok a → P a

theorem Safe.of_ok {α : Type} {P : α → Prop} {a : α} (h : P a) : Safe P (.ok a) :=
  ⟨trivial, fun _ h' => by cases h'; exact h⟩

theorem Safe.of_ub {α : Type} {P : α → Prop} : Safe P (.ub : Outcome α) := ⟨trivial, fun _ h => nomatch h⟩

theorem Safe.bind {α β : Type} {P : α → Prop} {Q : β → Prop} {o : Outcome α} {f : α → Outcome β}
    (ho : Safe P o) (hf : ∀ a, P a → Safe Q (f a)) : Safe Q (o.bind f) := by
  cases o with
  | ub => exact Safe.of_ub
  | stuck r => exact (ho.1 : False).elim
  | ok a => exact hf a (ho.2 a rfl)

/-- `List.mapM` in `Outcome`: if every element's step is not stuck and every `ok` value satisfies
`P`, the whole map is not stuck and every produced value satisfies `P`. -/
theorem listMapM_safe {α β : Type} (f : α → Outcome β) (P : β → Prop) :
    ∀ (l : List α), (∀ x ∈ l, (f x).NotStuck ∧ ∀ y, f x = .ok y → P y) →
      (l.mapM f).NotStuck ∧ ∀ ys, l.mapM f = .ok ys → ∀ y ∈ ys, P y
  | [], _ => ⟨trivial, fun ys h => by simp only [List.mapM_nil, Outcome.pure_eq] at h; cases h; simp⟩
  | x :: l, h => by
    obtain ⟨hns, hok⟩ := h x (List.mem_cons_self ..)
    have ih := listMapM_safe f P l (fun y hy => h y (List.mem_cons_of_mem _ hy))
    simp only [List.mapM_cons, Outcome.bindOp_eq]
    cases hfx : f x with
    | ub => exact ⟨trivial, fun ys hys => by cases hys⟩
    | stuck r => rw [hfx] at hns; exact absurd hns (by simp [Outcome.NotStuck])
    | ok y =>
      rw [Outcome.bind_ok]
      cases hl : l.mapM f with
      | ub => exact ⟨trivial, fun ys hys => by cases hys⟩
      | stuck r => rw [hl] at ih; exact absurd ih.1 (by simp [Outcome.NotStuck])
      | ok ys =>
        rw [Outcome.bind_ok, Outcome.pure_eq]
        refine ⟨trivial, fun zs hzs => ?_⟩
        cases hzs
        intro z hz
        rcases List.mem_cons.1 hz with hzy | hz
        · rw [hzy]; exact hok y hfx
        · exact ih.2 ys hl z hz

theorem mapM_scalars (g : RVal → Option Val) (hg : ∀ v, g (.scalar v) = some v) (vs : List Val) :
    (vs.map RVal.scalar).mapM g = some vs := by
  induction vs with
  | nil => rfl
  | cons v vs ih => simp [List.mapM_cons, hg, ih]

theorem mapM_tensors (g : RVal → Option Tensor) (hg : ∀ t, g (.tensor t) = some t) (ts : List Tensor) :
    (ts.map RVal.tensor).mapM g = some ts := by
  induction ts with
  | nil => rfl
  | cons t ts ih => simp [List.mapM_cons, hg, ih]

theorem liftEager_scalars (resElt : Ty) (f : List Val → Outcome Val) (vs : List Val) :
    liftEager resElt f (vs.map RVal.scalar) = (f vs).bind fun v => .ok (.scalar v) := by
  unfold liftEager
  split
  · next vs' h => rw [mapM_scalars _ (fun _ => rfl)] at h; cases h; rfl
  · next h => rw [mapM_scalars _ (fun _ => rfl)] at h; cases h

theorem liftEager_tensors (resElt : Ty) (f : List Val → Outcome Val) (t : Tensor) (ts : List Tensor)
    (hs : ∀ u ∈ ts, u.shape = t.shape) :
    liftEager resElt f ((t :: ts).map RVal.tensor) =
      ((allPoints t.shape).mapM fun i => f ((t :: ts).map (·.get i))).bind fun vals =>
        .ok (.tensor (Tensor.canon t.shape resElt fun i =>
          vals.toArray[linearIndex t.shape i]?.getD .poison)) := by
  have hall : (ts.all fun u => u.shape == t.shape) = true := by
    simp only [List.all_eq_true, beq_iff_eq]; exact hs
  unfold liftEager
  split
  · next vs h => simp [List.mapM_cons] at h
  · split
    · next t' ts' h =>
      rw [mapM_tensors _ (fun _ => rfl)] at h
      cases h
      simp only [hall, ↓reduceIte]
    · rename_i hne
      exact (hne t ts (mapM_tensors _ (fun _ => rfl) (t :: ts))).elim

end MerlinVerify.Check
