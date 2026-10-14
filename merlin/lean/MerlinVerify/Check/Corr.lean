/-
Value correspondence between a source and a target program (design §5.1, alpha-equivalence).

`ρ : Rho` pairs source SSA ids with target SSA ids, newest first. `Corr ρ envS envT` says every
source id mapped by `ρ` looks up the same value as its image. Binding fresh target ids keeps it.
-/
import MerlinVerify.Sem.Program

namespace MerlinVerify.Check

open MerlinVerify

abbrev Rho := List (ValueId × ValueId)

/-- The newest target id paired with source id `x`. -/
def Rho.find (ρ : Rho) (x : ValueId) : Option ValueId := List.lookup x ρ

def Corr (ρ : Rho) (envS envT : Env) : Prop :=
  ∀ x y, ρ.find x = some y → envS.get? x = envT.get? y

/-- `y` is not the image of any pair of `ρ`. -/
def Rho.notImage (ρ : Rho) (y : ValueId) : Prop := ∀ p ∈ ρ, p.2 ≠ y

theorem lookup_mem_image {ρ : Rho} {x y : ValueId} (h : ρ.find x = some y) : ∃ p ∈ ρ, p.2 = y := by
  unfold Rho.find at h
  induction ρ with
  | nil => simp [List.lookup] at h
  | cons p ρ ih =>
    obtain ⟨k, v⟩ := p
    simp only [List.lookup] at h
    split at h
    · exact ⟨(k, v), List.mem_cons_self .., by simp_all⟩
    · obtain ⟨q, hq, hq2⟩ := ih (by simpa [Rho.find] using h)
      exact ⟨q, List.mem_cons_of_mem _ hq, hq2⟩

theorem lookup_zip_append_image {xs ys : List ValueId} {ρ : Rho} {x y : ValueId}
    (h : Rho.find (xs.zip ys ++ ρ) x = some y) : y ∈ ys ∨ ∃ p ∈ ρ, p.2 = y := by
  obtain ⟨p, hp, hp2⟩ := lookup_mem_image h
  rcases List.mem_append.1 hp with hz | hr
  · left; rw [← hp2]; exact (List.of_mem_zip hz).2
  · right; exact ⟨p, hr, hp2⟩

/-- Binding the same values to source ids `xs` and fresh, distinct target ids `ys` extends the
correspondence by `xs.zip ys`. -/
theorem Corr.extend {ρ : Rho} {envS envT : Env} (h : Corr ρ envS envT) :
    ∀ (xs ys : List ValueId) (vs : List RVal), xs.length = vs.length → ys.length = vs.length →
      ys.Nodup → (∀ y ∈ ys, Rho.notImage ρ y) →
      Corr (xs.zip ys ++ ρ) (xs.zip vs ++ envS) (ys.zip vs ++ envT)
  | [], [], [], _, _, _, _ => by simpa using h
  | x :: xs, y :: ys, v :: vs, h1, h2, hnd, hfr => by
    have ih := Corr.extend h xs ys vs (by simpa using h1) (by simpa using h2)
      (List.nodup_cons.1 hnd).2 (fun y' hy' => hfr y' (List.mem_cons_of_mem _ hy'))
    intro x' y' hl
    simp only [Rho.find, List.zip_cons_cons, List.cons_append, List.lookup] at hl
    simp only [List.zip_cons_cons, List.cons_append, Env.get?, List.lookup]
    by_cases hx : x' = x
    · subst hx
      simp only [beq_self_eq_true, ↓reduceIte, Option.some.injEq] at hl
      subst hl
      simp
    · have hxb : (x' == x) = false := by simpa using hx
      rw [hxb] at hl
      simp only [hxb, Bool.false_eq_true, ↓reduceIte]
      simp only [Bool.false_eq_true, ↓reduceIte] at hl
      have hy : (y' == y) = false := by
        rcases lookup_zip_append_image hl with hin | ⟨p, hp, hp2⟩
        · have : y ∉ ys := (List.nodup_cons.1 hnd).1
          simp only [beq_eq_false_iff_ne]
          rintro rfl; exact this hin
        · simp only [beq_eq_false_iff_ne]
          rintro rfl
          exact hfr y' (List.mem_cons_self ..) p hp hp2
      simp only [hy, Bool.false_eq_true, ↓reduceIte]
      exact ih x' y' hl
  | [], _ :: _, _, h1, h2, _, _ => by simp at h1 h2; omega
  | _ :: _, [], _, h1, h2, _, _ => by simp at h1 h2; omega
  | [], [], _ :: _, h1, _, _, _ => by simp at h1
  | _ :: _, _ :: _, [], h1, _, _, _ => by simp at h1

/-- Corresponding operands look up equal values. -/
theorem Corr.getAll {ρ : Rho} {envS envT : Env} (h : Corr ρ envS envT) :
    ∀ (xs ys : List ValueId), xs.mapM ρ.find = some ys → envS.getAll xs = envT.getAll ys
  | [], [], _ => rfl
  | x :: xs, ys, hm => by
    simp only [List.mapM_cons] at hm
    cases hx : ρ.find x with
    | none => simp [hx] at hm
    | some y =>
      cases hxs : xs.mapM ρ.find with
      | none => simp [hx, hxs] at hm
      | some ys' =>
        simp only [hx, hxs, Option.bind_some, Option.pure_def, bind, Option.bind,
          Option.some.injEq] at hm
        subst hm
        have ih := Corr.getAll h xs ys' hxs
        simp only [Env.getAll, Outcome.mapM] at ih ⊢
        rw [h x y hx, ih]
  | [], _ :: _, hm => by simp at hm

end MerlinVerify.Check
