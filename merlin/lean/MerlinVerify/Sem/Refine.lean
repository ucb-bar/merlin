/-
G0-FROZEN. Refinement: "the target is at least as defined as the source, and agrees with it
wherever the source is defined" (design §4.5).

`tgt ⊑ src` reads `Refines tgt src`.
* A `poison` source element is refined by anything.
* A `ub` source outcome is refined by anything.
* A `stuck` outcome refines nothing and is refined by nothing, so an unmodelled construct on either
  side defeats a claim instead of satisfying it.
-/
import MerlinVerify.Sem.Outcome
import MerlinVerify.Sem.Value
import Mathlib.Data.List.Forall2

namespace MerlinVerify

def ValRefines (tgt src : Val) : Prop :=
  src = .poison ∨ tgt = src

def TensorRefines (tgt src : Tensor) : Prop :=
  tgt.shape = src.shape ∧ tgt.elt = src.elt ∧
    ∀ i, Idx.InBounds src.shape i → ValRefines (tgt.get i) (src.get i)

def RValRefines : RVal → RVal → Prop
  | .scalar t, .scalar s => ValRefines t s
  | .tensor t, .tensor s => TensorRefines t s
  | _, _ => False

def Refines {α : Type} (R : α → α → Prop) : Outcome α → Outcome α → Prop
  | _, .ub => True
  | .ok t, .ok s => R t s
  | _, _ => False

/-- Refinement of a list of results, position by position. -/
def ResultsRefine (tgt src : List RVal) : Prop :=
  List.Forall₂ RValRefines tgt src

theorem ValRefines.refl (v : Val) : ValRefines v v := Or.inr rfl

theorem ValRefines.trans {a b c : Val} (h₁ : ValRefines a b) (h₂ : ValRefines b c) :
    ValRefines a c := by
  rcases h₂ with h | h
  · exact Or.inl h
  · subst h; exact h₁

theorem TensorRefines.refl (t : Tensor) : TensorRefines t t :=
  ⟨rfl, rfl, fun _ _ => ValRefines.refl _⟩

theorem TensorRefines.trans {a b c : Tensor} (h₁ : TensorRefines a b) (h₂ : TensorRefines b c) :
    TensorRefines a c := by
  obtain ⟨s₁, e₁, v₁⟩ := h₁
  obtain ⟨s₂, e₂, v₂⟩ := h₂
  refine ⟨s₁.trans s₂, e₁.trans e₂, fun i hi => ?_⟩
  exact ValRefines.trans (v₁ i (s₂ ▸ hi)) (v₂ i hi)

theorem RValRefines.refl (v : RVal) : RValRefines v v := by
  cases v with
  | scalar v => exact ValRefines.refl v
  | tensor t => exact TensorRefines.refl t

theorem RValRefines.trans {a b c : RVal} (h₁ : RValRefines a b) (h₂ : RValRefines b c) :
    RValRefines a c := by
  cases a <;> cases b <;> cases c <;> simp only [RValRefines] at h₁ h₂ ⊢
  · exact ValRefines.trans h₁ h₂
  · exact TensorRefines.trans h₁ h₂

theorem ResultsRefine.refl (vs : List RVal) : ResultsRefine vs vs := by
  induction vs with
  | nil => exact List.Forall₂.nil
  | cons v vs ih => exact List.Forall₂.cons (RValRefines.refl v) ih

theorem ResultsRefine.trans {a b c : List RVal} (h₁ : ResultsRefine a b) (h₂ : ResultsRefine b c) :
    ResultsRefine a c := by
  induction h₁ generalizing c with
  | nil => cases h₂; exact List.Forall₂.nil
  | cons hab _ ih =>
    cases h₂ with
    | cons hbc hrest => exact List.Forall₂.cons (RValRefines.trans hab hbc) (ih hrest)

theorem Refines.trans {α : Type} {R : α → α → Prop} (hR : ∀ {a b c}, R a b → R b c → R a c)
    {x y z : Outcome α} (h₁ : Refines R x y) (h₂ : Refines R y z) : Refines R x z := by
  cases z with
  | ub => trivial
  | stuck r => cases y <;> simp [Refines] at h₂
  | ok c =>
    cases y with
    | ub => simp [Refines] at h₂
    | stuck r => simp [Refines] at h₂
    | ok b =>
      cases x with
      | ub => simp [Refines] at h₁
      | stuck r => simp [Refines] at h₁
      | ok a => exact hR h₁ h₂

theorem Refines.refl_ok {α : Type} {R : α → α → Prop} (hR : ∀ a, R a a) (a : α) :
    Refines R (.ok a) (.ok a) := hR a

end MerlinVerify
