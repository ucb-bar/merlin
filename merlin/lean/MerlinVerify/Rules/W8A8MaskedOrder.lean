/- Numerical order for genuine -Inf masks and finite scores. The actual ordered maximum
dominates every row member; finiteness still requires an unmasked member. -/
import MerlinVerify.Rules.W8A8MaskedMaximum
import MerlinVerify.Rules.W8A8RealOrder

namespace MerlinVerify.Rules.W8A8MaskedOrder

open MerlinVerify W8A8MaskedMaximum W8A8RoundedReal

def Below (left right : FBits .f32) : Prop :=
  left = negativeInfinity ∨
    (W8A8IEEE32.Finite left ∧ W8A8IEEE32.Finite right ∧ value left ≤ value right)

theorem below_refl {cap : Nat} {bits : FBits .f32} (admissible : Admissible cap bits) : Below bits bits := by
  rcases admissible with masked | finite
  · exact Or.inl masked
  · exact Or.inr ⟨finite.1, finite.1, le_rfl⟩

theorem below_trans {left middle right : FBits .f32}
    (first : Below left middle) (next : Below middle right) : Below left right := by
  rcases first with masked | ⟨leftFinite, middleFinite, firstOrder⟩
  · exact Or.inl masked
  · rcases next with masked | ⟨_, rightFinite, nextOrder⟩
    · rw [masked] at middleFinite
      exact (show ¬ W8A8IEEE32.Finite negativeInfinity by decide).elim middleFinite
    · exact Or.inr ⟨leftFinite, rightFinite, le_trans firstOrder nextOrder⟩

theorem maximum_dominates {cap : Nat} {left right : FBits .f32}
    (leftAdmissible : Admissible cap left) (rightAdmissible : Admissible cap right) :
    Below left (W8A8ByteFloat.maximum left right) ∧ Below right (W8A8ByteFloat.maximum left right) := by
  rcases leftAdmissible with rfl | leftFinite
  · rcases rightAdmissible with rfl | rightFinite
    · exact ⟨Or.inl rfl, Or.inl rfl⟩
    · rw [maximum_negative_infinity_left rightFinite.1]
      exact ⟨Or.inl rfl, below_refl (Or.inr rightFinite)⟩
  · rcases rightAdmissible with rfl | rightFinite
    · rw [maximum_negative_infinity_right leftFinite.1]
      exact ⟨below_refl (Or.inr leftFinite), Or.inl rfl⟩
    · have resultFinite := maximum_bounded_left leftFinite (Or.inr rightFinite)
      have ordered := W8A8RealOrder.maximum_dominates leftFinite.1 rightFinite.1
      exact ⟨Or.inr ⟨leftFinite.1, resultFinite.1, ordered.1⟩,
        Or.inr ⟨rightFinite.1, resultFinite.1, ordered.2⟩⟩

theorem fold_dominates {cap : Nat} (words : List (FBits .f32)) :
    ∀ initial, Admissible cap initial → (∀ word ∈ words, Admissible cap word) →
    Below initial (words.foldl W8A8ByteFloat.maximum initial) ∧
      (∀ word ∈ words, Below word (words.foldl W8A8ByteFloat.maximum initial)) := by
  induction words with
  | nil =>
    intro initial admissible _
    exact ⟨below_refl admissible, by intro _ absent; cases absent⟩
  | cons head tail induction =>
    intro initial admissible members
    have headAdmissible := members head (by simp)
    have first := maximum_dominates admissible headAdmissible
    have remaining := induction _ (maximum_admissible admissible headAdmissible)
      (fun word member => members word (by simp [member]))
    refine ⟨below_trans first.1 remaining.1, ?_⟩
    intro word member
    rcases List.mem_cons.mp member with rfl | later
    · exact below_trans first.2 remaining.1
    · exact remaining.2 word later

theorem finite_member_le_maximum {cap : Nat} (words : List (FBits .f32))
    (members : ∀ word ∈ words, Admissible cap word) {word : FBits .f32}
    (member : word ∈ words) (finite : W8A8IEEE32.Finite word) :
    value word ≤ value (words.foldl W8A8ByteFloat.maximum negativeInfinity) := by
  have ordered := (fold_dominates words negativeInfinity (Or.inl rfl) members).2 word member
  rcases ordered with masked | ⟨_, _, bounded⟩
  · rw [masked] at finite
    exact (show ¬ W8A8IEEE32.Finite negativeInfinity by decide).elim finite
  · exact bounded

end MerlinVerify.Rules.W8A8MaskedOrder
