/- Actual IEEE min/max-byte maximum for finite attention scores masked by negative infinity.
A finite row member is required explicitly; an all-masked row is never called finite. -/
import MerlinVerify.Rules.W8A8IEEEBounds

namespace MerlinVerify.Rules.W8A8MaskedMaximum

open MerlinVerify

abbrev negativeInfinity : FBits .f32 := 0xff800000

def Admissible (cap : Nat) (bits : FBits .f32) : Prop :=
  bits = negativeInfinity ∨ W8A8IEEEBounds.ExponentLE cap bits

instance (cap : Nat) (bits : FBits .f32) : Decidable (Admissible cap bits) :=
  inferInstanceAs (Decidable (bits = negativeInfinity ∨ W8A8IEEEBounds.ExponentLE cap bits))

theorem finite_magnitude {bits : FBits .f32} (finite : W8A8IEEE32.Finite bits) :
    W8A8ByteFloat.magnitudeWord bits < 0x7f800000 := by
  change W8A8ByteFloat.magnitudeWord bits / 2^23 < 255 at finite
  omega

theorem maximum_negative_infinity_left {bits : FBits .f32} (finite : W8A8IEEE32.Finite bits) :
    W8A8ByteFloat.maximum negativeInfinity bits = bits := by
  have nonNaN := W8A8IEEE32.finite_notNaN finite
  have magnitude := finite_magnitude finite
  have infNaN : W8A8ByteFloat.isNaN negativeInfinity = false := by decide
  have infSign : W8A8ByteFloat.negative negativeInfinity = true := by decide
  have infWord : W8A8ByteFloat.magnitudeWord negativeInfinity = 0x7f800000 := by decide
  have ordered : W8A8ByteFloat.precedes negativeInfinity bits = true := by
    unfold W8A8ByteFloat.precedes
    rw [infSign, infWord]
    cases W8A8ByteFloat.negative bits <;> simp [magnitude]
  unfold W8A8ByteFloat.maximum
  rw [infNaN, nonNaN, ordered]
  rfl

theorem maximum_negative_infinity_right {bits : FBits .f32} (finite : W8A8IEEE32.Finite bits) :
    W8A8ByteFloat.maximum bits negativeInfinity = bits := by
  have nonNaN := W8A8IEEE32.finite_notNaN finite
  have magnitude := finite_magnitude finite
  have infNaN : W8A8ByteFloat.isNaN negativeInfinity = false := by decide
  have infSign : W8A8ByteFloat.negative negativeInfinity = true := by decide
  have infWord : W8A8ByteFloat.magnitudeWord negativeInfinity = 0x7f800000 := by decide
  have ordered : W8A8ByteFloat.precedes bits negativeInfinity = false := by
    unfold W8A8ByteFloat.precedes
    rw [infSign, infWord]
    cases W8A8ByteFloat.negative bits <;> simp [show ¬ 0x7f800000 < W8A8ByteFloat.magnitudeWord bits by omega]
  unfold W8A8ByteFloat.maximum
  rw [infNaN, nonNaN, ordered]
  rfl

theorem maximum_choice (left right : FBits .f32) :
    W8A8ByteFloat.maximum left right = left ∨ W8A8ByteFloat.maximum left right = right := by
  unfold W8A8ByteFloat.maximum
  split
  · exact Or.inl rfl
  · split
    · exact Or.inr rfl
    · split
      · exact Or.inr rfl
      · exact Or.inl rfl

theorem maximum_admissible {cap : Nat} {left right : FBits .f32}
    (leftAdmissible : Admissible cap left) (rightAdmissible : Admissible cap right) :
    Admissible cap (W8A8ByteFloat.maximum left right) := by
  rcases leftAdmissible with rfl | leftFinite
  · rcases rightAdmissible with rfl | rightFinite
    · exact Or.inl (by decide)
    · rw [maximum_negative_infinity_left rightFinite.1]
      exact Or.inr rightFinite
  · rcases rightAdmissible with rfl | rightFinite
    · rw [maximum_negative_infinity_right leftFinite.1]
      exact Or.inr leftFinite
    · rcases maximum_choice left right with selected | selected
      · rw [selected]; exact Or.inr leftFinite
      · rw [selected]; exact Or.inr rightFinite

theorem maximum_bounded_left {cap : Nat} {left right : FBits .f32}
    (leftFinite : W8A8IEEEBounds.ExponentLE cap left) (rightAdmissible : Admissible cap right) :
    W8A8IEEEBounds.ExponentLE cap (W8A8ByteFloat.maximum left right) := by
  rcases rightAdmissible with rfl | rightFinite
  · rw [maximum_negative_infinity_right leftFinite.1]
    exact leftFinite
  · rcases maximum_choice left right with selected | selected
    · rw [selected]; exact leftFinite
    · rw [selected]; exact rightFinite

theorem maximum_bounded_right {cap : Nat} {left right : FBits .f32}
    (leftAdmissible : Admissible cap left) (rightFinite : W8A8IEEEBounds.ExponentLE cap right) :
    W8A8IEEEBounds.ExponentLE cap (W8A8ByteFloat.maximum left right) := by
  rcases leftAdmissible with rfl | leftFinite
  · rw [maximum_negative_infinity_left rightFinite.1]
    exact rightFinite
  · rcases maximum_choice left right with selected | selected
    · rw [selected]; exact leftFinite
    · rw [selected]; exact rightFinite

theorem fold_admissible {cap : Nat} (words : List (FBits .f32)) :
    ∀ initial, Admissible cap initial → (∀ word ∈ words, Admissible cap word) →
    Admissible cap (words.foldl W8A8ByteFloat.maximum initial) := by
  induction words with
  | nil => intro initial admissible _; exact admissible
  | cons head tail induction =>
    intro initial admissible members
    apply induction _ (maximum_admissible admissible (members head (by simp)))
    intro word member
    exact members word (by simp [member])

theorem fold_bounded_initial {cap : Nat} (words : List (FBits .f32)) :
    ∀ initial, W8A8IEEEBounds.ExponentLE cap initial → (∀ word ∈ words, Admissible cap word) →
    W8A8IEEEBounds.ExponentLE cap (words.foldl W8A8ByteFloat.maximum initial) := by
  induction words with
  | nil => intro initial bounded _; exact bounded
  | cons head tail induction =>
    intro initial bounded members
    apply induction _ (maximum_bounded_left bounded (members head (by simp)))
    intro word member
    exact members word (by simp [member])

/-- One actual finite member makes the ordered row reduction finite, even though its
initial value and other members may be genuine negative infinities. -/
theorem fold_bounded_member {cap : Nat} (words : List (FBits .f32)) :
    ∀ initial, Admissible cap initial → (∀ word ∈ words, Admissible cap word) →
    (∃ word ∈ words, W8A8IEEEBounds.ExponentLE cap word) →
    W8A8IEEEBounds.ExponentLE cap (words.foldl W8A8ByteFloat.maximum initial) := by
  induction words with
  | nil => intro _ _ _ witness; simp at witness
  | cons head tail induction =>
    intro initial admissible members witness
    by_cases headFinite : W8A8IEEEBounds.ExponentLE cap head
    · apply fold_bounded_initial tail _ (maximum_bounded_right admissible headFinite)
      intro word member
      exact members word (by simp [member])
    · apply induction _ (maximum_admissible admissible (members head (by simp)))
      · intro word member
        exact members word (by simp [member])
      · obtain ⟨word, member, bounded⟩ := witness
        exact ⟨word, (List.mem_cons.mp member).resolve_left
          (by intro equal; subst word; exact headFinite bounded), bounded⟩

theorem fold_choice (words : List (FBits .f32)) (initial : FBits .f32) :
    words.foldl W8A8ByteFloat.maximum initial = initial ∨
      ∃ word ∈ words, words.foldl W8A8ByteFloat.maximum initial = word := by
  induction words generalizing initial with
  | nil => exact Or.inl rfl
  | cons head tail induction =>
    rcases induction (W8A8ByteFloat.maximum initial head) with same | ⟨word, member, same⟩
    · rcases maximum_choice initial head with left | right
      · exact Or.inl (same.trans left)
      · exact Or.inr ⟨head, by simp, same.trans right⟩
    · exact Or.inr ⟨word, by simp [member], same⟩

/-- The maximum is an actual finite row member, not just an upper bound supplied as a hint. -/
theorem row_maximum_attained {cap : Nat} (words : List (FBits .f32))
    (members : ∀ word ∈ words, Admissible cap word)
    (unmasked : ∃ word ∈ words, W8A8IEEEBounds.ExponentLE cap word) :
    ∃ word ∈ words, words.foldl W8A8ByteFloat.maximum negativeInfinity = word ∧
      W8A8IEEEBounds.ExponentLE cap word := by
  have bounded := fold_bounded_member words negativeInfinity (Or.inl rfl) members unmasked
  rcases fold_choice words negativeInfinity with initial | ⟨word, member, same⟩
  · rw [initial] at bounded
    have impossible : ¬ W8A8IEEE32.Finite negativeInfinity := by decide
    exact (impossible bounded.1).elim
  · exact ⟨word, member, same, same ▸ bounded⟩

theorem all_masked (count : Nat) :
    (List.replicate count negativeInfinity).foldl W8A8ByteFloat.maximum negativeInfinity = negativeInfinity := by
  induction count with
  | zero => rfl
  | succ count induction =>
    simp only [List.replicate_succ, List.foldl_cons,
      show W8A8ByteFloat.maximum negativeInfinity negativeInfinity = negativeInfinity by decide]
    exact induction

theorem all_masked_not_finite (count : Nat) :
    ¬ W8A8IEEE32.Finite
      ((List.replicate count negativeInfinity).foldl W8A8ByteFloat.maximum negativeInfinity) := by
  rw [all_masked]
  decide

end MerlinVerify.Rules.W8A8MaskedMaximum
