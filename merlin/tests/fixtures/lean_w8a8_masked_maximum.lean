/- Kernel controls for genuine negative-infinity masks and an attained finite row maximum.
These component theorems are not an admission proof for the captured attention mask. -/
import MerlinVerify.Rules.W8A8AttentionDifference

open MerlinVerify Rules W8A8MaskedMaximum

namespace W8A8MaskedMaximumControls

theorem exceptional_domain_controls :
    Admissible 179 negativeInfinity ∧
    Admissible 179 (0x3f800000 : FBits .f32) ∧
    ¬ Admissible 179 (0x7f800000 : FBits .f32) ∧
    ¬ Admissible 179 (0x7fc00000 : FBits .f32) := by
  unfold Admissible
  decide

theorem signed_zero_and_infinity_maximum :
    W8A8ByteFloat.maximum negativeInfinity (0 : FBits .f32) = 0 ∧
    W8A8ByteFloat.maximum (0x80000000 : FBits .f32) negativeInfinity = 0x80000000 ∧
    W8A8ByteFloat.maximum (0x80000000 : FBits .f32) (0 : FBits .f32) = 0 := by decide

theorem arbitrary_nonempty_masked_row {cap : Nat} (words : List (FBits .f32))
    (members : ∀ word ∈ words, Admissible cap word)
    (unmasked : ∃ word ∈ words, W8A8IEEEBounds.ExponentLE cap word) :
    ∃ word ∈ words, words.foldl W8A8ByteFloat.maximum negativeInfinity = word ∧
      W8A8RoundedReal.exponential (W8A8ArithmeticModel.subtract32 word
        (words.foldl W8A8ByteFloat.maximum negativeInfinity)) = (0x3f800000 : FBits .f32) :=
  W8A8AttentionDifference.row_exponential_one words members unmasked

theorem inhabited_row : ∃ words : List (FBits .f32),
    (∀ word ∈ words, Admissible 179 word) ∧
    (∃ word ∈ words, W8A8IEEEBounds.ExponentLE 179 word) := by
  refine ⟨[0], ?_, ?_⟩
  · intro word member
    simp only [List.mem_singleton] at member
    subst word
    exact Or.inr (by decide)
  · exact ⟨0, by simp, by decide⟩

theorem all_masked_is_not_admitted (count : Nat) :
    ¬ W8A8IEEE32.Finite
      ((List.replicate count negativeInfinity).foldl W8A8ByteFloat.maximum negativeInfinity) ∧
    W8A8RoundedReal.exponential (W8A8ArithmeticModel.subtract32 negativeInfinity
      ((List.replicate count negativeInfinity).foldl W8A8ByteFloat.maximum negativeInfinity)) =
      (0x7fc00000 : FBits .f32) :=
  ⟨all_masked_not_finite count, W8A8AttentionDifference.all_masked_exponential_nan count⟩

theorem finite_member_has_exact_one (bits : FBits .f32) (finite : W8A8IEEE32.Finite bits) :
    W8A8RoundedReal.exponential (W8A8ArithmeticModel.subtract32 bits bits) = (0x3f800000 : FBits .f32) :=
  W8A8AttentionDifference.exponential_self_difference finite

theorem actual_masked_member_has_exact_zero (maximum : FBits .f32) (finite : W8A8IEEE32.Finite maximum) :
    W8A8RoundedReal.exponential (W8A8ArithmeticModel.subtract32 negativeInfinity maximum) = (0 : FBits .f32) :=
  W8A8AttentionDifference.masked_exponential_zero finite

#print axioms exceptional_domain_controls
#print axioms signed_zero_and_infinity_maximum
#print axioms arbitrary_nonempty_masked_row
#print axioms inhabited_row
#print axioms all_masked_is_not_admitted
#print axioms finite_member_has_exact_one
#print axioms actual_masked_member_has_exact_zero

end W8A8MaskedMaximumControls
