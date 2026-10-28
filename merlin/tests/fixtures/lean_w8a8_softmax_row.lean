/- Numerical-row nonvacuity and exceptional controls. These prove the named mathematical
row contract; actual captured SSA/mask binding and native fidelity remain unqualified. -/
import MerlinVerify.Rules.W8A8SoftmaxRow

open MerlinVerify Rules W8A8MaskedMaximum

namespace W8A8SoftmaxRowControls

theorem singleton_exponentials : W8A8SoftmaxRow.exponentials [0] = [0x3f800000] := by
  simp only [W8A8SoftmaxRow.exponentials, W8A8SoftmaxRow.rowMaximum, List.foldl_cons,
    List.foldl_nil, maximum_negative_infinity_left (show W8A8IEEE32.Finite (0 : FBits .f32) by decide),
    List.map_cons, List.map_nil, W8A8AttentionDifference.subtract_self (bits := (0 : FBits .f32)) (by decide),
    W8A8RoundedReal.exponential_zero]

theorem singleton_denominator : W8A8SoftmaxRow.denominator [0] = 0x3f800000 := by
  rw [W8A8SoftmaxRow.denominator, singleton_exponentials]
  decide

theorem masked_exponentials :
    W8A8SoftmaxRow.exponentials [negativeInfinity, 0] = [0, 0x3f800000] := by
  have repeated : W8A8ByteFloat.maximum negativeInfinity negativeInfinity = negativeInfinity := by decide
  simp only [W8A8SoftmaxRow.exponentials, W8A8SoftmaxRow.rowMaximum, List.foldl_cons,
    List.foldl_nil, repeated,
    maximum_negative_infinity_left (show W8A8IEEE32.Finite (0 : FBits .f32) by decide),
    List.map_cons, List.map_nil,
    W8A8AttentionDifference.negative_infinity_subtract_finite (bits := (0 : FBits .f32)) (by decide),
    W8A8AttentionDifference.subtract_self (bits := (0 : FBits .f32)) (by decide),
    W8A8RoundedReal.exponential_negative_infinity, W8A8RoundedReal.exponential_zero]

theorem masked_denominator : W8A8SoftmaxRow.denominator [negativeInfinity, 0] = 0x3f800000 := by
  rw [W8A8SoftmaxRow.denominator, masked_exponentials]
  decide

theorem arbitrary_counted_row {cap : Nat} (words : List (FBits .f32))
    (safe : cap ≤ 252) (members : ∀ word ∈ words, Admissible cap word)
    (unmasked : ∃ word ∈ words, W8A8IEEEBounds.ExponentLE cap word) (count : words.length ≤ 2^22) :
    W8A8IEEEBounds.ExponentLE (W8A8OrderedMagnitude.sumCap 149 words.length) (W8A8SoftmaxRow.denominator words) ∧
      W8A8IEEE32.Positive (W8A8SoftmaxRow.denominator words) :=
  W8A8SoftmaxRow.denominator_admissible words safe members unmasked count

theorem subnormal_and_overflow_controls :
    W8A8IEEE32.add (1 : FBits .f32) 0 = 1 ∧
    W8A8IEEE32.add (1 : FBits .f32) 1 = 2 ∧
    W8A8IEEE32.Positive (W8A8IEEE32.add (0x7f800000 : FBits .f32) 0x3f800000) ∧
    ¬ W8A8IEEE32.Finite (W8A8ArithmeticModel.subtract32 (0x7f7fffff : FBits .f32) 0xff7fffff) ∧
    W8A8OrderedMagnitude.sumCap 149 8 = 131 := by decide

theorem singleton_probabilities : W8A8SoftmaxRow.probabilities [0] = [0x3f800000] := by
  rw [W8A8SoftmaxRow.probabilities, singleton_exponentials, singleton_denominator]
  decide

theorem masked_probabilities :
    W8A8SoftmaxRow.probabilities [negativeInfinity, 0] = [0, 0x3f800000] := by
  rw [W8A8SoftmaxRow.probabilities, masked_exponentials, masked_denominator]
  decide

theorem arbitrary_denominator_lower {cap : Nat} (words : List (FBits .f32))
    (safe : cap ≤ 252) (members : ∀ word ∈ words, Admissible cap word)
    (unmasked : ∃ word ∈ words, W8A8IEEEBounds.ExponentLE cap word) (count : words.length ≤ 2^22) :
    W8A8IEEEBounds.ExponentLE (W8A8OrderedMagnitude.sumCap 149 words.length) (W8A8SoftmaxRow.denominator words) ∧
      1 ≤ W8A8RoundedReal.value (W8A8SoftmaxRow.denominator words) :=
  W8A8SoftmaxRow.denominator_finite_lower words safe members unmasked count

theorem arbitrary_probability_bound {cap : Nat} (words : List (FBits .f32))
    (safe : cap ≤ 252) (members : ∀ word ∈ words, Admissible cap word)
    (unmasked : ∃ word ∈ words, W8A8IEEEBounds.ExponentLE cap word) (count : words.length ≤ 2^22) :
    ∀ result ∈ W8A8SoftmaxRow.probabilities words,
      W8A8IEEEBounds.ExponentLE 128 result ∧ W8A8IEEE32.Nonnegative result :=
  W8A8SoftmaxRow.probabilities_bounded words safe members unmasked count

theorem eight_term_attention {cap : Nat} (words values : List (FBits .f32))
    (safe : cap ≤ 252) (members : ∀ word ∈ words, Admissible cap word)
    (unmasked : ∃ word ∈ words, W8A8IEEEBounds.ExponentLE cap word)
    (count : words.length = 8) (geometry : values.length = words.length)
    (valueBounds : ∀ entry ∈ values, W8A8IEEEBounds.ExponentLE 138 entry) :
    W8A8IEEEBounds.ExponentLE 146 (W8A8SoftmaxRow.weightedValue words values) := by
  have bounded := W8A8SoftmaxRow.weighted_value_bounded words values safe members unmasked
    (show words.length ≤ 2^22 by omega) geometry valueBounds (by decide)
  have outputCap : W8A8OrderedDot.cap 128 138 8 = 146 := by decide
  simpa only [count, outputCap] using bounded

/-- Nonzero alone admits a divisor that overflows the quotient. Signed zero,
subnormal numerators and negative unit divisors retain their actual bit behavior. -/
theorem unit_division_controls :
    W8A8IEEE32.div (0x3f800000 : FBits .f32) 0x3f800000 = 0x3f800000 ∧
    W8A8IEEE32.div (1 : FBits .f32) 0x3f800000 = 1 ∧
    W8A8IEEE32.div (0x80000000 : FBits .f32) 0x3f800000 = 0x80000000 ∧
    W8A8IEEE32.div (0xbf800000 : FBits .f32) 0xbf800000 = 0x3f800000 ∧
    W8A8IEEE32.Nonzero (1 : FBits .f32) ∧
    ¬ W8A8UnitAdd.AtLeastOne (1 : FBits .f32) ∧
    W8A8IEEE32.div (0x3f800000 : FBits .f32) 1 = 0x7f800000 ∧
    ¬ W8A8IEEE32.Finite (W8A8IEEE32.div (0x3f800000 : FBits .f32) 1) := by decide

theorem all_masked_probabilities_nan :
    W8A8SoftmaxRow.probabilities [negativeInfinity] = [0x7fc00000] := by
  have maximum : W8A8ByteFloat.maximum negativeInfinity negativeInfinity = negativeInfinity := by decide
  have scalar := W8A8RoundedReal.exponential_nan W8A8AttentionDifference.negative_infinity_self_nan
  have exponentials : W8A8SoftmaxRow.exponentials [negativeInfinity] = [0x7fc00000] := by
    simp only [W8A8SoftmaxRow.exponentials, W8A8SoftmaxRow.rowMaximum, List.foldl_cons,
      List.foldl_nil, List.map_cons, List.map_nil, maximum, scalar]
  have denominator : W8A8SoftmaxRow.denominator [negativeInfinity] = 0x7fc00000 := by
    rw [W8A8SoftmaxRow.denominator, exponentials]
    decide
  rw [W8A8SoftmaxRow.probabilities, exponentials, denominator]
  decide

#print axioms singleton_exponentials
#print axioms singleton_denominator
#print axioms masked_exponentials
#print axioms masked_denominator
#print axioms arbitrary_counted_row
#print axioms subnormal_and_overflow_controls
#print axioms singleton_probabilities
#print axioms masked_probabilities
#print axioms arbitrary_denominator_lower
#print axioms arbitrary_probability_bound
#print axioms eight_term_attention
#print axioms unit_division_controls
#print axioms all_masked_probabilities_nan

end W8A8SoftmaxRowControls
