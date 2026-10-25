/- Concrete masked max/sub/exp and ordered denominator properties for a numerical row.
The actual unmasked-member witness is essential. Source SSA/layout binding remains a
separate obligation; these numerical row/weighted-trace results are not a whole checker. -/
import MerlinVerify.Rules.W8A8MaskedOrder
import MerlinVerify.Rules.W8A8AttentionDifference
import MerlinVerify.Rules.W8A8SubtractionOrder
import MerlinVerify.Rules.W8A8SubtractionBounds
import MerlinVerify.Rules.W8A8PositiveAdd
import MerlinVerify.Rules.W8A8UnitAdd
import MerlinVerify.Rules.W8A8DivisionBounds
import MerlinVerify.Rules.W8A8OrderedDot

namespace MerlinVerify.Rules.W8A8SoftmaxRow

open MerlinVerify W8A8MaskedMaximum W8A8RoundedReal W8A8OrderedMagnitude

def rowMaximum (words : List (FBits .f32)) : FBits .f32 :=
  words.foldl W8A8ByteFloat.maximum negativeInfinity

noncomputable def exponentials (words : List (FBits .f32)) : List (FBits .f32) :=
  words.map fun word => exponential (W8A8ArithmeticModel.subtract32 word (rowMaximum words))

noncomputable def denominator (words : List (FBits .f32)) : FBits .f32 :=
  (exponentials words).foldl W8A8IEEE32.add 0#32

noncomputable def probabilities (words : List (FBits .f32)) : List (FBits .f32) :=
  (exponentials words).map fun term => W8A8IEEE32.div term (denominator words)

noncomputable def weightedValue (words values : List (FBits .f32)) : FBits .f32 :=
  W8A8OrderedDot.evaluate ((probabilities words).zip values) 0#32

theorem exponential_unit {bits : FBits .f32} (finite : W8A8IEEE32.Finite bits) (nonpositive : value bits ≤ 0) :
    0 ≤ value (exponential bits) ∧ value (exponential bits) ≤ 1 := by
  have positive := Real.exp_pos (value bits)
  have upper := Real.exp_le_one_iff.mpr nonpositive
  rw [exponential, ite_eq_left finite, roundedReal_unit (by linarith) upper]
  exact ⟨minimizer_nonnegative (signedFiniteChoice_spec _).2 (le_of_lt positive),
    (minimizer_within_unit (signedFiniteChoice_spec _).2 (by linarith) upper).2⟩

theorem member_exponential {cap : Nat} (words : List (FBits .f32))
    (safe : cap ≤ 252) (members : ∀ word ∈ words, Admissible cap word)
    (unmasked : ∃ word ∈ words, W8A8IEEEBounds.ExponentLE cap word)
    {word : FBits .f32} (member : word ∈ words) :
    W8A8IEEEBounds.ExponentLE 127 (exponential (W8A8ArithmeticModel.subtract32 word (rowMaximum words))) ∧
      W8A8IEEE32.Nonnegative (exponential (W8A8ArithmeticModel.subtract32 word (rowMaximum words))) ∧
      (0 ≤ value (exponential (W8A8ArithmeticModel.subtract32 word (rowMaximum words))) ∧
        value (exponential (W8A8ArithmeticModel.subtract32 word (rowMaximum words))) ≤ 1) := by
  have maximum : W8A8IEEEBounds.ExponentLE cap (rowMaximum words) :=
    fold_bounded_member words negativeInfinity (Or.inl rfl) members unmasked
  rcases members word member with masked | bounded
  · rw [masked, W8A8AttentionDifference.masked_exponential_zero maximum.1]
    refine ⟨by decide, by decide, ?_, ?_⟩
    all_goals rw [value_positive_zero]
    all_goals norm_num
  · have difference := W8A8SubtractionBounds.subtract_bounded bounded maximum (by simpa using safe)
    have ordered := W8A8MaskedOrder.finite_member_le_maximum words members member bounded.1
    have nonpositive := W8A8SubtractionOrder.finite_difference_nonpositive bounded.1 maximum.1 ordered
    exact ⟨exponential_cap difference.1 nonpositive, exponential_nonnegative difference.1 nonpositive,
      exponential_unit difference.1 nonpositive⟩

theorem row_properties {cap : Nat} (words : List (FBits .f32))
    (safe : cap ≤ 252) (members : ∀ word ∈ words, Admissible cap word)
    (unmasked : ∃ word ∈ words, W8A8IEEEBounds.ExponentLE cap word) :
    (∀ result ∈ exponentials words, W8A8IEEEBounds.ExponentLE 127 result ∧ W8A8IEEE32.Nonnegative result ∧
      (0 ≤ value result ∧ value result ≤ 1)) ∧
      (0x3f800000 : FBits .f32) ∈ exponentials words := by
  constructor
  · intro result member
    obtain ⟨word, inputMember, rfl⟩ := List.mem_map.mp member
    exact member_exponential words safe members unmasked inputMember
  · obtain ⟨word, member, _, exactOne⟩ := W8A8AttentionDifference.row_exponential_one words members unmasked
    exact List.mem_map.mpr ⟨word, member, exactOne⟩

theorem units_of_unit_range {bits : FBits .f32} (nonnegative : 0 ≤ value bits) (upper : value bits ≤ 1) :
    units bits ≤ 2^149 := by
  have absolute : |value bits| ≤ 1 := by rw [abs_of_nonneg nonnegative]; exact upper
  rw [abs_value] at absolute
  have realBound : (units bits : ℝ) ≤ (2:ℝ)^149 := (div_le_one (by positivity)).mp absolute
  exact_mod_cast realBound

theorem fold_nonnegative (words : List (FBits .f32))
    (members : ∀ word ∈ words, W8A8IEEE32.Nonnegative word) :
    ∀ initial, W8A8IEEE32.Nonnegative initial → W8A8IEEE32.Nonnegative (words.foldl W8A8IEEE32.add initial) := by
  induction words with
  | nil => intro initial nonnegative; exact nonnegative
  | cons head tail induction =>
    intro initial nonnegative
    exact induction (fun word member => members word (by simp [member])) _
      (W8A8IEEE32.add_nonnegative nonnegative (members head (by simp)))

/-- Counted finite/nonnegative ordered summation. The separate positivity theorem uses
the actual exp(0)=1 member; positivity does not follow from exponent caps alone. -/
theorem denominator_bounded {cap : Nat} (words : List (FBits .f32))
    (safe : cap ≤ 252) (members : ∀ word ∈ words, Admissible cap word)
    (unmasked : ∃ word ∈ words, W8A8IEEEBounds.ExponentLE cap word) (count : words.length ≤ 2^22) :
    W8A8IEEEBounds.ExponentLE (sumCap 149 words.length) (denominator words) ∧
      W8A8IEEE32.Nonnegative (denominator words) := by
  have properties := (row_properties words safe members unmasked).1
  have terms : ∀ result ∈ exponentials words, W8A8IEEE32.Finite result ∧ units result ≤ 2^149 := by
    intro result member
    have property := properties result member
    exact ⟨property.1.1, units_of_unit_range property.2.2.1 property.2.2.2⟩
  have bounded := sum_exponent_bounded (limit := 149) (by decide) (exponentials words) terms
    (by simpa [exponentials] using count)
  refine ⟨?_, fold_nonnegative _ (fun result member => (properties result member).2.1) _ (by decide)⟩
  simpa only [denominator, exponentials, List.length_map] using bounded

theorem denominator_positive {cap : Nat} (words : List (FBits .f32))
    (safe : cap ≤ 252) (members : ∀ word ∈ words, Admissible cap word)
    (unmasked : ∃ word ∈ words, W8A8IEEEBounds.ExponentLE cap word) :
    W8A8IEEE32.Positive (denominator words) := by
  have properties := row_properties words safe members unmasked
  exact W8A8PositiveAdd.fold_positive_member (exponentials words) 0#32 (by decide)
    (fun result member => (properties.1 result member).2.1)
    ⟨0x3f800000, properties.2, by decide⟩

/-- The actual unit term gives a lower exponent threshold through the ordered
rounded additions. Finiteness is supplied separately by `denominator_bounded`. -/
theorem denominator_unit {cap : Nat} (words : List (FBits .f32))
    (safe : cap ≤ 252) (members : ∀ word ∈ words, Admissible cap word)
    (unmasked : ∃ word ∈ words, W8A8IEEEBounds.ExponentLE cap word) :
    W8A8UnitAdd.AtLeastOne (denominator words) := by
  have properties := row_properties words safe members unmasked
  exact W8A8UnitAdd.fold_unit_member (exponentials words) 0#32 (by decide)
    (fun result member => (properties.1 result member).2.1)
    ⟨0x3f800000, properties.2, by decide⟩

theorem denominator_admissible {cap : Nat} (words : List (FBits .f32))
    (safe : cap ≤ 252) (members : ∀ word ∈ words, Admissible cap word)
    (unmasked : ∃ word ∈ words, W8A8IEEEBounds.ExponentLE cap word) (count : words.length ≤ 2^22) :
    W8A8IEEEBounds.ExponentLE (sumCap 149 words.length) (denominator words) ∧
      W8A8IEEE32.Positive (denominator words) :=
  ⟨(denominator_bounded words safe members unmasked count).1,
    denominator_positive words safe members unmasked⟩

/-- Both the finite encoding and its real lower bound are established. The count
premise is essential to the finite ordered sum, not to the unit-member argument. -/
theorem denominator_finite_lower {cap : Nat} (words : List (FBits .f32))
    (safe : cap ≤ 252) (members : ∀ word ∈ words, Admissible cap word)
    (unmasked : ∃ word ∈ words, W8A8IEEEBounds.ExponentLE cap word) (count : words.length ≤ 2^22) :
    W8A8IEEEBounds.ExponentLE (sumCap 149 words.length) (denominator words) ∧
      1 ≤ value (denominator words) := by
  refine ⟨(denominator_bounded words safe members unmasked count).1, ?_⟩
  have unit := denominator_unit words safe members unmasked
  have lower := W8A8UnitAdd.units_lower unit.2
  have realLower : (2 : ℝ)^149 ≤ (units (denominator words) : ℝ) := by exact_mod_cast lower
  simp only [value, unit.1.2, Bool.false_eq_true, ↓reduceIte, magnitude]
  exact (le_div_iff₀ (by positivity)).mpr (by simpa using realLower)

/-- The cap is conservative: it proves finite, nonnegative actual division,
including underflow. Exact real probabilities and native fidelity are not assumed. -/
theorem probabilities_bounded {cap : Nat} (words : List (FBits .f32))
    (safe : cap ≤ 252) (members : ∀ word ∈ words, Admissible cap word)
    (unmasked : ∃ word ∈ words, W8A8IEEEBounds.ExponentLE cap word) (count : words.length ≤ 2^22) :
    ∀ result ∈ probabilities words,
      W8A8IEEEBounds.ExponentLE 128 result ∧ W8A8IEEE32.Nonnegative result := by
  have properties := (row_properties words safe members unmasked).1
  have finite := (denominator_bounded words safe members unmasked count).1.1
  have lower := (denominator_unit words safe members unmasked).2
  have positive := denominator_positive words safe members unmasked
  intro result member
  obtain ⟨term, found, rfl⟩ := List.mem_map.mp member
  have property := properties term found
  exact ⟨W8A8DivisionBounds.divide_by_unit_bounded property.1 finite lower (by decide),
    W8A8IEEE32.div_nonnegative property.2.1 positive finite⟩

/-- One ordered output lane, with every V entry paired to its probability. The
equal-length premise rules out silently truncating the contraction with `zip`. -/
theorem weighted_value_bounded {cap valueCap : Nat} (words values : List (FBits .f32))
    (safe : cap ≤ 252) (members : ∀ word ∈ words, Admissible cap word)
    (unmasked : ∃ word ∈ words, W8A8IEEEBounds.ExponentLE cap word) (count : words.length ≤ 2^22)
    (geometry : values.length = words.length)
    (valueBounds : ∀ entry ∈ values, W8A8IEEEBounds.ExponentLE valueCap entry)
    (productsSafe : W8A8OrderedDot.productLimit 128 valueCap ≤ 252) :
    W8A8IEEEBounds.ExponentLE (W8A8OrderedDot.cap 128 valueCap words.length) (weightedValue words values) := by
  have probabilitiesLength : (probabilities words).length = words.length := by
    simp only [probabilities, exponentials, List.length_map]
  have pairsLength : ((probabilities words).zip values).length = words.length := by
    simp only [List.length_zip, probabilitiesLength, geometry, Nat.min_self]
  have probabilityBounds := probabilities_bounded words safe members unmasked count
  have inputs : ∀ pair ∈ (probabilities words).zip values,
      W8A8IEEEBounds.ExponentLE 128 pair.1 ∧ W8A8IEEEBounds.ExponentLE valueCap pair.2 := by
    intro pair member
    have projections := List.of_mem_zip member
    exact ⟨(probabilityBounds pair.1 projections.1).1, valueBounds pair.2 projections.2⟩
  have admissible : W8A8OrderedDot.admissible 128 valueCap ((probabilities words).zip values).length = true := by
    apply decide_eq_true
    rw [pairsLength]
    exact ⟨productsSafe, count⟩
  simpa only [weightedValue, pairsLength] using W8A8OrderedDot.evaluate_bounded inputs admissible

end MerlinVerify.Rules.W8A8SoftmaxRow
