/- Connect the byte-level IEEE ordering used by actual maximum to finite real values.
No comparison law is assumed about an arbitrary FloatModel. -/
import MerlinVerify.Rules.W8A8RealValue

namespace MerlinVerify.Rules.W8A8RealOrder

open MerlinVerify W8A8OrderedMagnitude W8A8RoundedReal

theorem units_leading {bits : FBits .f32} (nonzero : W8A8ByteFloat.exponent bits ≠ 0) :
    2^(W8A8ByteFloat.exponent bits + 22) ≤ units bits := by
  have positive : 0 < W8A8ByteFloat.exponent bits := Nat.pos_of_ne_zero nonzero
  unfold units
  rw [ite_eq_right nonzero]
  have powers : W8A8ByteFloat.exponent bits + 22 = 23 + (W8A8ByteFloat.exponent bits - 1) := by omega
  rw [powers, Nat.pow_add]
  exact Nat.mul_le_mul_right _ (Nat.le_add_right _ _)

theorem units_monotone {left right : FBits .f32} (finite : W8A8IEEE32.Finite left)
    (ordered : W8A8ByteFloat.magnitudeWord left ≤ W8A8ByteFloat.magnitudeWord right) :
    units left ≤ units right := by
  have exponent : W8A8ByteFloat.exponent left ≤ W8A8ByteFloat.exponent right :=
    Nat.div_le_div_right ordered
  by_cases same : W8A8ByteFloat.exponent left = W8A8ByteFloat.exponent right
  · have fraction : W8A8ByteFloat.fraction left ≤ W8A8ByteFloat.fraction right := by
      unfold W8A8ByteFloat.exponent at same
      unfold W8A8ByteFloat.fraction
      omega
    simp only [units, same]
    split
    · exact fraction
    · exact Nat.mul_le_mul_right _ (Nat.add_le_add_left fraction _)
  · have strict : W8A8ByteFloat.exponent left < W8A8ByteFloat.exponent right := by omega
    have leftUnits := units_lt_of_exponent
      (show W8A8IEEEBounds.ExponentLE (W8A8ByteFloat.exponent left) left from ⟨finite, le_rfl⟩)
    have rightLeading := units_leading (bits := right) (by omega)
    have powers : 2^(W8A8ByteFloat.exponent left + 23) ≤ 2^(W8A8ByteFloat.exponent right + 22) :=
      Nat.pow_le_pow_right (by decide) (by omega)
    exact Nat.le_of_lt (Nat.lt_of_lt_of_le leftUnits (Nat.le_trans powers rightLeading))

theorem magnitude_monotone {left right : FBits .f32} (finite : W8A8IEEE32.Finite left)
    (ordered : W8A8ByteFloat.magnitudeWord left ≤ W8A8ByteFloat.magnitudeWord right) :
    magnitude left ≤ magnitude right := by
  unfold magnitude
  apply div_le_div_of_nonneg_right
  · exact_mod_cast units_monotone finite ordered
  · positivity

theorem precedes_sound {left right : FBits .f32}
    (leftFinite : W8A8IEEE32.Finite left) (rightFinite : W8A8IEEE32.Finite right)
    (ordered : W8A8ByteFloat.precedes left right = true) : value left ≤ value right := by
  cases leftSign : W8A8ByteFloat.negative left with
  | false =>
    cases rightSign : W8A8ByteFloat.negative right with
    | false =>
      have words : W8A8ByteFloat.magnitudeWord left < W8A8ByteFloat.magnitudeWord right := by
        simpa [W8A8ByteFloat.precedes, leftSign, rightSign] using ordered
      have magnitude := magnitude_monotone leftFinite (Nat.le_of_lt words)
      simpa only [value, leftSign, rightSign, Bool.false_eq_true, ↓reduceIte] using magnitude
    | true => simp [W8A8ByteFloat.precedes, leftSign, rightSign] at ordered
  | true =>
    cases rightSign : W8A8ByteFloat.negative right with
    | false =>
      have leftBound := magnitude_nonneg left
      have rightBound := magnitude_nonneg right
      simp only [value, leftSign, rightSign, Bool.false_eq_true, ↓reduceIte]
      linarith
    | true =>
      have words : W8A8ByteFloat.magnitudeWord right < W8A8ByteFloat.magnitudeWord left := by
        simpa [W8A8ByteFloat.precedes, leftSign, rightSign] using ordered
      have magnitude := magnitude_monotone rightFinite (Nat.le_of_lt words)
      simpa only [value, leftSign, rightSign, ↓reduceIte] using neg_le_neg magnitude

theorem not_precedes_sound {left right : FBits .f32}
    (leftFinite : W8A8IEEE32.Finite left) (rightFinite : W8A8IEEE32.Finite right)
    (ordered : W8A8ByteFloat.precedes left right = false) : value right ≤ value left := by
  cases leftSign : W8A8ByteFloat.negative left with
  | false =>
    cases rightSign : W8A8ByteFloat.negative right with
    | false =>
      have words : W8A8ByteFloat.magnitudeWord right ≤ W8A8ByteFloat.magnitudeWord left := by
        have notLess : ¬ W8A8ByteFloat.magnitudeWord left < W8A8ByteFloat.magnitudeWord right := by
          simpa [W8A8ByteFloat.precedes, leftSign, rightSign] using ordered
        omega
      have magnitude := magnitude_monotone rightFinite words
      simpa only [value, leftSign, rightSign, Bool.false_eq_true, ↓reduceIte] using magnitude
    | true =>
      have leftBound := magnitude_nonneg left
      have rightBound := magnitude_nonneg right
      simp only [value, leftSign, rightSign, Bool.false_eq_true, ↓reduceIte]
      linarith
  | true =>
    cases rightSign : W8A8ByteFloat.negative right with
    | false => simp [W8A8ByteFloat.precedes, leftSign, rightSign] at ordered
    | true =>
      have words : W8A8ByteFloat.magnitudeWord left ≤ W8A8ByteFloat.magnitudeWord right := by
        have notLess : ¬ W8A8ByteFloat.magnitudeWord right < W8A8ByteFloat.magnitudeWord left := by
          simpa [W8A8ByteFloat.precedes, leftSign, rightSign] using ordered
        omega
      have magnitude := magnitude_monotone leftFinite words
      simpa only [value, leftSign, rightSign, ↓reduceIte] using neg_le_neg magnitude

theorem maximum_dominates {left right : FBits .f32}
    (leftFinite : W8A8IEEE32.Finite left) (rightFinite : W8A8IEEE32.Finite right) :
    value left ≤ value (W8A8ByteFloat.maximum left right) ∧
      value right ≤ value (W8A8ByteFloat.maximum left right) := by
  simp only [W8A8ByteFloat.maximum, W8A8IEEE32.finite_notNaN leftFinite,
    W8A8IEEE32.finite_notNaN rightFinite, Bool.false_eq_true, ↓reduceIte]
  cases ordered : W8A8ByteFloat.precedes left right with
  | false => exact ⟨le_rfl, not_precedes_sound leftFinite rightFinite ordered⟩
  | true => exact ⟨precedes_sound leftFinite rightFinite ordered, le_rfl⟩

end MerlinVerify.Rules.W8A8RealOrder
