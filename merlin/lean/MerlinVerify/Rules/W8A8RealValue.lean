/- Exact binary32 encodings interpreted as signed real integer units.
Noncomputable mathematical candidate; no native fidelity or independent approval is claimed. -/
import MerlinVerify.Rules.W8A8OrderedMagnitude
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

namespace MerlinVerify.Rules.W8A8RoundedReal

open MerlinVerify Rules

noncomputable section

def magnitude (bits : FBits .f32) : ℝ :=
  (W8A8OrderedMagnitude.units bits : ℝ) / (2:ℝ)^149

def value (bits : FBits .f32) : ℝ :=
  if W8A8ByteFloat.negative bits then -magnitude bits else magnitude bits

theorem magnitude_nonneg (bits : FBits .f32) : 0 ≤ magnitude bits := by
  unfold magnitude
  positivity

theorem abs_value (bits : FBits .f32) : |value bits| = magnitude bits := by
  unfold value
  split <;> simp [abs_of_nonneg (magnitude_nonneg bits)]

theorem value_one : value (0x3f800000 : FBits .f32) = 1 := by
  norm_num [value, magnitude, W8A8OrderedMagnitude.units, W8A8ByteFloat.negative,
    W8A8ByteFloat.exponent, W8A8ByteFloat.fraction, W8A8ByteFloat.magnitudeWord]

theorem value_neg_one : value (0xbf800000 : FBits .f32) = -1 := by
  norm_num [value, magnitude, W8A8OrderedMagnitude.units, W8A8ByteFloat.negative,
    W8A8ByteFloat.exponent, W8A8ByteFloat.fraction, W8A8ByteFloat.magnitudeWord]

theorem units_zero_fields {bits : FBits .f32} (zero : W8A8OrderedMagnitude.units bits = 0) :
    W8A8ByteFloat.exponent bits = 0 ∧ W8A8ByteFloat.fraction bits = 0 := by
  unfold W8A8OrderedMagnitude.units at zero
  dsimp only at zero
  split at zero
  · exact ⟨by assumption, zero⟩
  · have positive := Nat.two_pow_pos (W8A8ByteFloat.exponent bits - 1)
    have nonzero : 0 < (2^23 + W8A8ByteFloat.fraction bits) *
        2^(W8A8ByteFloat.exponent bits - 1) := Nat.mul_pos (by omega) positive
    omega

theorem value_zero_units {bits : FBits .f32} (zero : value bits = 0) :
    W8A8OrderedMagnitude.units bits = 0 := by
  have absolute := abs_value bits
  rw [zero, abs_zero] at absolute
  have realZero : (W8A8OrderedMagnitude.units bits : ℝ) = 0 := by
    exact (div_eq_zero_iff.mp absolute.symm).resolve_right (by positivity)
  exact_mod_cast realZero

theorem value_one_encoding {bits : FBits .f32} (one : value bits = 1) :
    bits = (0x3f800000 : FBits .f32) := by
  have sign : W8A8ByteFloat.negative bits = false := by
    cases signed : W8A8ByteFloat.negative bits with
    | false => rfl
    | true =>
      have nonnegative := magnitude_nonneg bits
      simp only [value, signed, ↓reduceIte] at one
      linarith
  have unitsReal : (W8A8OrderedMagnitude.units bits : ℝ) = (2:ℝ)^149 := by
    simp only [value, sign, Bool.false_eq_true, ↓reduceIte, magnitude] at one
    exact (div_eq_one_iff_eq (by positivity)).mp one
  have unitsNat : W8A8OrderedMagnitude.units bits = (2:ℕ)^149 := by exact_mod_cast unitsReal
  have cap : W8A8ByteFloat.exponent bits ≤ 127 :=
    W8A8OrderedMagnitude.exponent_le_of_units (power := 150) (by rw [unitsNat]; decide)
  have finite : W8A8IEEE32.Finite bits := by change W8A8ByteFloat.exponent bits < 255; omega
  have exponent : W8A8ByteFloat.exponent bits = 127 := by
    by_contra different
    have smaller : W8A8IEEEBounds.ExponentLE 126 bits := ⟨finite, by omega⟩
    have strict := W8A8OrderedMagnitude.units_lt_of_exponent smaller
    rw [unitsNat] at strict
    exact (Nat.lt_irrefl (2^149)) strict
  have fraction : W8A8ByteFloat.fraction bits = 0 := by
    norm_num [W8A8OrderedMagnitude.units, exponent] at unitsNat
    omega
  apply BitVec.eq_of_toNat_eq
  have unsigned : bits.toNat < 2^31 := by
    have notNegative : ¬ 2^31 ≤ bits.toNat := of_decide_eq_false sign
    omega
  have e : bits.toNat / 2^23 = 127 := by
    simpa only [W8A8ByteFloat.exponent, W8A8ByteFloat.magnitudeWord, Nat.mod_eq_of_lt unsigned] using exponent
  have f : bits.toNat % 2^23 = 0 := by
    simpa only [W8A8ByteFloat.fraction, W8A8ByteFloat.magnitudeWord, Nat.mod_eq_of_lt unsigned] using fraction
  change bits.toNat = 0x3f800000
  omega

theorem units_zero_value {bits : FBits .f32} (zero : W8A8OrderedMagnitude.units bits = 0) :
    value bits = 0 := by
  simp [value, magnitude, zero]

theorem value_zero_encoding {bits : FBits .f32} (zero : value bits = 0) :
    bits = (0 : FBits .f32) ∨ bits = (0x80000000 : FBits .f32) := by
  obtain ⟨exponent, fraction⟩ := units_zero_fields (value_zero_units zero)
  have magnitudeZero : W8A8ByteFloat.magnitudeWord bits = 0 := by
    unfold W8A8ByteFloat.exponent at exponent
    unfold W8A8ByteFloat.fraction at fraction
    omega
  have bound : bits.toNat < 2^32 := bits.isLt
  unfold W8A8ByteFloat.magnitudeWord at magnitudeZero
  have encodings : bits.toNat = 0 ∨ bits.toNat = 0x80000000 := by omega
  rcases encodings with positive | negative
  · exact Or.inl (BitVec.eq_of_toNat_eq positive)
  · exact Or.inr (BitVec.eq_of_toNat_eq negative)

theorem value_positive_zero : value (0 : FBits .f32) = 0 := by
  norm_num [value, magnitude, W8A8OrderedMagnitude.units, W8A8ByteFloat.negative,
    W8A8ByteFloat.exponent, W8A8ByteFloat.fraction, W8A8ByteFloat.magnitudeWord]

theorem value_negative_zero : value (0x80000000 : FBits .f32) = 0 := by
  norm_num [value, magnitude, W8A8OrderedMagnitude.units, W8A8ByteFloat.negative,
    W8A8ByteFloat.exponent, W8A8ByteFloat.fraction, W8A8ByteFloat.magnitudeWord]

end
end MerlinVerify.Rules.W8A8RoundedReal
