/- Positive representable-scale mantissas cannot round to zero. These proofs follow both
actual rounding shifts and the actual packed exponent/fraction fields. -/
import MerlinVerify.Rules.W8A8IEEERoundingBounds

namespace MerlinVerify.Rules.W8A8PositiveRounding

open MerlinVerify Float.Model.UnpackedFloat
open Float.Model (Format UnpackedFloat totalExponent)

theorem rounded_mantissa_lower (mantissa : ExtendedMantissa) :
    mantissa.mantissa ≤ mantissa.roundedMantissa := by
  cases mantissa with
  | mk value round sticky =>
    cases round <;> cases sticky <;>
      simp [ExtendedMantissa.roundedMantissa, ExtendedMantissa.accuracy, Accuracy.roundToNearestEven]

theorem target_mantissa_positive (spec : Format) {mantissa : Nat} {exponent : Int} (accuracy : Accuracy)
    (positive : 0 < mantissa) (lower : spec.minExponent ≤ exponent) :
    0 < (shiftToTargetExponent spec mantissa exponent accuracy).1.mantissa := by
  have initial : (ExtendedMantissa.ofMantissaAndAccuracy mantissa accuracy).mantissa = mantissa := by
    cases accuracy with
    | exact => rfl
    | inexact ordering => cases ordering <;> rfl
  have amount : (spec.targetExponent (totalExponent mantissa exponent) - exponent).toNat ≤ mantissa.log2 := by
    simp only [Format.targetExponent, totalExponent, Format.mantissaBits]
    omega
  have denominator : 2^((spec.targetExponent (totalExponent mantissa exponent) - exponent).toNat) ≤ mantissa :=
    Nat.le_trans (Nat.pow_le_pow_right (by decide) amount) (Nat.log2_self_le (Nat.ne_of_gt positive))
  simp only [shiftToTargetExponent, shiftToExponent, W8A8IEEEBounds.shifted_mantissa, initial]
  exact Nat.div_pos denominator (Nat.two_pow_pos _)

theorem rounded_components (spec : Format) (sign : Sign) {mantissa : Nat} {exponent : Int} (accuracy : Accuracy)
    (positive : 0 < mantissa) (lower : spec.minExponent ≤ exponent) :
    ∃ result resultExponent nonzero,
      roundWithAccuracy spec sign mantissa exponent accuracy = .finite sign result resultExponent nonzero ∧
      result < 2^(spec.mantissaBitsWithoutImplicit + 1) ∧ spec.minExponent ≤ resultExponent := by
  let first := shiftToTargetExponent spec mantissa exponent accuracy
  let rounded := first.1.roundedMantissa
  let second := shiftToTargetExponent spec rounded first.2 .exact
  have firstPositive : 0 < first.1.mantissa := target_mantissa_positive spec accuracy positive lower
  have roundedPositive : 0 < rounded := Nat.lt_of_lt_of_le firstPositive (rounded_mantissa_lower first.1)
  have firstLower : spec.minExponent ≤ first.2 := by
    have actual := W8A8IEEERoundingBounds.shift_exponent spec mantissa exponent accuracy
    change first.2 = _ at actual
    rw [actual]
    omega
  have finalPositive : 0 < second.1.mantissa := target_mantissa_positive spec .exact roundedPositive firstLower
  have finalLower : spec.minExponent ≤ second.2 := by
    have actual := W8A8IEEERoundingBounds.shift_exponent spec rounded first.2 .exact
    change second.2 = _ at actual
    rw [actual]
    omega
  refine ⟨second.1.mantissa, second.2, finalPositive, ?_, ?_, finalLower⟩
  · change (if h : second.1.mantissa = 0 then UnpackedFloat.zero sign
      else UnpackedFloat.finite sign second.1.mantissa second.2 (Nat.pos_of_ne_zero h)) = _
    rw [dif_neg (Nat.ne_of_gt finalPositive)]
  · exact W8A8IEEERoundingBounds.target_mantissa_lt spec rounded first.2 .exact

theorem packed_nonzero (sign : Sign) {mantissa : Nat} {exponent : Int} (positive : 0 < mantissa)
    (upper : mantissa < 2^24) (lower : -149 ≤ exponent) :
    W8A8IEEE32.Nonzero (W8A8IEEE32.pack (.finite sign mantissa exponent positive)) := by
  change W8A8IEEE32.Nonzero (UnpackedFloat.pack Format.binary32 (.finite sign mantissa exponent positive))
  simp only [UnpackedFloat.pack]
  split
  · cases sign <;> decide
  · next noOverflow =>
    split
    · have exponentPositive : 0 < (exponent + 127 + 23).toNat := by omega
      have exponentSmall : (exponent + 127 + 23).toNat < 256 := by
        change ¬ 256 ≤ (exponent + 127 + 23).toNat + 1 at noOverflow
        omega
      intro zero
      change W8A8ByteFloat.magnitudeWord
        (packComponents Format.binary32 sign (BitVec.ofNat 8 (exponent + 127 + 23).toNat)
          (BitVec.ofNat 23 mantissa)) = 0 at zero
      have exponentZero : W8A8ByteFloat.exponent
          (packComponents Format.binary32 sign (BitVec.ofNat 8 (exponent + 127 + 23).toNat)
            (BitVec.ofNat 23 mantissa)) = 0 := by simp [W8A8ByteFloat.exponent, zero]
      rw [W8A8IEEE32.exponent_packComponents, BitVec.toNat_ofNat, Nat.mod_eq_of_lt exponentSmall] at exponentZero
      omega
    · next subnormal =>
      have small : mantissa < 2^23 := by
        by_contra tooLarge
        have logarithm : mantissa.log2 = 23 :=
          (Nat.log2_eq_iff (Nat.ne_of_gt positive)).2 ⟨by omega, upper⟩
        change ¬ mantissa.log2 + 1 = 24 at subnormal
        omega
      intro zero
      have fractionZero : W8A8ByteFloat.fraction
          (packComponents Format.binary32 sign 0 (BitVec.ofNat 23 mantissa)) = 0 := by
        simp [W8A8ByteFloat.fraction, zero]
      rw [W8A8IEEE32.fraction_packComponents, BitVec.toNat_ofNat, Nat.mod_eq_of_lt small] at fractionZero
      omega

theorem round_components (spec : Format) (sign : Sign) {mantissa : Nat} {exponent : Int}
    (positive : 0 < mantissa) (lower : spec.minExponent ≤ exponent) :
    ∃ result resultExponent nonzero,
      round spec sign mantissa exponent = .finite sign result resultExponent nonzero ∧
      result < 2^(spec.mantissaBitsWithoutImplicit + 1) ∧ spec.minExponent ≤ resultExponent := by
  let target := spec.targetExponent (totalExponent mantissa exponent)
  let amount := (exponent - target).toNat
  have targetLower : spec.minExponent ≤ target := by dsimp only [target, Format.targetExponent]; omega
  have decreasedLower : spec.minExponent ≤ exponent - (amount : Int) := by dsimp only [amount]; omega
  have shiftedPositive : 0 < mantissa <<< amount := by
    rw [Nat.shiftLeft_eq]
    exact Nat.mul_pos positive (Nat.two_pow_pos _)
  exact rounded_components spec sign .exact shiftedPositive decreasedLower

theorem round_nonzero (sign : Sign) {mantissa : Nat} {exponent : Int}
    (positive : 0 < mantissa) (lower : -149 ≤ exponent) :
    W8A8IEEE32.Nonzero (W8A8IEEE32.pack (round Format.binary32 sign mantissa exponent)) := by
  obtain ⟨result, resultExponent, nonzero, rounded, small, exponentLower⟩ :=
    round_components Format.binary32 sign positive lower
  rw [rounded]
  exact packed_nonzero sign nonzero small exponentLower

end MerlinVerify.Rules.W8A8PositiveRounding
