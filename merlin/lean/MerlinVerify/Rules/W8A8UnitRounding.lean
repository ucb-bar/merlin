/- Actual rounding at a representable unit threshold retains its normal leading bit.
Both shifts and packing are proved; no abstract rounding-order law is assumed. -/
import MerlinVerify.Rules.W8A8PositiveRounding

namespace MerlinVerify.Rules.W8A8UnitRounding

open MerlinVerify Rules Float.Model.UnpackedFloat
open Float.Model (Format UnpackedFloat totalExponent)

/-- When a positive canonical mantissa has at least24bits, truncating it to24bits
retains the implicit leading bit. The exponent premise excludes a subnormal clamp. -/
theorem first_mantissa_lower {mantissa : Nat} {exponent : Int} (accuracy : Accuracy)
    (positive : 0 < mantissa) (wide : 23 ≤ mantissa.log2)
    (normal : -149 ≤ (mantissa.log2 : Int) + exponent - 23) :
    2^23 ≤ (shiftToTargetExponent Format.binary32 mantissa exponent accuracy).1.mantissa := by
  have initial : (ExtendedMantissa.ofMantissaAndAccuracy mantissa accuracy).mantissa = mantissa := by
    cases accuracy with
    | exact => rfl
    | inexact ordering => cases ordering <;> rfl
  have amount : (Format.binary32.targetExponent (totalExponent mantissa exponent) - exponent).toNat =
      mantissa.log2 - 23 := by
    simp only [Format.targetExponent, totalExponent, Format.mantissaBits,
      Format.minExponent] at *
    omega
  simp only [shiftToTargetExponent, shiftToExponent, W8A8IEEEBounds.shifted_mantissa, initial, amount]
  have bound : 2^23 * 2^(mantissa.log2 - 23) ≤ mantissa := by
    rw [← Nat.pow_add, Nat.add_sub_of_le wide]
    exact Nat.log2_self_le (Nat.ne_of_gt positive)
  exact (Nat.le_div_iff_mul_le (Nat.two_pow_pos _)).2 bound

theorem second_mantissa_lower {mantissa : Nat} {exponent : Int}
    (lower : 2^23 ≤ mantissa) (upper : mantissa ≤ 2^24) (exponentLower : -23 ≤ exponent) :
    2^23 ≤ (shiftToTargetExponent Format.binary32 mantissa exponent .exact).1.mantissa := by
  by_cases endpoint : mantissa = 2^24
  · have logarithm : mantissa.log2 = 24 := by rw [endpoint]; decide
    have target : Format.binary32.targetExponent (totalExponent mantissa exponent) = exponent + 1 := by
      simp only [Format.targetExponent, totalExponent, Format.mantissaBits,
        Format.minExponent, logarithm]
      omega
    simp only [shiftToTargetExponent, shiftToExponent, W8A8IEEEBounds.shifted_mantissa, target]
    have amount : (exponent + 1 - exponent).toNat = 1 := by omega
    rw [amount]
    change 2^23 ≤ mantissa / 2^1
    rw [endpoint]
    decide
  · have small : mantissa < 2^24 := by omega
    have nonzero : mantissa ≠ 0 := by omega
    have logarithm : mantissa.log2 = 23 := (Nat.log2_eq_iff nonzero).2 ⟨lower, small⟩
    have target : Format.binary32.targetExponent (totalExponent mantissa exponent) = exponent := by
      simp only [Format.targetExponent, totalExponent, Format.mantissaBits,
        Format.minExponent, logarithm]
      omega
    simp only [shiftToTargetExponent, shiftToExponent, W8A8IEEEBounds.shifted_mantissa, target, Int.sub_self,
      Int.toNat_zero, Nat.pow_zero, Nat.div_one, ExtendedMantissa.ofMantissaAndAccuracy]
    exact lower

theorem round_unit_components (sign : Sign) {mantissa : Nat} {exponent : Int}
    (positive : 0 < mantissa) (leading : 0 ≤ (mantissa.log2 : Int) + exponent) :
    ∃ result resultExponent nonzero,
      round Format.binary32 sign mantissa exponent = .finite sign result resultExponent nonzero ∧
      (2^23 ≤ result ∧ result < 2^24) ∧ -23 ≤ resultExponent := by
  let target := Format.binary32.targetExponent (totalExponent mantissa exponent)
  let amount := (exponent - target).toNat
  let shifted := mantissa <<< amount
  let decreased := exponent - (amount : Int)
  let first := shiftToTargetExponent Format.binary32 shifted decreased .exact
  let rounded := first.1.roundedMantissa
  let second := shiftToTargetExponent Format.binary32 rounded first.2 .exact
  have targetEquation : target = (mantissa.log2 : Int) + exponent - 23 := by
    dsimp only [target, Format.targetExponent, totalExponent, Format.mantissaBits, Format.binary32,
      Format.minExponent]
    omega
  have decreasedUpper : decreased ≤ target := by dsimp only [decreased, amount]; omega
  have shiftedPositive : 0 < shifted := by
    dsimp only [shifted]
    rw [Nat.shiftLeft_eq]
    exact Nat.mul_pos positive (Nat.two_pow_pos _)
  have logarithm : shifted.log2 = mantissa.log2 + amount :=
    W8A8IEEERoundingBounds.log2_shift positive amount
  have preserved : (shifted.log2 : Int) + decreased = (mantissa.log2 : Int) + exponent := by
    rw [logarithm]
    dsimp only [decreased]
    omega
  have wide : 23 ≤ shifted.log2 := by omega
  have normal : -149 ≤ (shifted.log2 : Int) + decreased - 23 := by omega
  have firstLower : 2^23 ≤ first.1.mantissa := first_mantissa_lower .exact shiftedPositive wide normal
  have firstExponent : -23 ≤ first.2 := by
    have actual := W8A8IEEERoundingBounds.shift_exponent Format.binary32 shifted decreased .exact
    change first.2 = _ at actual
    rw [actual]
    simp only [Format.binary32] at *
    omega
  have roundedLower : 2^23 ≤ rounded :=
    Nat.le_trans firstLower (W8A8PositiveRounding.rounded_mantissa_lower first.1)
  have roundedUpper : rounded ≤ 2^24 := by
    have small := W8A8IEEERoundingBounds.target_mantissa_lt Format.binary32 shifted decreased .exact
    have bounded := W8A8IEEEBounds.roundedMantissa_le first.1
    change first.1.mantissa < 2^24 at small
    change rounded ≤ first.1.mantissa + 1 at bounded
    omega
  have finalLower : 2^23 ≤ second.1.mantissa := second_mantissa_lower roundedLower roundedUpper firstExponent
  have finalPositive : 0 < second.1.mantissa := Nat.lt_of_lt_of_le (by decide) finalLower
  have finalExponent : -23 ≤ second.2 := by
    have actual := W8A8IEEERoundingBounds.shift_exponent Format.binary32 rounded first.2 .exact
    change second.2 = _ at actual
    rw [actual]
    omega
  refine ⟨second.1.mantissa, second.2, finalPositive, ?_, ⟨finalLower, ?_⟩, finalExponent⟩
  · change (if h : second.1.mantissa = 0 then UnpackedFloat.zero sign
      else UnpackedFloat.finite sign second.1.mantissa second.2 (Nat.pos_of_ne_zero h)) = _
    rw [dite_eq_right (Nat.ne_of_gt finalPositive)]
  · exact W8A8IEEERoundingBounds.target_mantissa_lt Format.binary32 rounded first.2 .exact

theorem packed_unit_lower (sign : Sign) {mantissa : Nat} {exponent : Int} (positive : 0 < mantissa)
    (lower : 2^23 ≤ mantissa) (upper : mantissa < 2^24) (exponentLower : -23 ≤ exponent) :
    127 ≤ W8A8ByteFloat.exponent (W8A8IEEE32.pack (.finite sign mantissa exponent positive)) := by
  have logarithm : mantissa.log2 = 23 := (Nat.log2_eq_iff (Nat.ne_of_gt positive)).2 ⟨lower, upper⟩
  have normal : mantissa.log2 + 1 = Format.binary32.mantissaBits := by rw [logarithm]; rfl
  simp only [W8A8IEEE32.pack, UnpackedFloat.pack, normal, ↓reduceIte]
  split
  · cases sign <;> decide
  · next noOverflow =>
    have small : (exponent + 127 + 23).toNat < 256 := by
      change ¬ 256 ≤ (exponent + 127 + 23).toNat + 1 at noOverflow
      omega
    have large : 127 ≤ (exponent + 127 + 23).toNat := by omega
    simp only [W8A8IEEE32.exponent_packComponents, BitVec.toNat_ofNat]
    change 127 ≤ (exponent + 127 + 23).toNat % 256
    rw [Nat.mod_eq_of_lt small]
    exact large

theorem rounded_unit_lower (sign : Sign) {mantissa : Nat} {exponent : Int}
    (positive : 0 < mantissa) (leading : 0 ≤ (mantissa.log2 : Int) + exponent) :
    127 ≤ W8A8ByteFloat.exponent (W8A8IEEE32.pack (round Format.binary32 sign mantissa exponent)) := by
  obtain ⟨result, resultExponent, nonzero, rounded, ⟨lower, upper⟩, exponentLower⟩ :=
    round_unit_components sign positive leading
  rw [rounded]
  exact packed_unit_lower sign nonzero lower upper exponentLower

theorem leading_of_units {mantissa : Nat} {exponent : Int} (lower : -149 ≤ exponent)
    (units : 2^149 ≤ mantissa * 2^(exponent + 149).toNat) : 0 ≤ (mantissa.log2 : Int) + exponent := by
  by_contra negative
  have powers : mantissa.log2 + 1 + (exponent + 149).toNat ≤ 149 := by omega
  have strict : mantissa * 2^(exponent + 149).toNat < 2^(mantissa.log2 + 1 + (exponent + 149).toNat) := by
    rw [Nat.pow_add]
    exact Nat.mul_lt_mul_of_pos_right Nat.lt_log2_self (Nat.two_pow_pos _)
  have bound := Nat.lt_of_lt_of_le strict (Nat.pow_le_pow_right (by decide) powers)
  omega


end MerlinVerify.Rules.W8A8UnitRounding
