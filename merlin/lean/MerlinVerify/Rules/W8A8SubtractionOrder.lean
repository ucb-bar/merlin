/- Actual exponent-aligned binary32 subtraction preserves nonpositive differences.
The premise compares finite operand values; the result sign follows from the computed
integer mantissa and the existing rounding model, not an assumed transcendental law. -/
import MerlinVerify.Rules.W8A8RealSign
import MerlinVerify.Rules.W8A8ArithmeticModel

namespace MerlinVerify.Rules.W8A8SubtractionOrder

open MerlinVerify W8A8RoundedReal W8A8RealSign W8A8OrderedMagnitude
open Float.Model (UnpackedFloat)
open Float.Model.UnpackedFloat

theorem finite_difference_nonpositive {left right : FBits .f32}
    (leftFinite : W8A8IEEE32.Finite left) (rightFinite : W8A8IEEE32.Finite right)
    (ordered : value left ≤ value right) : value (W8A8ArithmeticModel.subtract32 left right) ≤ 0 := by
  have leftNotNaN := W8A8IEEE32.unpack_notNaN (W8A8IEEE32.finite_notNaN leftFinite)
  have rightNotNaN := W8A8IEEE32.unpack_notNaN (W8A8IEEE32.finite_notNaN rightFinite)
  have leftNotInf := W8A8IEEE32.unpack_notInf leftFinite
  have rightNotInf := W8A8IEEE32.unpack_notInf rightFinite
  cases leftDecoded : W8A8IEEE32.unpack left with
  | notANumber => exact (leftNotNaN leftDecoded).elim
  | infinity sign => exact (leftNotInf sign leftDecoded).elim
  | zero leftSign =>
    cases rightDecoded : W8A8IEEE32.unpack right with
    | notANumber => exact (rightNotNaN rightDecoded).elim
    | infinity sign => exact (rightNotInf sign rightDecoded).elim
    | zero rightSign =>
      simp only [W8A8ArithmeticModel.subtract32, leftDecoded, rightDecoded, UnpackedFloat.sub]
      split <;> exact le_of_eq (units_zero_value (zero_units _).2)
    | finite rightSign mantissa exponent positive =>
      have nonnegative : 0 ≤ value right := by rw [value_zero_unpack leftDecoded] at ordered; exact ordered
      have sign := unpack_nonnegative_sign rightDecoded nonnegative
      subst rightSign
      simp only [W8A8ArithmeticModel.subtract32, leftDecoded, rightDecoded, UnpackedFloat.sub]
      exact pack_negative_value rfl
  | finite leftSign leftMantissa leftExponent leftPositive =>
    cases rightDecoded : W8A8IEEE32.unpack right with
    | notANumber => exact (rightNotNaN rightDecoded).elim
    | infinity sign => exact (rightNotInf sign rightDecoded).elim
    | zero rightSign =>
      have nonpositive : value left ≤ 0 := by rw [value_zero_unpack rightDecoded] at ordered; exact ordered
      have sign := unpack_nonpositive_sign leftDecoded nonpositive
      subst leftSign
      simp only [W8A8ArithmeticModel.subtract32, leftDecoded, rightDecoded, UnpackedFloat.sub]
      exact pack_negative_value rfl
    | finite rightSign rightMantissa rightExponent rightPositive =>
      let common := min leftExponent rightExponent
      let difference := leftSign.apply ((leftMantissa <<< (leftExponent - common).toNat : Nat) : Int) -
        rightSign.apply ((rightMantissa <<< (rightExponent - common).toNat : Nat) : Int)
      have leftLower := (unpack_units leftDecoded).1
      have rightLower := (unpack_units rightDecoded).1
      have commonLower : -149 ≤ common := by dsimp only [common]; omega
      have leftAligned : common ≤ leftExponent := by dsimp only [common]; omega
      have rightAligned : common ≤ rightExponent := by dsimp only [common]; omega
      have leftExact := aligned_signed_units leftDecoded commonLower leftAligned
      have rightExact := aligned_signed_units rightDecoded commonLower rightAligned
      have differenceExact : difference * ((2^(common + 149).toNat : Nat) : Int) =
          signedUnits left - signedUnits right := by
        dsimp only [difference]
        rw [sub_mul, leftExact, rightExact]
      have unitsOrdered := signed_units_le_of_value_le ordered
      have differenceNonpositive : difference ≤ 0 := by
        by_contra positive
        have powerPositive : (0:Int) < ((2^(common + 149).toNat : Nat) : Int) := by
          exact_mod_cast Nat.two_pow_pos (common + 149).toNat
        have positiveProduct := mul_pos (show 0 < difference by omega) powerPositive
        rw [differenceExact] at positiveProduct
        omega
      simp only [W8A8ArithmeticModel.subtract32, leftDecoded, rightDecoded, UnpackedFloat.sub]
      change value (W8A8IEEE32.pack
        (UnpackedFloat.normalize W8A8IEEE32.format difference common .positive)) ≤ 0
      exact normalize_nonpositive common differenceNonpositive

end MerlinVerify.Rules.W8A8SubtractionOrder
