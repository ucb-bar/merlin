/- Concrete finite-output and exponent bounds for one binary32 subtraction. The shared
pair-bound conversion is reused; the signed aligned subtraction and rounding are checked. -/
import MerlinVerify.Rules.W8A8PairBounds
import MerlinVerify.Rules.W8A8ArithmeticModel

namespace MerlinVerify.Rules.W8A8SubtractionBounds

open MerlinVerify W8A8OrderedMagnitude
open Float.Model (UnpackedFloat)
open Float.Model.UnpackedFloat

theorem repack_with_sign {bits : FBits .f32} {original : Sign} {mantissa : Nat} {exponent : Int}
    {positive : 0 < mantissa} (finite : W8A8IEEE32.Finite bits)
    (decoded : W8A8IEEE32.unpack bits = .finite original mantissa exponent positive) (sign : Sign) :
    W8A8IEEE32.Finite (W8A8IEEE32.pack (.finite sign mantissa exponent positive)) ∧
      units (W8A8IEEE32.pack (.finite sign mantissa exponent positive)) ≤ units bits := by
  obtain ⟨lower, exactUnits⟩ := unpack_units decoded
  have upper := (W8A8IEEEBounds.unpack_finite_bounds
    (show W8A8IEEEBounds.ExponentLE 254 bits from ⟨finite, by change W8A8ByteFloat.exponent bits < 255 at finite; omega⟩)
    decoded).2
  change exponent ≤ 104 at upper
  refine ⟨W8A8IEEEBounds.pack_finite_exponent upper, ?_⟩
  rw [exactUnits]
  exact pack_units_le lower upper

theorem subtract_units_bounded {left right : FBits .f32} {limit : Nat}
    (leftFinite : W8A8IEEE32.Finite left) (rightFinite : W8A8IEEE32.Finite right)
    (safe : limit ≤ 252) (bound : units left + units right < 2^(limit + 24)) :
    W8A8IEEE32.Finite (W8A8ArithmeticModel.subtract32 left right) ∧
      units (W8A8ArithmeticModel.subtract32 left right) ≤ units left + units right + 2^limit := by
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
      split
      all_goals exact ⟨(zero_units _).1, by rw [(zero_units _).2]; exact Nat.zero_le _⟩
    | finite rightSign mantissa exponent positive =>
      have repacked := repack_with_sign rightFinite rightDecoded (-rightSign)
      simp only [W8A8ArithmeticModel.subtract32, leftDecoded, rightDecoded, UnpackedFloat.sub]
      have widened : units right ≤ units left + units right + 2^limit :=
        Nat.le_trans (Nat.le_add_left _ _) (Nat.le_add_right _ _)
      exact ⟨repacked.1, Nat.le_trans repacked.2 widened⟩
  | finite leftSign leftMantissa leftExponent leftPositive =>
    cases rightDecoded : W8A8IEEE32.unpack right with
    | notANumber => exact (rightNotNaN rightDecoded).elim
    | infinity sign => exact (rightNotInf sign rightDecoded).elim
    | zero rightSign =>
      have repacked := repack_with_sign leftFinite leftDecoded leftSign
      simp only [W8A8ArithmeticModel.subtract32, leftDecoded, rightDecoded, UnpackedFloat.sub]
      have widened : units left ≤ units left + units right + 2^limit :=
        Nat.le_trans (Nat.le_add_right _ _) (Nat.le_add_right _ _)
      exact ⟨repacked.1, Nat.le_trans repacked.2 widened⟩
    | finite rightSign rightMantissa rightExponent rightPositive =>
      obtain ⟨leftLower, leftUnits⟩ := unpack_units leftDecoded
      obtain ⟨rightLower, rightUnits⟩ := unpack_units rightDecoded
      let common := min leftExponent rightExponent
      let difference := leftSign.apply (leftMantissa <<< (leftExponent - common).toNat) -
        rightSign.apply (rightMantissa <<< (rightExponent - common).toNat)
      have commonLower : -149 ≤ common := by dsimp only [common]; omega
      have leftAligned : common ≤ leftExponent := by dsimp only [common]; omega
      have rightAligned : common ≤ rightExponent := by dsimp only [common]; omega
      have differenceBound : difference.natAbs * 2^(common + 149).toNat ≤ units left + units right := by
        have triangle := Nat.mul_le_mul_right (2^(common + 149).toNat)
          (Int.natAbs_sub_le (leftSign.apply (leftMantissa <<< (leftExponent - common).toNat))
            (rightSign.apply (rightMantissa <<< (rightExponent - common).toNat)))
        rw [Nat.add_mul, apply_natAbs, apply_natAbs, aligned_units _ commonLower leftAligned,
          aligned_units _ commonLower rightAligned, ← leftUnits, ← rightUnits] at triangle
        exact triangle
      have normalized := normalize_bounded .positive commonLower safe (Nat.lt_of_le_of_lt differenceBound bound)
      simp only [W8A8ArithmeticModel.subtract32, leftDecoded, rightDecoded, UnpackedFloat.sub]
      change W8A8IEEE32.Finite (W8A8IEEE32.pack
          (UnpackedFloat.normalize W8A8IEEE32.format difference common .positive)) ∧
        units (W8A8IEEE32.pack (UnpackedFloat.normalize W8A8IEEE32.format difference common .positive)) ≤ _
      exact ⟨normalized.1, Nat.le_trans normalized.2 (Nat.add_le_add_right differenceBound _)⟩

theorem subtract_bounded {left right : FBits .f32} {leftCap rightCap : Nat}
    (leftBound : W8A8IEEEBounds.ExponentLE leftCap left)
    (rightBound : W8A8IEEEBounds.ExponentLE rightCap right) (safe : max leftCap rightCap ≤ 252) :
    W8A8IEEEBounds.ExponentLE (W8A8PairBounds.addCap leftCap rightCap)
      (W8A8ArithmeticModel.subtract32 left right) := by
  obtain ⟨finite, rounded⟩ := subtract_units_bounded leftBound.1 rightBound.1 safe
    (W8A8PairBounds.input_units_bound leftBound rightBound)
  exact W8A8PairBounds.result_bounded leftBound rightBound finite rounded

end MerlinVerify.Rules.W8A8SubtractionBounds
