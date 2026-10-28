/- Actual binary32 division by a finite value of magnitude at least one. The cap
follows the pinned division core and both rounding shifts, without a reciprocal law. -/
import MerlinVerify.Rules.W8A8IEEEDivision32
import MerlinVerify.Rules.W8A8UnitAdd

namespace MerlinVerify.Rules.W8A8DivisionBounds

open MerlinVerify Float.Model.UnpackedFloat
open Float.Model (Format UnpackedFloat totalExponent)

def divisionCap (numerator : Nat) : Nat := max (numerator + 1) 3

theorem core_bounded (leftSign rightSign : Sign) {left right upper : Nat}
    {leftExponent rightExponent : Int} (leftPositive : 0 < left) (rightPositive : 0 < right)
    (leftBound : left < 2^24) (safe : upper ≤ 253)
    (exponentBound : (divCore Format.binary32 left leftExponent right rightExponent).2.1 ≤
      max ((max upper 1 : Nat) - (151 : Int)) (-149)) :
    W8A8IEEEBounds.ExponentLE (divisionCap upper)
      (W8A8IEEE32.pack (UnpackedFloat.div Format.binary32
        (.finite leftSign left leftExponent leftPositive)
        (.finite rightSign right rightExponent rightPositive))) := by
  let core := divCore Format.binary32 left leftExponent right rightExponent
  have mantissa : core.1 < 2^25 :=
    W8A8IEEERoundingBounds.division_mantissa_lt Format.binary32 leftExponent rightExponent
      leftPositive rightPositive leftBound
  have logarithm : core.1.log2 ≤ 24 := by
    by_cases zero : core.1 = 0
    · simp [zero]
    · have bound := (Nat.log2_lt zero).2 mantissa
      omega
  change W8A8IEEEBounds.ExponentLE (divisionCap upper)
    (W8A8IEEE32.pack (roundWithAccuracy Format.binary32 (leftSign / rightSign) core.1 core.2.1 core.2.2))
  rcases W8A8IEEERoundingBounds.rounded_zero_or_finite Format.binary32
      (leftSign / rightSign) core.1 core.2.1 core.2.2 with rounded | ⟨m, e, positive, rounded⟩
  · rw [rounded]
    cases leftSign <;> cases rightSign <;>
      simp [W8A8IEEEBounds.ExponentLE, W8A8IEEE32.pack, UnpackedFloat.pack, packedZero,
        W8A8FiniteLaws.Finite, W8A8IEEE32.exponent_packComponents]
  · rw [rounded]
    have bound := W8A8IEEERoundingBounds.rounded_exponent rounded
    simp only [Format.minExponent, Format.mantissaBits] at bound
    change core.2.1 ≤ max ((max upper 1 : Nat) - (151 : Int)) (-149) at exponentBound
    apply W8A8IEEEBounds.pack_exponent_le (by omega)
    simp only [divisionCap]
    omega

/-- The denominator threshold is checked on its encoding, not inferred from a
nonzero-only premise. Numerator signs and underflow/signed zero remain unrestricted. -/
theorem divide_by_unit_bounded {numerator denominator : FBits .f32} {upper : Nat}
    (numeratorBound : W8A8IEEEBounds.ExponentLE upper numerator)
    (denominatorFinite : W8A8IEEE32.Finite denominator)
    (denominatorLower : 127 ≤ W8A8ByteFloat.exponent denominator) (safe : upper ≤ 253) :
    W8A8IEEEBounds.ExponentLE (divisionCap upper) (W8A8IEEE32.div numerator denominator) := by
  have numeratorNotNaN := W8A8IEEE32.unpack_notNaN (W8A8IEEE32.finite_notNaN numeratorBound.1)
  have numeratorNotInf := W8A8IEEE32.unpack_notInf numeratorBound.1
  have denominatorNotNaN := W8A8IEEE32.unpack_notNaN (W8A8IEEE32.finite_notNaN denominatorFinite)
  have denominatorNotInf := W8A8IEEE32.unpack_notInf denominatorFinite
  have denominatorNonzero : W8A8IEEE32.Nonzero denominator := by
    intro zero
    have exponentZero : W8A8ByteFloat.exponent denominator = 0 := by
      simp [W8A8ByteFloat.exponent, zero]
    omega
  have denominatorNotZero := W8A8IEEE32.unpack_notZero denominatorNonzero
  cases denominatorDecoded : W8A8IEEE32.unpack denominator with
  | notANumber => exact (denominatorNotNaN denominatorDecoded).elim
  | infinity sign => exact (denominatorNotInf sign denominatorDecoded).elim
  | zero sign => exact (denominatorNotZero sign denominatorDecoded).elim
  | finite denominatorSign denominatorMantissa denominatorExponent denominatorPositive =>
    obtain ⟨denominatorMantissaLower, denominatorExponentLower⟩ :=
      W8A8UnitAdd.unpack_unit_fields denominatorDecoded denominatorLower
    have denominatorMantissaUpper := (W8A8PositiveAdd.unpack_fields denominatorDecoded).1
    have denominatorLogarithm : denominatorMantissa.log2 = 23 :=
      (Nat.log2_eq_iff (Nat.ne_of_gt denominatorPositive)).2
        ⟨denominatorMantissaLower, denominatorMantissaUpper⟩
    cases numeratorDecoded : W8A8IEEE32.unpack numerator with
    | notANumber => exact (numeratorNotNaN numeratorDecoded).elim
    | infinity sign => exact (numeratorNotInf sign numeratorDecoded).elim
    | zero sign =>
      simp only [W8A8IEEE32.div, numeratorDecoded, denominatorDecoded, UnpackedFloat.div]
      cases sign <;> cases denominatorSign <;>
        simp [W8A8IEEEBounds.ExponentLE, W8A8IEEE32.pack, UnpackedFloat.pack, packedZero,
          W8A8FiniteLaws.Finite, W8A8IEEE32.exponent_packComponents]
    | finite numeratorSign numeratorMantissa numeratorExponent numeratorPositive =>
      obtain ⟨numeratorMantissaUpper, numeratorExponentUpper⟩ :=
        W8A8IEEEBounds.unpack_finite_bounds numeratorBound numeratorDecoded
      have numeratorLogarithm : numeratorMantissa.log2 ≤ 23 := by
        have bound := (Nat.log2_lt (Nat.ne_of_gt numeratorPositive)).2 numeratorMantissaUpper
        omega
      simp only [W8A8IEEE32.div, numeratorDecoded, denominatorDecoded]
      apply core_bounded _ _ numeratorPositive denominatorPositive numeratorMantissaUpper safe
      simp only [divCore, Format.targetExponent, totalExponent, Format.mantissaBits,
        Format.minExponent, denominatorLogarithm]
      change min (numeratorExponent - denominatorExponent)
        (max ((numeratorMantissa.log2 : Int) + 1 + numeratorExponent -
          (23 + 1 + denominatorExponent) - (1 + 23)) (-149)) ≤ _
      omega

end MerlinVerify.Rules.W8A8DivisionBounds
