/- Finiteness of the two actual binary32 division forms required by dynamic W8A8.
Bounds follow the pinned division core and both rounding shifts, including underflow. -/
import MerlinVerify.Rules.W8A8IEEEWiden

namespace MerlinVerify.Rules.W8A8IEEEDivision32

open MerlinVerify Float.Model.UnpackedFloat
open Float.Model (Format UnpackedFloat totalExponent)

theorem finite_core (leftSign rightSign : Sign) {left right : Nat} {leftExp rightExp : Int}
    (leftPositive : 0 < left) (rightPositive : 0 < right)
    (leftBound : left < 2^24)
    (exponentBound : (divCore Format.binary32 left leftExp right rightExp).2.1 ≤ 102) :
    W8A8FiniteLaws.Finite (W8A8IEEE32.pack (UnpackedFloat.div Format.binary32
      (.finite leftSign left leftExp leftPositive) (.finite rightSign right rightExp rightPositive))) := by
  let core := divCore Format.binary32 left leftExp right rightExp
  have mantissa : core.1 < 2^25 :=
    W8A8IEEERoundingBounds.division_mantissa_lt Format.binary32 leftExp rightExp leftPositive rightPositive leftBound
  have logarithm : core.1.log2 ≤ 24 := by
    by_cases zero : core.1 = 0
    · simp [zero]
    · have bound := (Nat.log2_lt zero).2 mantissa
      omega
  change W8A8FiniteLaws.Finite
    (W8A8IEEE32.pack (roundWithAccuracy Format.binary32 (leftSign / rightSign) core.1 core.2.1 core.2.2))
  rcases W8A8IEEERoundingBounds.rounded_zero_or_finite Format.binary32
      (leftSign / rightSign) core.1 core.2.1 core.2.2 with rounded | ⟨m, e, positive, rounded⟩
  · rw [rounded]
    cases leftSign <;> cases rightSign <;> decide
  · rw [rounded]
    have bound := W8A8IEEERoundingBounds.rounded_exponent rounded
    simp only [Format.minExponent, Format.mantissaBits] at bound
    apply W8A8IEEEBounds.pack_finite_exponent
    change core.2.1 ≤ 102 at exponentBound
    omega

theorem scale_divide_finite {x : FBits .f32} (finite : W8A8FiniteLaws.Finite x) :
    W8A8FiniteLaws.Finite (W8A8IEEE32.div x W8A8Scale.denominator) := by
  have nonNaN := W8A8IEEE32.unpack_notNaN (W8A8IEEE32.finite_notNaN finite)
  have notInf := W8A8IEEE32.unpack_notInf finite
  have divisor : W8A8IEEE32.unpack W8A8Scale.denominator =
      .finite .positive 16646144 (-17) (by decide) := rfl
  cases decoded : W8A8IEEE32.unpack x with
  | notANumber => exact False.elim (nonNaN decoded)
  | infinity sign => exact False.elim (notInf sign decoded)
  | zero sign =>
    simp only [W8A8IEEE32.div, decoded, divisor, UnpackedFloat.div]
    cases sign <;> decide
  | finite sign m e positive =>
    have facts : W8A8IEEEBounds.ExponentLE 254 x := ⟨finite, by omega⟩
    obtain ⟨mantissa, exponent⟩ := W8A8IEEEBounds.unpack_finite_bounds facts decoded
    have logBound := (Nat.log2_lt (Nat.ne_of_gt positive)).2 mantissa
    simp only [W8A8IEEE32.div, decoded, divisor]
    apply finite_core _ _ positive (by decide) mantissa
    simp only [divCore, Format.targetExponent, totalExponent, Format.mantissaBits,
      Format.minExponent]
    change min (e - (-17))
      (max ((m.log2 : Int) + 1 + e - (23 + 1 + (-17)) - (1 + 23)) (-149)) ≤ 102
    omega

theorem reciprocal_finite {scale : FBits .f32} (normal : W8A8FiniteLaws.scaleCheck scale = true) :
    W8A8FiniteLaws.Finite (W8A8IEEE32.div 0x3f800000#32 scale) := by
  obtain ⟨m, e, positive, decoded, low, high, lower, upper⟩ := W8A8IEEEWiden.normal_fields normal
  have log : m.log2 = 23 := (Nat.log2_eq_iff (Nat.ne_of_gt positive)).2 ⟨low, high⟩
  simp only [W8A8IEEE32.div, decoded, W8A8IEEE32.unpack_one]
  apply finite_core _ _ (by decide) positive (by decide)
  simp only [divCore, Format.targetExponent, totalExponent, log, Format.mantissaBits,
    Format.minExponent]
  change min ((-23) - e) (max ((23 : Int) + 1 + (-23) - (23 + 1 + e) - (1 + 23)) (-149)) ≤ 102
  omega

#print axioms scale_divide_finite
#print axioms reciprocal_finite

end MerlinVerify.Rules.W8A8IEEEDivision32
