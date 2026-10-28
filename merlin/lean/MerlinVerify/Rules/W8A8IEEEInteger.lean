/- Finite signed-i32/i64 conversion for the same explicit arithmetic base. The proof bounds
the actual rounded exponent; it neither enumerates integers nor assumes conversion laws. -/
import MerlinVerify.Rules.W8A8IEEEPrimitives
import MerlinVerify.Rules.W8A8IEEERoundingBounds

namespace MerlinVerify.Rules.W8A8IEEEInteger

open MerlinVerify Float.Model.UnpackedFloat
open Float.Model (Format UnpackedFloat)

theorem round_integral_finite (s : Sign) {m : Nat} (bound : m < 2^64) :
    W8A8FiniteLaws.Finite (W8A8IEEE32.pack (round Format.binary32 s m 0)) := by
  by_cases zero : m = 0
  · subst m
    cases s <;> decide
  have positive := Nat.pos_of_ne_zero zero
  rcases W8A8IEEERoundingBounds.round_zero_or_finite Format.binary32 s m 0 with
      rounded | ⟨mantissa, exponent, nonzero, rounded⟩
  · rw [rounded]
    cases s <;> decide
  · rw [rounded]
    have exponentBound := W8A8IEEERoundingBounds.round_exponent positive rounded
    have logarithm := (Nat.log2_lt zero).2 bound
    apply W8A8IEEEBounds.pack_finite_exponent
    simp only [Format.targetExponent, Float.Model.totalExponent, Format.mantissaBits,
      Format.minExponent] at exponentBound
    omega

theorem ofInt_finite {value : Int} (lower : -(2^63) ≤ value) (upper : value ≤ 2^63) :
    W8A8FiniteLaws.Finite (W8A8IEEE32.pack (ofInt Format.binary32 value)) := by
  simp only [ofInt, normalize]
  split
  · next negative =>
    have sign : value < 0 := by simpa only [Int.compare_eq_lt] using negative
    exact round_integral_finite _ (by omega)
  · decide
  · next positive =>
    have sign : 0 < value := by simpa only [Int.compare_eq_gt] using positive
    exact round_integral_finite _ (by omega)

theorem signed_finite {width : Nat} (allowed : width = 32 ∨ width = 64) (value : BitVec width) :
    W8A8FiniteLaws.Finite (W8A8ArithmeticModel.signedInteger32 value) := by
  have lower := BitVec.le_toInt value
  have upper := BitVec.toInt_lt (x := value)
  apply ofInt_finite
  · rcases allowed with rfl | rfl <;> omega
  · rcases allowed with rfl | rfl <;> omega

#print axioms signed_finite

end MerlinVerify.Rules.W8A8IEEEInteger
