/- Exact cancellation and genuine negative-infinity mask cases for the pinned binary32
subtraction. An attained finite maximum supplies an actual exp(0)=1 row member. -/
import MerlinVerify.Rules.W8A8MaskedMaximum
import MerlinVerify.Rules.W8A8Transcendentals
import MerlinVerify.Rules.W8A8ArithmeticModel

namespace MerlinVerify.Rules.W8A8AttentionDifference

open MerlinVerify
open Float.Model (UnpackedFloat)
open W8A8MaskedMaximum

theorem subtract_self {bits : FBits .f32} (finite : W8A8IEEE32.Finite bits) :
    W8A8ArithmeticModel.subtract32 bits bits = (0 : FBits .f32) := by
  have notNaN := W8A8IEEE32.unpack_notNaN (W8A8IEEE32.finite_notNaN finite)
  have notInf := W8A8IEEE32.unpack_notInf finite
  unfold W8A8ArithmeticModel.subtract32
  cases decoded : W8A8IEEE32.unpack bits with
  | notANumber => exact (notNaN decoded).elim
  | infinity sign => exact (notInf sign decoded).elim
  | zero sign => cases sign <;> rfl
  | finite sign mantissa exponent positive =>
    simp [UnpackedFloat.sub, UnpackedFloat.normalize]
    rfl

theorem negative_infinity_subtract_finite {bits : FBits .f32} (finite : W8A8IEEE32.Finite bits) :
    W8A8ArithmeticModel.subtract32 negativeInfinity bits = negativeInfinity := by
  have notNaN := W8A8IEEE32.unpack_notNaN (W8A8IEEE32.finite_notNaN finite)
  have notInf := W8A8IEEE32.unpack_notInf finite
  have decodedInfinity : W8A8IEEE32.unpack negativeInfinity = .infinity .negative := by rfl
  unfold W8A8ArithmeticModel.subtract32
  rw [decodedInfinity]
  cases decoded : W8A8IEEE32.unpack bits with
  | notANumber => exact (notNaN decoded).elim
  | infinity sign => exact (notInf sign decoded).elim
  | zero sign => rfl
  | finite sign mantissa exponent positive => rfl

theorem negative_infinity_self_nan :
    W8A8ByteFloat.isNaN (W8A8ArithmeticModel.subtract32 negativeInfinity negativeInfinity) = true := by decide

theorem exponential_self_difference {bits : FBits .f32} (finite : W8A8IEEE32.Finite bits) :
    W8A8RoundedReal.exponential (W8A8ArithmeticModel.subtract32 bits bits) = (0x3f800000 : FBits .f32) := by
  rw [subtract_self finite, W8A8RoundedReal.exponential_zero]

theorem masked_exponential_zero {bits : FBits .f32} (finite : W8A8IEEE32.Finite bits) :
    W8A8RoundedReal.exponential (W8A8ArithmeticModel.subtract32 negativeInfinity bits) = (0 : FBits .f32) := by
  rw [negative_infinity_subtract_finite finite, W8A8RoundedReal.exponential_negative_infinity]

/-- With no unmasked member the maximum remains -Inf; its own shifted exponential is NaN.
The finite-member premise in the row theorem below cannot be dropped. -/
theorem all_masked_exponential_nan (count : Nat) :
    W8A8RoundedReal.exponential
      (W8A8ArithmeticModel.subtract32 negativeInfinity
        ((List.replicate count negativeInfinity).foldl W8A8ByteFloat.maximum negativeInfinity)) =
      (0x7fc00000 : FBits .f32) := by
  rw [all_masked]
  exact W8A8RoundedReal.exponential_nan negative_infinity_self_nan

theorem row_exponential_one {cap : Nat} (words : List (FBits .f32))
    (members : ∀ word ∈ words, Admissible cap word)
    (unmasked : ∃ word ∈ words, W8A8IEEEBounds.ExponentLE cap word) :
    ∃ word ∈ words, words.foldl W8A8ByteFloat.maximum negativeInfinity = word ∧
      W8A8RoundedReal.exponential (W8A8ArithmeticModel.subtract32 word
        (words.foldl W8A8ByteFloat.maximum negativeInfinity)) = (0x3f800000 : FBits .f32) := by
  obtain ⟨word, member, maximum, bounded⟩ := row_maximum_attained words members unmasked
  refine ⟨word, member, maximum, ?_⟩
  rw [maximum]
  exact exponential_self_difference bounded.1

end MerlinVerify.Rules.W8A8AttentionDifference
