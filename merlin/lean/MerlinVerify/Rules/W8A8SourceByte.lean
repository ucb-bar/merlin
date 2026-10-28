/- Full signed-byte clipping for the captured-source quantization review candidate.
This differs from the dynamic policy's symmetric -127/127 range. -/
import MerlinVerify.Rules.W8A8ByteModel

namespace MerlinVerify.Rules.W8A8SourceByte

open MerlinVerify W8A8ByteFloat

def clamp (x : FBits .f32) : FBits .f32 :=
  minimum (maximum x 0xc3000000#32) 0x42fe0000#32

theorem negative128_defined {x : FBits .f32}
    (sign : negative x = true) (word : magnitudeWord x ≤ 0x43000000) :
    ∃ byte, toByte x = some byte := by
  by_cases small : magnitudeWord x < 0x43000000
  · exact toByte_defined_of_small (by have he := exponent_small small; omega) (magnitude_small small)
  · have same : magnitudeWord x = 0x43000000 := by omega
    have he : exponent x = 134 := by simp [exponent, same]
    have hf : fraction x = 0 := by simp [fraction, same]
    have value : truncatedInt x = -128 := by
      simp [truncatedInt, sign, truncatedMagnitude, he, hf]
    exact ⟨BitVec.ofInt 8 (-128), by simp [toByte, he, value]⟩

theorem clamp_conversion_defined {x : FBits .f32} (nan : isNaN x = false) :
    ∃ byte, toByte (clamp x) = some byte := by
  have upperNaN : isNaN 0x42fe0000#32 = false := by decide
  have lowerNaN : isNaN 0xc3000000#32 = false := by decide
  have upperSign : negative 0x42fe0000#32 = false := by decide
  have lowerSign : negative 0xc3000000#32 = true := by decide
  have upperWord : magnitudeWord 0x42fe0000#32 = 0x42fe0000 := by decide
  have lowerWord : magnitudeWord 0xc3000000#32 = 0x43000000 := by decide
  cases sign : negative x with
  | true =>
    by_cases over : 0x43000000 < magnitudeWord x
    · refine ⟨128#8, ?_⟩
      simp only [clamp, maximum, nan, lowerNaN, precedes, sign, lowerSign, lowerWord,
        over, decide_true, Bool.false_eq_true, ↓reduceIte]
      rfl
    · have unchanged : clamp x = x := by
        simp [clamp, minimum, maximum, precedes, nan, upperNaN, lowerNaN,
          upperSign, lowerSign, lowerWord, sign, over]
      rw [unchanged]
      exact negative128_defined sign (by omega)
  | false =>
    by_cases less : magnitudeWord x < 0x42fe0000
    · have unchanged : clamp x = x := by
        simp [clamp, minimum, maximum, precedes, nan, upperNaN, lowerNaN,
          upperSign, lowerSign, upperWord, sign, less]
      rw [unchanged]
      have small : magnitudeWord x < 0x43000000 := by omega
      exact toByte_defined_of_small (by have he := exponent_small small; omega) (magnitude_small small)
    · refine ⟨127#8, ?_⟩
      simp only [clamp, maximum, minimum, nan, lowerNaN, upperNaN, precedes,
        sign, lowerSign, upperSign, upperWord, less, decide_false, Bool.false_eq_true, ↓reduceIte]
      rfl

theorem nan_conversion_poison {x : FBits .f32} (nan : isNaN x = true) :
    toByte (clamp x) = none := by
  have he : exponent x = 255 := (of_decide_eq_true nan).1
  simp [clamp, minimum, maximum, nan, toByte, he]

theorem conversion_domain (x : FBits .f32) :
    (toByte (clamp x)).isSome = true ↔ isNaN x = false := by
  cases nan : isNaN x with
  | false => obtain ⟨byte, converted⟩ := clamp_conversion_defined nan; simp [converted]
  | true => simp [nan_conversion_poison nan]

#print axioms clamp_conversion_defined
#print axioms conversion_domain

end MerlinVerify.Rules.W8A8SourceByte
