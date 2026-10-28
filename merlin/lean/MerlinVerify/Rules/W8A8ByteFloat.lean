/- Concrete binary32 byte-conversion candidate for numerical qualification. All computations
use integer fields; no host float primitive or new float law is used. This does not install a
new frozen FloatModel or approve a source quantization specification. -/
import MerlinVerify.Rules.W8A8QuantizeDomain

namespace MerlinVerify.Rules.W8A8ByteFloat

open MerlinVerify

def magnitudeWord (x : FBits .f32) : Nat := x.toNat % 2^31
def negative (x : FBits .f32) : Bool := decide (2^31 ≤ x.toNat)
def exponent (x : FBits .f32) : Nat := magnitudeWord x / 2^23
def fraction (x : FBits .f32) : Nat := magnitudeWord x % 2^23
def isNaN (x : FBits .f32) : Bool := decide (exponent x = 255 ∧ fraction x ≠ 0)

/-- Absolute integer part, rounded toward zero. Subnormals have magnitude below one.
Nonfinite encodings are rejected by `toByte` before this value is used. -/
def truncatedMagnitude (x : FBits .f32) : Nat :=
  if exponent x = 0 then 0
  else if 150 ≤ exponent x then (2^23 + fraction x) * 2^(exponent x - 150)
  else (2^23 + fraction x) / 2^(150 - exponent x)

def truncatedInt (x : FBits .f32) : Int :=
  if negative x then -(Int.ofNat (truncatedMagnitude x)) else Int.ofNat (truncatedMagnitude x)

/-- Signed-byte conversion with the LLVM round-toward-zero/out-of-range contract. -/
def toByte (x : FBits .f32) : Option (BitVec 8) :=
  if exponent x = 255 then none
  else if -128 ≤ truncatedInt x ∧ truncatedInt x < 128 then some (BitVec.ofInt 8 (truncatedInt x))
  else none

/-- Symmetric clipping to -127/127, preserving signed zero and leaving NaNs nonconvertible. -/
def clamp127 (x : FBits .f32) : FBits .f32 :=
  if isNaN x then x
  else if 0x42fe0000 < magnitudeWord x then
    if negative x then 0xc2fe0000#32 else 0x42fe0000#32
  else x

/-- Numerical order on non-NaN encodings, with -0 before +0 for IEEE min/max. -/
def precedes (x y : FBits .f32) : Bool :=
  if negative x then
    if negative y then decide (magnitudeWord y < magnitudeWord x) else true
  else if negative y then false else decide (magnitudeWord x < magnitudeWord y)

/-- NaNs propagate. Payload selection is deterministic here; numerical qualification may
compare arithmetic NaNs by classification, since payload choice is not the byte-conversion claim. -/
def minimum (x y : FBits .f32) : FBits .f32 :=
  if isNaN x then x else if isNaN y then y else if precedes x y then x else y

def maximum (x y : FBits .f32) : FBits .f32 :=
  if isNaN x then x else if isNaN y then y else if precedes x y then y else x

private theorem positive_bound_eq {x : FBits .f32}
    (sign : negative x = false) (word : magnitudeWord x = 0x42fe0000) : x = 0x42fe0000#32 := by
  apply BitVec.eq_of_toNat_eq
  have positive : x.toNat < 2^31 := by
    have notNegative : ¬ 2^31 ≤ x.toNat := of_decide_eq_false sign
    omega
  change x.toNat = 0x42fe0000
  simpa only [magnitudeWord, Nat.mod_eq_of_lt positive] using word

theorem min_max_clamp {x : FBits .f32} (nan : isNaN x = false) :
    maximum (minimum x 0x42fe0000#32) 0xc2fe0000#32 = clamp127 x := by
  have upperNaN : isNaN 0x42fe0000#32 = false := by decide
  have lowerNaN : isNaN 0xc2fe0000#32 = false := by decide
  have upperSign : negative 0x42fe0000#32 = false := by decide
  have lowerSign : negative 0xc2fe0000#32 = true := by decide
  have upperWord : magnitudeWord 0x42fe0000#32 = 0x42fe0000 := by decide
  have lowerWord : magnitudeWord 0xc2fe0000#32 = 0x42fe0000 := by decide
  cases sign : negative x with
  | true =>
    simp [minimum, maximum, precedes, clamp127, nan, upperNaN, lowerNaN,
      upperSign, lowerSign, lowerWord, sign]
  | false =>
    by_cases less : magnitudeWord x < 0x42fe0000
    · have notGreater : ¬ 0x42fe0000 < magnitudeWord x := by omega
      simp [minimum, maximum, precedes, clamp127, nan, upperNaN, lowerNaN,
        upperSign, lowerSign, upperWord, sign, less, notGreater]
    · by_cases same : magnitudeWord x = 0x42fe0000
      · have equal := positive_bound_eq sign same
        subst x
        rfl
      · have greater : 0x42fe0000 < magnitudeWord x := by omega
        have order : precedes x 0x42fe0000#32 = false := by
          simp [precedes, sign, upperSign, upperWord, less]
        have clipped : minimum x 0x42fe0000#32 = 0x42fe0000#32 := by
          simp only [minimum, nan, upperNaN, order, Bool.false_eq_true, ↓reduceIte]
          rfl
        have clamped : clamp127 x = 0x42fe0000#32 := by
          simp only [clamp127, nan, greater, sign, Bool.false_eq_true, ↓reduceIte]
          rfl
        rw [clipped, clamped]
        rfl

theorem exponent_small {x : FBits .f32} (h : magnitudeWord x < 0x43000000) : exponent x < 134 := by
  unfold exponent
  omega

theorem magnitude_small {x : FBits .f32} (h : magnitudeWord x < 0x43000000) :
    truncatedMagnitude x ≤ 127 := by
  have he := exponent_small h
  have hf : fraction x < 2^23 := Nat.mod_lt _ (by decide)
  unfold truncatedMagnitude
  split
  · omega
  · split
    · omega
    · have denominator : 2^17 ≤ 2^(150 - exponent x) :=
        Nat.pow_le_pow_right (by decide) (by omega)
      calc
        (2^23 + fraction x) / 2^(150 - exponent x) ≤ (2^24 - 1) / 2^17 :=
          Nat.div_le_div (by omega) denominator (by decide)
        _ = 127 := by decide

theorem clamp_magnitude {x : FBits .f32} (h : isNaN x = false) :
    magnitudeWord (clamp127 x) ≤ 0x42fe0000 := by
  simp only [clamp127, h, Bool.false_eq_true, ↓reduceIte]
  split
  · cases negative x <;> decide
  · omega

theorem toByte_defined_of_small {x : FBits .f32}
    (finite : exponent x < 255) (bounded : truncatedMagnitude x ≤ 127) :
    ∃ byte, toByte x = some byte := by
  have fits : -128 ≤ truncatedInt x ∧ truncatedInt x < 128 := by
    unfold truncatedInt
    simp only [Int.ofNat_eq_natCast]
    split <;> omega
  exact ⟨BitVec.ofInt 8 (truncatedInt x), by simp [toByte, show exponent x ≠ 255 by omega, fits]⟩

/-- This includes infinities: clipping makes them finite. Only NaNs remain outside the
conversion domain. No premise about an unrestricted FloatModel has been silently added. -/
theorem clamp_conversion_defined {x : FBits .f32} (h : isNaN x = false) :
    ∃ byte, toByte (clamp127 x) = some byte := by
  have word : magnitudeWord (clamp127 x) < 0x43000000 := by
    have bound := clamp_magnitude h
    omega
  exact toByte_defined_of_small (by have he := exponent_small word; omega) (magnitude_small word)

theorem nan_conversion_poison {x : FBits .f32} (h : isNaN x = true) :
    toByte (clamp127 x) = none := by
  have exceptional : exponent x = 255 := (of_decide_eq_true h).1
  simp [clamp127, h, toByte, exceptional]

theorem clamp_conversion_domain (x : FBits .f32) :
    (toByte (clamp127 x)).isSome = true ↔ isNaN x = false := by
  cases classification : isNaN x with
  | false =>
    obtain ⟨byte, converted⟩ := clamp_conversion_defined classification
    simp [converted]
  | true => simp [nan_conversion_poison classification]

#print axioms magnitude_small
#print axioms min_max_clamp
#print axioms clamp_conversion_defined
#print axioms clamp_conversion_domain

end MerlinVerify.Rules.W8A8ByteFloat
