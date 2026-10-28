/- Signed integer-unit interpretation of finite binary32 words and the sign of actual
rounding/packing. These bridge the existing concrete arithmetic to real-order properties. -/
import MerlinVerify.Rules.W8A8RealValue
import MerlinVerify.Rules.W8A8OrderedSum

namespace MerlinVerify.Rules.W8A8RealSign

open MerlinVerify W8A8RoundedReal W8A8OrderedMagnitude
open Float.Model (UnpackedFloat)
open Float.Model.UnpackedFloat

def signedUnits (bits : FBits .f32) : Int :=
  if W8A8ByteFloat.negative bits then -(units bits : Int) else (units bits : Int)

theorem value_signed_units (bits : FBits .f32) :
    value bits = (signedUnits bits : ℝ) / (2:ℝ)^149 := by
  unfold value signedUnits magnitude
  cases W8A8ByteFloat.negative bits <;> simp [neg_div]

theorem sign_of_bits (bits : FBits .f32) :
    Sign.ofBitVec (unpackSign (spec := W8A8IEEE32.format) bits) =
      if W8A8ByteFloat.negative bits then .negative else .positive := by
  have bound : bits.toNat < 2^32 := bits.isLt
  cases signed : W8A8ByteFloat.negative bits with
  | false =>
    have raw : ¬ 2^31 ≤ bits.toNat := of_decide_eq_false signed
    have zero : unpackSign (spec := W8A8IEEE32.format) bits = 0#1 := by
      apply BitVec.eq_of_toNat_eq
      change (BitVec.extractLsb' 31 1 bits).toNat = 0
      rw [BitVec.extractLsb'_toNat, Nat.shiftRight_eq_div_pow]
      omega
    rw [zero]
    rfl
  | true =>
    have raw : 2^31 ≤ bits.toNat := of_decide_eq_true signed
    have one : unpackSign (spec := W8A8IEEE32.format) bits = 1#1 := by
      apply BitVec.eq_of_toNat_eq
      change (BitVec.extractLsb' 31 1 bits).toNat = 1
      rw [BitVec.extractLsb'_toNat, Nat.shiftRight_eq_div_pow]
      omega
    rw [one]
    rfl

theorem unpack_finite_sign {bits : FBits .f32} {sign : Sign} {mantissa : Nat} {exponent : Int}
    {positive : 0 < mantissa} (decoded : W8A8IEEE32.unpack bits = .finite sign mantissa exponent positive) :
    sign = Sign.ofBitVec (unpackSign (spec := W8A8IEEE32.format) bits) := by
  unfold W8A8IEEE32.unpack UnpackedFloat.unpack at decoded
  dsimp only at decoded
  split at decoded
  · split at decoded <;> cases decoded
  · split at decoded
    · split at decoded
      · cases decoded
      · cases decoded; rfl
    · cases decoded; rfl

theorem signed_units_unpack {bits : FBits .f32} {sign : Sign} {mantissa : Nat} {exponent : Int}
    {positive : 0 < mantissa} (decoded : W8A8IEEE32.unpack bits = .finite sign mantissa exponent positive) :
    signedUnits bits = sign.apply ((mantissa * 2^(exponent + 149).toNat : Nat) : Int) := by
  have exactUnits := (unpack_units decoded).2
  rw [unpack_finite_sign decoded, sign_of_bits]
  unfold signedUnits
  rw [exactUnits]
  cases W8A8ByteFloat.negative bits <;> rfl

theorem units_zero_unpack {bits : FBits .f32} {sign : Sign}
    (decoded : W8A8IEEE32.unpack bits = .zero sign) : units bits = 0 := by
  have zero : W8A8ByteFloat.magnitudeWord bits = 0 := by
    by_contra nonzero
    exact W8A8IEEE32.unpack_notZero nonzero sign decoded
  simp [units, W8A8ByteFloat.exponent, W8A8ByteFloat.fraction, zero]

theorem value_zero_unpack {bits : FBits .f32} {sign : Sign}
    (decoded : W8A8IEEE32.unpack bits = .zero sign) : value bits = 0 :=
  units_zero_value (units_zero_unpack decoded)

def NegativeSign : UnpackedFloat → Prop
  | .notANumber => False
  | .infinity sign | .zero sign | .finite sign _ _ _ => sign = .negative

theorem packComponents_negative (exponent : BitVec 8) (mantissa : BitVec 23) :
    W8A8ByteFloat.negative (packComponents W8A8IEEE32.format .negative exponent mantissa) = true := by
  change decide (2^31 ≤ ((1#1 ++ exponent) ++ mantissa).toNat) = true
  rw [W8A8IEEE32.append_toNat, W8A8IEEE32.append_toNat]
  simp only [BitVec.toNat_ofNat, decide_eq_true_eq]
  omega

theorem pack_negative {unpacked : UnpackedFloat} (negative : NegativeSign unpacked) :
    W8A8ByteFloat.negative (W8A8IEEE32.pack unpacked) = true := by
  cases unpacked with
  | notANumber => cases negative
  | infinity sign => cases negative; exact packComponents_negative _ _
  | zero sign => cases negative; exact packComponents_negative _ _
  | finite sign mantissa exponent positive =>
    cases negative
    simp only [W8A8IEEE32.pack, UnpackedFloat.pack]
    split
    · exact packComponents_negative _ _
    · split <;> exact packComponents_negative _ _

theorem negative_value {bits : FBits .f32} (negative : W8A8ByteFloat.negative bits = true) : value bits ≤ 0 := by
  rw [value, negative]
  exact neg_nonpos.mpr (magnitude_nonneg bits)

theorem pack_negative_value {unpacked : UnpackedFloat} (negative : NegativeSign unpacked) :
    value (W8A8IEEE32.pack unpacked) ≤ 0 := negative_value (pack_negative negative)

theorem round_negative (mantissa : Nat) (exponent : Int) (accuracy : Accuracy) :
    NegativeSign (roundWithAccuracy W8A8IEEE32.format .negative mantissa exponent accuracy) := by
  unfold roundWithAccuracy
  dsimp only
  split <;> rfl

theorem rounded_negative_value (mantissa : Nat) (exponent : Int) :
    value (W8A8IEEE32.pack (UnpackedFloat.round W8A8IEEE32.format .negative mantissa exponent)) ≤ 0 := by
  apply pack_negative_value
  unfold UnpackedFloat.round
  exact round_negative _ _ _

theorem normalize_nonpositive {mantissa : Int} (exponent : Int) (nonpositive : mantissa ≤ 0) :
    value (W8A8IEEE32.pack (UnpackedFloat.normalize W8A8IEEE32.format mantissa exponent .positive)) ≤ 0 := by
  unfold UnpackedFloat.normalize
  split
  · exact rounded_negative_value _ _
  · rw [units_zero_value (zero_units .positive).2]
  · next positive =>
    have strictly : 0 < mantissa := by simpa only [Int.compare_eq_gt] using positive
    omega

theorem apply_mul (sign : Sign) (left right : Nat) :
    sign.apply ((left * right : Nat) : Int) = sign.apply (left : Int) * (right : Int) := by
  cases sign <;> simp [Sign.apply, Nat.cast_mul]

theorem signed_units_le_of_value_le {left right : FBits .f32} (ordered : value left ≤ value right) :
    signedUnits left ≤ signedUnits right := by
  have denominator : (0:ℝ) < (2:ℝ)^149 := by positivity
  have multiplied := mul_le_mul_of_nonneg_right ordered (le_of_lt denominator)
  rw [value_signed_units, value_signed_units,
    div_mul_cancel₀ _ (ne_of_gt denominator), div_mul_cancel₀ _ (ne_of_gt denominator)] at multiplied
  exact_mod_cast multiplied

theorem unpack_nonpositive_sign {bits : FBits .f32} {sign : Sign} {mantissa : Nat} {exponent : Int}
    {positive : 0 < mantissa} (decoded : W8A8IEEE32.unpack bits = .finite sign mantissa exponent positive)
    (nonpositive : value bits ≤ 0) : sign = .negative := by
  have ordered : value bits ≤ value (0 : FBits .f32) := by rw [value_positive_zero]; exact nonpositive
  have unitsOrder := signed_units_le_of_value_le ordered
  rw [signed_units_unpack decoded, show signedUnits (0 : FBits .f32) = 0 by decide] at unitsOrder
  have magnitude : 0 < mantissa * 2^(exponent + 149).toNat := Nat.mul_pos positive (Nat.two_pow_pos _)
  have magnitudeInt : (0:Int) < ((mantissa * 2^(exponent + 149).toNat : Nat) : Int) := by exact_mod_cast magnitude
  cases sign with
  | negative => rfl
  | positive => simp only [Sign.apply] at unitsOrder; omega

theorem unpack_nonnegative_sign {bits : FBits .f32} {sign : Sign} {mantissa : Nat} {exponent : Int}
    {positive : 0 < mantissa} (decoded : W8A8IEEE32.unpack bits = .finite sign mantissa exponent positive)
    (nonnegative : 0 ≤ value bits) : sign = .positive := by
  have ordered : value (0 : FBits .f32) ≤ value bits := by rw [value_positive_zero]; exact nonnegative
  have unitsOrder := signed_units_le_of_value_le ordered
  rw [signed_units_unpack decoded, show signedUnits (0 : FBits .f32) = 0 by decide] at unitsOrder
  have magnitude : 0 < mantissa * 2^(exponent + 149).toNat := Nat.mul_pos positive (Nat.two_pow_pos _)
  have magnitudeInt : (0:Int) < ((mantissa * 2^(exponent + 149).toNat : Nat) : Int) := by exact_mod_cast magnitude
  cases sign with
  | negative => simp only [Sign.apply] at unitsOrder; omega
  | positive => rfl

theorem aligned_signed_units {bits : FBits .f32} {sign : Sign} {mantissa : Nat} {exponent common : Int}
    {positive : 0 < mantissa} (decoded : W8A8IEEE32.unpack bits = .finite sign mantissa exponent positive)
    (lower : -149 ≤ common) (aligned : common ≤ exponent) :
    sign.apply ((mantissa <<< (exponent - common).toNat : Nat) : Int) *
      ((2^(common + 149).toNat : Nat) : Int) = signedUnits bits := by
  rw [← apply_mul, aligned_units mantissa lower aligned, signed_units_unpack decoded]

end MerlinVerify.Rules.W8A8RealSign
