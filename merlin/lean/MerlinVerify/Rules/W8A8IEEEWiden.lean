/- Exact widening of positive normal binary32 scales, followed by the existing WideScale
interval. This is a proof of the declared conversion computation, not a host conversion. -/
import MerlinVerify.Rules.W8A8IEEENarrow

namespace MerlinVerify.Rules.W8A8IEEEWiden

open MerlinVerify Float.Model.UnpackedFloat
open Float.Model (Format UnpackedFloat totalExponent)

theorem normal_fields {x : FBits .f32} (normal : W8A8FiniteLaws.scaleCheck x = true) :
    ∃ m e positive, W8A8IEEE32.unpack x = .finite .positive m e positive ∧
      2^23 ≤ m ∧ m < 2^24 ∧ -149 ≤ e ∧ e ≤ 104 := by
  have parts := normal
  simp only [W8A8FiniteLaws.scaleCheck, W8A8SourceQuantize.scaleAdmissible,
    Bool.and_eq_true, decide_eq_true_eq] at parts
  obtain ⟨⟨⟨sign, nonzero⟩, finite⟩, exponent⟩ := parts
  have signFalse : W8A8ByteFloat.negative x = false := by
    cases signed : W8A8ByteFloat.negative x <;> simp_all
  have positiveSign := W8A8IEEE32.unpack_positiveSign
    (show W8A8IEEE32.Nonnegative x from ⟨W8A8IEEE32.finite_notNaN finite, signFalse⟩)
  have fieldExponent := W8A8IEEE32.exponent_eq_unpack x
  rw [fieldExponent] at exponent finite
  have ordinary : unpackExponent (spec := Format.binary32) x ≠ -1#8 := by
    intro exceptional
    have bad : (unpackExponent (spec := Format.binary32) x).toNat = 255 := by rw [exceptional]; rfl
    omega
  have notZero : unpackExponent (spec := Format.binary32) x ≠ 0#8 := by
    intro zero
    have bad : (unpackExponent (spec := Format.binary32) x).toNat = 0 := by rw [zero]; rfl
    omega
  let m := (1#1 ++ (unpackMantissa (spec := Format.binary32) x)).toNat
  let e := ((unpackExponent (spec := Format.binary32) x).toNat : Int) - 150
  have positive : 0 < m := by
    dsimp only [m]
    rw [W8A8IEEE32.append_toNat]
    change 0 < 1 * 2^23 + _
    omega
  have decoded : W8A8IEEE32.unpack x =
      .finite (Sign.ofBitVec (unpackSign (spec := Format.binary32) x)) m e positive := by
    simp only [W8A8IEEE32.unpack, UnpackedFloat.unpack, ordinary, notZero, ↓reduceIte]
    rfl
  rw [decoded] at positiveSign
  dsimp only [W8A8IEEE32.PositiveSign] at positiveSign
  rw [positiveSign] at decoded
  refine ⟨m, e, positive, decoded, ?_, ?_, ?_, ?_⟩
  · dsimp only [m]
    rw [W8A8IEEE32.append_toNat]
    change 8388608 ≤ 1 * 2^23 + _
    omega
  · dsimp only [m]
    rw [W8A8IEEE32.append_toNat]
    have fraction := (unpackMantissa (spec := Format.binary32) x).isLt
    change 1 * 2^23 + (unpackMantissa (spec := Format.binary32) x).toNat < 16777216
    change (unpackMantissa (spec := Format.binary32) x).toNat < 8388608 at fraction
    omega
  · dsimp only [e]; omega
  · dsimp only [e]; omega

theorem rounded_widen {m : Nat} {e : Int} (positive : 0 < m)
    (low : 2^23 ≤ m) (high : m < 2^24) (exponent : -149 ≤ e) :
    round Format.binary64 .positive m e =
      .finite .positive (m <<< 29) (e - 29) (Nat.shiftLeft_pos_iff.mpr positive) := by
  have log : m.log2 = 23 := (Nat.log2_eq_iff (Nat.ne_of_gt positive)).2 ⟨low, high⟩
  have target : Format.binary64.targetExponent (totalExponent m e) = e - 29 := by
    simp only [Format.targetExponent, totalExponent, log, Format.mantissaBits,
      Format.minExponent]
    omega
  have shifted := W8A8IEEERoundingBounds.log2_shift positive 29
  have logShift : (m <<< 29).log2 = 52 := by omega
  have fixed : Format.binary64.targetExponent (totalExponent (m <<< 29) (e - 29)) = e - 29 := by
    simp only [Format.targetExponent, totalExponent, logShift, Format.mantissaBits,
      Format.minExponent]
    omega
  have amount : (e - (e - 29)).toNat = 29 := by omega
  simp only [round, target, decreaseExponent, amount]
  exact W8A8IEEERoundingBounds.rounded_fixed _ _ (Nat.shiftLeft_pos_iff.mpr positive) fixed

theorem pack_positive64 (exponent : BitVec 11) (mantissa : BitVec 52) :
    (packComponents Format.binary64 .positive exponent mantissa).toNat =
      exponent.toNat * 2^52 + mantissa.toNat := by
  change (((0#1 ++ exponent) ++ mantissa) : BitVec 64).toNat = _
  rw [W8A8IEEE32.append_toNat, W8A8IEEE32.append_toNat]
  simp

theorem packed_widen_wide {m : Nat} {e : Int} (positive : 0 < m)
    (low : 2^23 ≤ m) (high : m < 2^24) (lower : -149 ≤ e) (upper : e ≤ 104) :
    W8A8FiniteLaws.WideScale (UnpackedFloat.pack Format.binary64
      (.finite .positive (m <<< 29) (e - 29) (Nat.shiftLeft_pos_iff.mpr positive))) := by
  have log : m.log2 = 23 := (Nat.log2_eq_iff (Nat.ne_of_gt positive)).2 ⟨low, high⟩
  have shifted := W8A8IEEERoundingBounds.log2_shift positive 29
  have logShift : (m <<< 29).log2 = 52 := by omega
  have biased : 897 ≤ (e - 29 + 1023 + 52).toNat ∧ (e - 29 + 1023 + 52).toNat ≤ 1150 := by omega
  have noOverflow : ¬ 2048 ≤ (e - 29 + 1023 + 52).toNat + 1 := by omega
  have noOverflow' : ¬ 2^Format.binary64.exponentBits ≤
      (e - 29 + Format.binary64.exponentBias + Format.binary64.mantissaBitsWithoutImplicit).toNat + 1 :=
    noOverflow
  have normalMantissa : (m <<< 29).log2 + 1 = Format.binary64.mantissaBits := by rw [logShift]; rfl
  have packed : UnpackedFloat.pack Format.binary64
      (.finite .positive (m <<< 29) (e - 29) (Nat.shiftLeft_pos_iff.mpr positive)) =
      packComponents Format.binary64 .positive
        (BitVec.ofNat 11 (e - 29 + 1023 + 52).toNat) (BitVec.ofNat 52 (m <<< 29)) := by
    simp only [UnpackedFloat.pack, ite_eq_right noOverflow', normalMantissa, ite_true]
    rfl
  change 0x3810000000000000 ≤ ((UnpackedFloat.pack Format.binary64
      (.finite .positive (m <<< 29) (e - 29) (Nat.shiftLeft_pos_iff.mpr positive))) : BitVec 64).toNat ∧
    ((UnpackedFloat.pack Format.binary64
      (.finite .positive (m <<< 29) (e - 29) (Nat.shiftLeft_pos_iff.mpr positive))) : BitVec 64).toNat <
        0x7ff0000000000000
  rw [packed, pack_positive64]
  simp only [BitVec.toNat_ofNat]
  have fraction := Nat.mod_lt (m <<< 29) (by decide : 0 < 2^52)
  omega

theorem widen_wide {x : FBits .f32} (normal : W8A8FiniteLaws.scaleCheck x = true) :
    W8A8FiniteLaws.WideScale (W8A8ArithmeticModel.widen x) := by
  obtain ⟨m, e, positive, decoded, low, high, lower, upper⟩ := normal_fields normal
  simp only [W8A8ArithmeticModel.widen, decoded, W8A8ArithmeticModel.rounded,
    rounded_widen positive low high lower]
  exact packed_widen_wide positive low high lower upper

#print axioms normal_fields
#print axioms rounded_widen
#print axioms widen_wide

end MerlinVerify.Rules.W8A8IEEEWiden
