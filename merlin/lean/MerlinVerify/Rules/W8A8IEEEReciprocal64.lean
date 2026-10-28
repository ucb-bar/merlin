/- The source binary64 reciprocal stays below the existing exact 2^126 endpoint. This
proves a range property of pinned binary64 division, not equality with binary32 division. -/
import MerlinVerify.Rules.W8A8IEEEDivision32
import MerlinVerify.Rules.W8A8IEEERoundingPower

namespace MerlinVerify.Rules.W8A8IEEEReciprocal64

open MerlinVerify Float.Model.UnpackedFloat
open Float.Model (Format UnpackedFloat totalExponent)

theorem pack_narrowable {m : Nat} {e : Int} (positive : 0 < m)
    (exponent : e ≤ 74) (endpoint : e = 74 → m ≤ 2^52) :
    W8A8FiniteLaws.Narrowable (UnpackedFloat.pack Format.binary64 (.finite .positive m e positive)) := by
  change ((UnpackedFloat.pack Format.binary64 (.finite .positive m e positive)) : BitVec 64).toNat ≤
    0x47d0000000000000
  have biased : (e + 1023 + 52).toNat ≤ 1149 := by omega
  have noOverflow : ¬ 2^Format.binary64.exponentBits ≤
      (e + Format.binary64.exponentBias + Format.binary64.mantissaBitsWithoutImplicit).toNat + 1 := by
    change ¬2048 ≤ (e + 1023 + 52).toNat + 1
    omega
  simp only [UnpackedFloat.pack, ite_eq_right noOverflow]
  split
  · next normal =>
    have leading : m.log2 = 52 := by change m.log2 + 1 = 53 at normal; omega
    rw [W8A8IEEEWiden.pack_positive64]
    by_cases top : e = 74
    · subst e
      have lower := Nat.log2_self_le (Nat.ne_of_gt positive)
      rw [leading] at lower
      have exactMantissa : m = 2^52 := by have bound := endpoint rfl; omega
      subst m
      decide
    · simp only [BitVec.toNat_ofNat]
      change ((e + 1023 + 52).toNat % 2048) * 2^52 + m % 2^52 ≤ 0x47d0000000000000
      have lowerExponent : (e + 1023 + 52).toNat ≤ 1148 := by omega
      have fraction := Nat.mod_lt m (by decide : 0 < 2^52)
      omega
  · rw [W8A8IEEEWiden.pack_positive64]
    have fraction := (BitVec.ofNat 52 m).isLt
    change 0 * 2^52 + (BitVec.ofNat 52 m).toNat ≤ 0x47d0000000000000
    change (BitVec.ofNat 52 m).toNat < 4503599627370496 at fraction
    omega

theorem rounded_narrowable (m : Nat) (e : Int) (a : Accuracy)
    (mantissa : m ≤ 2^53) (exponent : e ≤ 73) :
    W8A8FiniteLaws.Narrowable
      (UnpackedFloat.pack Format.binary64 (roundWithAccuracy Format.binary64 .positive m e a)) := by
  rcases W8A8IEEERoundingBounds.rounded_zero_or_finite Format.binary64 .positive m e a with
      rounded | ⟨m', e', positive, rounded⟩
  · rw [rounded]
    change (0 : Nat) ≤ 0x47d0000000000000
    decide
  · rw [rounded]
    have bounds := W8A8IEEERoundingPower.rounded_power_bound
      (spec := Format.binary64) (E := 74) mantissa (by omega) (by decide) rounded
    exact pack_narrowable positive bounds.1 bounds.2

theorem wide_fields {x : FBits .f64} (wide : W8A8FiniteLaws.WideScale x) :
    ∃ m e positive, UnpackedFloat.unpack Format.binary64 x = .finite .positive m e positive ∧
      2^52 ≤ m ∧ m < 2^53 ∧ -178 ≤ e ∧ e ≤ 971 := by
  have fields := BitVec.extractLsb'_toNat 52 11 x
  have signField := BitVec.extractLsb'_toNat 63 1 x
  simp only [Nat.shiftRight_eq_div_pow] at fields signField
  have range : 897 ≤ (unpackExponent (spec := Format.binary64) x).toNat ∧
      (unpackExponent (spec := Format.binary64) x).toNat ≤ 2046 := by
    change 897 ≤ (BitVec.extractLsb' 52 11 x).toNat ∧ (BitVec.extractLsb' 52 11 x).toNat ≤ 2046
    rw [fields]
    unfold W8A8FiniteLaws.WideScale at wide
    omega
  have signed : unpackSign (spec := Format.binary64) x = 0#1 := by
    apply BitVec.eq_of_toNat_eq
    change (BitVec.extractLsb' 63 1 x).toNat = 0
    rw [signField]
    unfold W8A8FiniteLaws.WideScale at wide
    omega
  have ordinary : unpackExponent (spec := Format.binary64) x ≠ -1#11 := by
    intro equal
    have bad : (unpackExponent (spec := Format.binary64) x).toNat = 2047 := by rw [equal]; rfl
    omega
  have notZero : unpackExponent (spec := Format.binary64) x ≠ 0#11 := by
    intro equal
    have bad : (unpackExponent (spec := Format.binary64) x).toNat = 0 := by rw [equal]; rfl
    omega
  let m := (1#1 ++ (unpackMantissa (spec := Format.binary64) x)).toNat
  let e := ((unpackExponent (spec := Format.binary64) x).toNat : Int) - 1075
  have positive : 0 < m := by
    dsimp only [m]
    rw [W8A8IEEE32.append_toNat]
    change 0 < 1 * 2^52 + _
    omega
  refine ⟨m, e, positive, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [UnpackedFloat.unpack, ordinary, notZero, ↓reduceIte, signed]
    rfl
  · dsimp only [m]
    rw [W8A8IEEE32.append_toNat]
    change 4503599627370496 ≤ 1 * 2^52 + _
    omega
  · dsimp only [m]
    rw [W8A8IEEE32.append_toNat]
    have fraction := (unpackMantissa (spec := Format.binary64) x).isLt
    change 1 * 2^52 + (unpackMantissa (spec := Format.binary64) x).toNat < 9007199254740992
    change (unpackMantissa (spec := Format.binary64) x).toNat < 4503599627370496 at fraction
    omega
  · dsimp only [e]; omega
  · dsimp only [e]; omega

theorem core_bounds {m : Nat} (e : Int) (positive : 0 < m)
    (low : 2^52 ≤ m) (high : m < 2^53) (lower : -178 ≤ e) :
    (divCore Format.binary64 (2^52) (-52) m e).1 ≤ 2^53 ∧
      (divCore Format.binary64 (2^52) (-52) m e).2.1 ≤ 73 := by
  have log : m.log2 = 52 := (Nat.log2_eq_iff (Nat.ne_of_gt positive)).2 ⟨low, high⟩
  have oneLog : (2^52 : Nat).log2 = 52 := by decide
  let target := min ((-52) - e)
    (Format.binary64.targetExponent (totalExponent (2^52) (-52) - totalExponent m e))
  let amount := ((-52) - e - target).toNat
  have targetLower : (-52) - e - 53 ≤ target := by
    dsimp only [target, Format.targetExponent, totalExponent, Format.mantissaBits, Format.minExponent]
    rw [log, oneLog]
    change (-52) - e - 53 ≤ min ((-52) - e) (max ((52 : Int) + 1 + (-52) - (52 + 1 + e) - 53) (-1074))
    omega
  have targetUpper : target ≤ (-52) - e := by dsimp only [target]; omega
  have amountBound : amount ≤ 53 := by dsimp only [amount]; omega
  have quotient : (2^52 * 2^amount) / m ≤ 2^amount := by
    have numerator : 2^52 * 2^amount ≤ 2^amount * m := by
      simpa [Nat.mul_comm] using Nat.mul_le_mul_left (2^amount) low
    calc
      (2^52 * 2^amount) / m ≤ (2^amount * m) / m := Nat.div_le_div_right numerator
      _ = 2^amount := Nat.mul_div_cancel _ positive
  constructor
  · change ((2^52) <<< amount) / m ≤ 2^53
    rw [Nat.shiftLeft_eq]
    exact Nat.le_trans quotient (Nat.pow_le_pow_right (by decide) amountBound)
  · change target ≤ 73
    dsimp only [target, Format.targetExponent, totalExponent, Format.mantissaBits, Format.minExponent]
    rw [log, oneLog]
    change min ((-52) - e) (max ((52 : Int) + 1 + (-52) - (52 + 1 + e) - 53) (-1074)) ≤ 73
    omega

theorem reciprocal_narrowable {divisor : FBits .f64} (wide : W8A8FiniteLaws.WideScale divisor) :
    W8A8FiniteLaws.Narrowable (W8A8ArithmeticModel.divide64 0x3ff0000000000000#64 divisor) := by
  obtain ⟨m, e, positive, decoded, low, high, lower, upper⟩ := wide_fields wide
  have one : UnpackedFloat.unpack Format.binary64 0x3ff0000000000000#64 =
      .finite .positive (2^52) (-52) (by decide) := rfl
  have bounds := core_bounds e positive low high lower
  let core := divCore Format.binary64 (2^52) (-52) m e
  simp only [W8A8ArithmeticModel.divide64, one, decoded, UnpackedFloat.div]
  change W8A8FiniteLaws.Narrowable
    (UnpackedFloat.pack Format.binary64 (roundWithAccuracy Format.binary64 .positive core.1 core.2.1 core.2.2))
  exact rounded_narrowable _ _ _ bounds.1 bounds.2

#print axioms rounded_narrowable
#print axioms reciprocal_narrowable

end MerlinVerify.Rules.W8A8IEEEReciprocal64
