/- Integer-unit magnitude bounds for the actual ordered binary32 computation.
This additive family preserves the earlier qualified checker and its assumptions. -/
import MerlinVerify.Rules.W8A8OrderedMagnitude
import MerlinVerify.Rules.W8A8IEEERoundingBounds

namespace MerlinVerify.Rules.W8A8OrderedMagnitude

open MerlinVerify Float.Model.UnpackedFloat W8A8IEEE32
open Float.Model (UnpackedFloat)

theorem truncation_units_le (m amount : Nat) {e : Int} (lower : -149 ≤ e) :
    (m / 2^amount) * 2^(e + amount + 149).toNat ≤ m * 2^(e + 149).toNat := by
  have exponents : (e + (amount : Int) + 149).toNat = amount + (e + 149).toNat := by omega
  rw [exponents, Nat.pow_add, ← Nat.mul_assoc]
  exact Nat.mul_le_mul_right _ (Nat.div_mul_le_self m (2^amount))

/-- Keep both actual shifts: the second truncates the rounded mantissa, so it cannot
increase magnitude. The sole upward error is at most one first-shift unit. -/
theorem rounded_units_le {s s' : Sign} {m m' : Nat} {e e' : Int} {a : Accuracy}
    {positive : 0 < m'} (lower : -149 ≤ e)
    (rounded : roundWithAccuracy format s m e a = .finite s' m' e' positive) :
    -149 ≤ e' ∧ m' * 2^(e' + 149).toNat ≤ m * 2^(e + 149).toNat +
      2^((shiftToTargetExponent format m e a).2 + 149).toNat := by
  let amount := (format.targetExponent (Float.Model.totalExponent m e) - e).toNat
  let first := shiftToTargetExponent format m e a
  let r := first.1.roundedMantissa
  let next := (format.targetExponent (Float.Model.totalExponent r first.2) - first.2).toNat
  let second := shiftToTargetExponent format r first.2 .exact
  have firstExponent : first.2 = e + amount := rfl
  have secondExponent : second.2 = first.2 + next := rfl
  have firstLower : -149 ≤ first.2 := by rw [firstExponent]; omega
  have secondLower : -149 ≤ second.2 := by rw [secondExponent]; omega
  have initial : (ExtendedMantissa.ofMantissaAndAccuracy m a).mantissa = m := by
    cases a with
    | exact => rfl
    | inexact order => cases order <;> rfl
  have firstMantissa : first.1.mantissa = m / 2^amount := by
    exact (W8A8IEEEBounds.shifted_mantissa _ _).trans (congrArg (· / 2^amount) initial)
  have secondMantissa : second.1.mantissa = r / 2^next :=
    W8A8IEEEBounds.shifted_mantissa (ExtendedMantissa.ofMantissaAndAccuracy r .exact) next
  have firstBound : first.1.mantissa * 2^(first.2 + 149).toNat ≤ m * 2^(e + 149).toNat := by
    rw [firstMantissa, firstExponent]
    exact truncation_units_le m amount lower
  have finalBound : second.1.mantissa * 2^(second.2 + 149).toNat ≤
      m * 2^(e + 149).toNat + 2^(first.2 + 149).toNat := by
    calc
      _ ≤ r * 2^(first.2 + 149).toNat := by
        rw [secondMantissa, secondExponent]
        exact truncation_units_le r next firstLower
      _ ≤ (first.1.mantissa + 1) * 2^(first.2 + 149).toNat :=
        Nat.mul_le_mul_right _ (W8A8IEEEBounds.roundedMantissa_le first.1)
      _ = first.1.mantissa * 2^(first.2 + 149).toNat + 2^(first.2 + 149).toNat := by
        rw [Nat.add_mul, Nat.one_mul]
      _ ≤ _ := Nat.add_le_add_right firstBound _
  change (if h : second.1.mantissa = 0 then UnpackedFloat.zero s
    else UnpackedFloat.finite s second.1.mantissa second.2 (Nat.pos_of_ne_zero h)) = _ at rounded
  split at rounded
  · cases rounded
  · cases rounded
    exact ⟨secondLower, finalBound⟩

/-- The preliminary left shift used by `round` preserves the input magnitude exactly.
Its first right-shift exponent is the original target, including subnormal inputs. -/
theorem round_units_le {s s' : Sign} {m m' : Nat} {e e' : Int}
    {positive' : 0 < m'} (positive : 0 < m) (lower : -149 ≤ e)
    (rounded : round format s m e = .finite s' m' e' positive') :
    -149 ≤ e' ∧ m' * 2^(e' + 149).toNat ≤ m * 2^(e + 149).toNat +
      2^(format.targetExponent (Float.Model.totalExponent m e) + 149).toNat := by
  let target := format.targetExponent (Float.Model.totalExponent m e)
  let amount := (e - target).toNat
  have targetLower : -149 ≤ target := by
    change -149 ≤ max ((m.log2 : Int) + 1 + e - (1 + 23)) (-149)
    omega
  have decreasedLower : -149 ≤ e - (amount : Int) := by dsimp only [amount]; omega
  have decreasedUpper : e - (amount : Int) ≤ target := by dsimp only [amount]; omega
  have logs := W8A8IEEERoundingBounds.log2_shift positive amount
  have firstExponent :
      (shiftToTargetExponent format (m <<< amount) (e - amount) .exact).2 = target := by
    rw [W8A8IEEERoundingBounds.shift_exponent, logs]
    change max (e - (amount : Int))
      (max ((m.log2 + amount : Nat) + (e - (amount : Int)) - 23) (-149)) = target
    have normalized : max ((m.log2 + amount : Nat) + (e - (amount : Int)) - 23) (-149) = target := by
      dsimp only [target, Float.Model.Format.targetExponent, Float.Model.totalExponent,
        Float.Model.Format.mantissaBits, Float.Model.Format.minExponent]
      omega
    rw [normalized]
    omega
  have exactUnits : (m <<< amount) * 2^(e - amount + 149).toNat = m * 2^(e + 149).toNat := by
    have exponent : amount + (e - (amount : Int) + 149).toNat = (e + 149).toNat := by omega
    rw [Nat.shiftLeft_eq, Nat.mul_assoc, ← Nat.pow_add, exponent]
  change roundWithAccuracy format s (m <<< amount) (e - amount) .exact = _ at rounded
  obtain ⟨finalLower, bound⟩ := rounded_units_le decreasedLower rounded
  rw [firstExponent, exactUnits] at bound
  exact ⟨finalLower, bound⟩

theorem target_le_of_units {m limit : Nat} {e : Int} (positive : 0 < m) (lower : -149 ≤ e)
    (bound : m * 2^(e + 149).toNat < 2^(limit + 24)) :
    format.targetExponent (Float.Model.totalExponent m e) ≤ (limit : Int) - 149 := by
  have inputPositive := Nat.mul_pos positive (Nat.two_pow_pos (e + 149).toNat)
  have logarithm := (Nat.log2_lt (Nat.ne_of_gt inputPositive)).2 bound
  have logs := W8A8IEEERoundingBounds.log2_shift positive (e + 149).toNat
  rw [Nat.shiftLeft_eq] at logs
  rw [logs] at logarithm
  have exponent : ((e + 149).toNat : Int) = e + 149 := by omega
  change max ((m.log2 : Int) + 1 + e - (1 + 23)) (-149) ≤ (limit : Int) - 149
  omega

/-- A finite packed round grows by at most one declared magnitude unit. This follows
the actual pinned rounding computations; it is not an assumed error model. -/
theorem round_bounded {s : Sign} {m limit : Nat} {e : Int} (positive : 0 < m)
    (lower : -149 ≤ e) (limitSafe : limit ≤ 252)
    (bound : m * 2^(e + 149).toNat < 2^(limit + 24)) :
    Finite (W8A8IEEE32.pack (round format s m e)) ∧
      units (W8A8IEEE32.pack (round format s m e)) ≤ m * 2^(e + 149).toNat + 2^limit := by
  have target := target_le_of_units positive lower bound
  rcases W8A8IEEERoundingBounds.round_zero_or_finite format s m e with
    zero | ⟨mantissa, exponent, positive', rounded⟩
  · rw [zero]
    exact ⟨(zero_units s).1, by rw [(zero_units s).2]; exact Nat.zero_le _⟩
  · rw [rounded]
    obtain ⟨resultLower, unitsBound⟩ := round_units_le positive lower rounded
    have exponentBound := W8A8IEEERoundingBounds.round_exponent positive rounded
    have resultUpper : exponent ≤ 104 := by omega
    have errorExponent : (format.targetExponent (Float.Model.totalExponent m e) + 149).toNat ≤ limit := by
      omega
    refine ⟨W8A8IEEEBounds.pack_finite_exponent resultUpper, ?_⟩
    exact Nat.le_trans (pack_units_le resultLower resultUpper) (Nat.le_trans unitsBound
      (Nat.add_le_add_left (Nat.pow_le_pow_right (by decide) errorExponent) _))

#print axioms rounded_units_le
#print axioms round_units_le
#print axioms round_bounded

end MerlinVerify.Rules.W8A8OrderedMagnitude
