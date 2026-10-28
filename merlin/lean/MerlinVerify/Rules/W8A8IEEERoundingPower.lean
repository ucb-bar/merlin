/- Rounding below an exactly representable power of two cannot cross that power. The
mantissa proof keeps residual round/sticky bits explicit at the upper endpoint. -/
import MerlinVerify.Rules.W8A8IEEERoundingBounds

namespace MerlinVerify.Rules.W8A8IEEERoundingPower

open MerlinVerify Float.Model.UnpackedFloat
open Float.Model (Format UnpackedFloat totalExponent)

theorem rounded_eq_of_roundBit (em : ExtendedMantissa) (bit : em.roundBit = false) :
    em.roundedMantissa = em.mantissa := by
  rcases em with ⟨m, r, s⟩
  cases r <;> cases s <;>
    simp_all [ExtendedMantissa.roundedMantissa, ExtendedMantissa.accuracy, Accuracy.roundToNearestEven]

theorem shifted_roundBit (em : ExtendedMantissa) (amount : Nat) :
    (em >>> (amount + 1)).roundBit = ((em >>> amount).mantissa % 2 != 0) := by
  rfl

theorem shifted_rounded_le_power (em : ExtendedMantissa) {p amount : Nat} (positive : 0 < amount)
    (bound : em.mantissa ≤ 2^(p + amount)) : (em >>> amount).roundedMantissa ≤ 2^p := by
  rcases Nat.eq_or_lt_of_le bound with endpoint | below
  · cases amount with
    | zero => omega
    | succ amount =>
      have previous : (em >>> amount).mantissa = 2^(p + 1) := by
        rw [W8A8IEEEBounds.shifted_mantissa, endpoint,
          show p + (amount + 1) = (p + 1) + amount from by omega,
          Nat.pow_add, Nat.mul_div_cancel _ (Nat.two_pow_pos _)]
      have bit : (em >>> (amount + 1)).roundBit = false := by
        rw [shifted_roundBit, previous, Nat.pow_succ]
        simp
      rw [rounded_eq_of_roundBit _ bit, W8A8IEEEBounds.shifted_mantissa, endpoint,
        Nat.pow_add, Nat.mul_div_cancel _ (Nat.two_pow_pos _)]
      exact Nat.le_refl _
  · have mantissa : (em >>> amount).mantissa < 2^p := by
      rw [W8A8IEEEBounds.shifted_mantissa]
      apply (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).2
      simpa [Nat.pow_add] using below
    have rounded := W8A8IEEEBounds.roundedMantissa_le (em >>> amount)
    omega

theorem quotient_le_power {m p amount : Nat} (positive : 0 < amount) (bound : m ≤ 2^(p+1)) :
    m / 2^amount ≤ 2^p := by
  have upper : m ≤ 2^p * 2^amount := by
    rw [← Nat.pow_add]
    exact Nat.le_trans bound (Nat.pow_le_pow_right (by decide) (by omega))
  calc
    m / 2^amount ≤ (2^p * 2^amount) / 2^amount := Nat.div_le_div_right upper
    _ = 2^p := Nat.mul_div_cancel _ (Nat.two_pow_pos _)

theorem log2_le_power {m p : Nat} (bound : m ≤ 2^p) : m.log2 ≤ p := by
  by_cases zero : m = 0
  · simp [zero]
  · have next : 2^p < 2^(p+1) := Nat.pow_lt_pow_right (by decide) (by omega)
    have logarithm := (Nat.log2_lt zero).2 (Nat.lt_of_le_of_lt bound next)
    omega

/-- At the largest possible output exponent the final mantissa cannot exceed the
implicit leading bit. This retains the endpoint information lost by an exponent-only cap. -/
theorem rounded_power_bound {spec : Format} {s s' : Sign} {m m' : Nat} {e e' E : Int}
    {a : Accuracy} {positive : 0 < m'}
    (mantissa : m ≤ 2^(spec.mantissaBitsWithoutImplicit + 1))
    (exponent : e ≤ E - 1) (minimum : spec.minExponent ≤ E - 1)
    (rounded : roundWithAccuracy spec s m e a = .finite s' m' e' positive) :
    e' ≤ E ∧ (e' = E → m' ≤ 2^spec.mantissaBitsWithoutImplicit) := by
  let p := spec.mantissaBitsWithoutImplicit
  let amount := (spec.targetExponent (totalExponent m e) - e).toNat
  let first := shiftToTargetExponent spec m e a
  let r := first.1.roundedMantissa
  let nextAmount := (spec.targetExponent (totalExponent r first.2) - first.2).toNat
  let second := shiftToTargetExponent spec r first.2 .exact
  have initialMantissa : (ExtendedMantissa.ofMantissaAndAccuracy m a).mantissa = m := by
    cases a with
    | exact => rfl
    | inexact cmp => cases cmp <;> rfl
  have firstEquation : first.2 = e + amount := rfl
  have secondEquation : second.2 = first.2 + nextAmount := rfl
  have secondMantissa : second.1.mantissa = r / 2^nextAmount :=
    W8A8IEEEBounds.shifted_mantissa (ExtendedMantissa.ofMantissaAndAccuracy r .exact) nextAmount
  have firstExponent := W8A8IEEERoundingBounds.shift_exponent spec m e a
  change first.2 = _ at firstExponent
  have secondExponent := W8A8IEEERoundingBounds.shift_exponent spec r first.2 .exact
  change second.2 = _ at secondExponent
  have inputLog := log2_le_power mantissa
  have firstLower : spec.minExponent ≤ first.2 := by rw [firstExponent]; omega
  have firstUpper : first.2 ≤ E := by rw [firstExponent]; omega
  have roundedMax : r ≤ 2^(p+1) := by
    have small := W8A8IEEERoundingBounds.target_mantissa_lt spec m e a
    have bound := W8A8IEEEBounds.roundedMantissa_le first.1
    change first.1.mantissa < 2^(p+1) at small
    change r ≤ first.1.mantissa + 1 at bound
    omega
  have roundedLog := log2_le_power roundedMax
  have result : second.2 ≤ E ∧ (second.2 = E → second.1.mantissa ≤ 2^p) := by
    by_cases zero : amount = 0
    · have equal : first.2 = e := by rw [firstEquation, zero]; simp
      have finalUpper : second.2 ≤ e + 1 := by rw [secondExponent]; dsimp only [p] at *; omega
      refine ⟨by omega, ?_⟩
      intro endpoint
      have positiveAmount : 0 < nextAmount := by omega
      rw [secondMantissa]
      exact quotient_le_power positiveAmount roundedMax
    · have positiveAmount : 0 < amount := Nat.pos_of_ne_zero zero
      have larger : m ≤ 2^(p + amount) :=
        Nat.le_trans mantissa (Nat.pow_le_pow_right (by decide) (by omega))
      have small : r ≤ 2^p := shifted_rounded_le_power
        (ExtendedMantissa.ofMantissaAndAccuracy m a) positiveAmount
        (by rw [initialMantissa]; exact larger)
      have logSmall := log2_le_power small
      have finalUpper : second.2 ≤ first.2 := by rw [secondExponent]; dsimp only [p] at *; omega
      refine ⟨by omega, fun _ => ?_⟩
      rw [secondMantissa]
      exact Nat.le_trans (Nat.div_le_self _ _) small
  change (if h : second.1.mantissa = 0 then UnpackedFloat.zero s
    else UnpackedFloat.finite s second.1.mantissa second.2 (Nat.pos_of_ne_zero h)) = _ at rounded
  split at rounded
  · cases rounded
  · cases rounded
    exact result

#print axioms shifted_rounded_le_power
#print axioms rounded_power_bound

end MerlinVerify.Rules.W8A8IEEERoundingPower
