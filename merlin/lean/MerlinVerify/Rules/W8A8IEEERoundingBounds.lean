/- Precision-parametric bounds for the pinned rounded arithmetic. These are proofs about
the actual shift/round/division computations, without new numerical-law hypotheses. -/
import MerlinVerify.Rules.W8A8IEEEBounds

namespace MerlinVerify.Rules.W8A8IEEERoundingBounds

open MerlinVerify Float.Model.UnpackedFloat
open Float.Model (Format UnpackedFloat totalExponent)

theorem shift_exponent (spec : Format) (m : Nat) (e : Int) (a : Accuracy) :
    (shiftToTargetExponent spec m e a).2 =
      max e (max ((m.log2 : Int) + e - spec.mantissaBitsWithoutImplicit) spec.minExponent) := by
  simp only [shiftToTargetExponent, shiftToExponent, Format.targetExponent,
    totalExponent, Format.mantissaBits]
  omega

theorem target_mantissa_lt (spec : Format) (m : Nat) (e : Int) (a : Accuracy) :
    (shiftToTargetExponent spec m e a).1.mantissa < 2^(spec.mantissaBitsWithoutImplicit + 1) := by
  have initial : (ExtendedMantissa.ofMantissaAndAccuracy m a).mantissa = m := by
    cases a with
    | exact => rfl
    | inexact cmp => cases cmp <;> rfl
  simp only [shiftToTargetExponent, shiftToExponent, W8A8IEEEBounds.shifted_mantissa, initial,
    Format.targetExponent, totalExponent, Format.mantissaBits]
  apply (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).2
  rw [← Nat.pow_add]
  have exponents : m.log2 + 1 ≤ spec.mantissaBitsWithoutImplicit + 1 +
      (max ((m.log2 : Int) + 1 + e - (1 + spec.mantissaBitsWithoutImplicit)) spec.minExponent - e).toNat := by omega
  exact Nat.lt_of_lt_of_le Nat.lt_log2_self (Nat.pow_le_pow_right (by decide) exponents)

theorem rounded_exponent {spec : Format} {s : Sign} {m : Nat} {e : Int} {a : Accuracy}
    {s' : Sign} {m' : Nat} {e' : Int} {positive : 0 < m'}
    (rounded : roundWithAccuracy spec s m e a = .finite s' m' e' positive) :
    e' ≤ max e (max ((m.log2 : Int) + e - spec.mantissaBitsWithoutImplicit) spec.minExponent) + 1 := by
  have mantissa := target_mantissa_lt spec m e a
  have first := shift_exponent spec m e a
  have upper := W8A8IEEEBounds.roundedMantissa_le (shiftToTargetExponent spec m e a).1
  have logBound : ((shiftToTargetExponent spec m e a).1.roundedMantissa).log2 ≤
      spec.mantissaBitsWithoutImplicit + 1 := by
    by_cases zero : (shiftToTargetExponent spec m e a).1.roundedMantissa = 0
    · simp [zero]
    · have powNext : 2^(spec.mantissaBitsWithoutImplicit + 1) < 2^(spec.mantissaBitsWithoutImplicit + 2) := by
        exact Nat.pow_lt_pow_right (by decide) (by omega)
      have bound := (Nat.log2_lt zero).2 (Nat.lt_of_le_of_lt (by omega) powNext)
      omega
  have second := shift_exponent spec (shiftToTargetExponent spec m e a).1.roundedMantissa
    (shiftToTargetExponent spec m e a).2 .exact
  unfold roundWithAccuracy at rounded
  dsimp only at rounded
  split at rounded
  · cases rounded
  · cases rounded
    omega

theorem division_mantissa_lt (spec : Format) {m₁ m₂ : Nat} (e₁ e₂ : Int)
    (positive₁ : 0 < m₁) (positive₂ : 0 < m₂)
    (mantissa₁ : m₁ < 2^(spec.mantissaBitsWithoutImplicit + 1)) :
    (divCore spec m₁ e₁ m₂ e₂).1 < 2^(spec.mantissaBitsWithoutImplicit + 2) := by
  let target := min (e₁ - e₂)
    (spec.targetExponent (totalExponent m₁ e₁ - totalExponent m₂ e₂))
  let amount := (e₁ - e₂ - target).toNat
  have log₁ : m₁.log2 ≤ spec.mantissaBitsWithoutImplicit := by
    have := (Nat.log2_lt (Nat.ne_of_gt positive₁)).2 mantissa₁
    omega
  have lower : (m₁.log2 : Int) - m₂.log2 + e₁ - e₂ -
      (spec.mantissaBitsWithoutImplicit + 1) ≤ target := by
    dsimp only [target, Format.targetExponent, totalExponent, Format.mantissaBits]
    omega
  have exponents : m₁.log2 + 1 + amount ≤ m₂.log2 + (spec.mantissaBitsWithoutImplicit + 2) := by
    have targetUpper : target ≤ e₁ - e₂ := by dsimp only [target]; omega
    dsimp only [amount]
    omega
  have numerator : m₁ * 2^amount < 2^m₂.log2 * 2^(spec.mantissaBitsWithoutImplicit + 2) := by
    have small := Nat.mul_lt_mul_of_pos_right (Nat.lt_log2_self (n := m₁)) (Nat.two_pow_pos amount)
    rw [← Nat.pow_add] at small ⊢
    exact Nat.lt_of_lt_of_le small (Nat.pow_le_pow_right (by decide) exponents)
  have lower₂ := Nat.log2_self_le (Nat.ne_of_gt positive₂)
  have upper := Nat.lt_of_lt_of_le numerator
    (Nat.mul_le_mul_right (2^(spec.mantissaBitsWithoutImplicit + 2)) lower₂)
  change (m₁ <<< amount) / m₂ < _
  rw [Nat.shiftLeft_eq]
  apply (Nat.div_lt_iff_lt_mul positive₂).2
  simpa [Nat.mul_comm] using upper

theorem log2_shift {m : Nat} (positive : 0 < m) (amount : Nat) :
    (m <<< amount).log2 = m.log2 + amount := by
  rw [Nat.shiftLeft_eq]
  have powPositive := Nat.two_pow_pos amount
  apply (Nat.log2_eq_iff (Nat.ne_of_gt (Nat.mul_pos positive powPositive))).2
  constructor
  · rw [Nat.pow_add]
    exact Nat.mul_le_mul_right _ (Nat.log2_self_le (Nat.ne_of_gt positive))
  · have bound := Nat.mul_lt_mul_of_pos_right (Nat.lt_log2_self (n := m)) powPositive
    convert bound using 1; simp [Nat.pow_add, Nat.mul_comm, Nat.mul_left_comm]

theorem round_exponent {spec : Format} {s : Sign} {m : Nat} {e : Int}
    {s' : Sign} {m' : Nat} {e' : Int} {positive' : 0 < m'}
    (positive : 0 < m)
    (rounded : round spec s m e = .finite s' m' e' positive') :
    e' ≤ spec.targetExponent (totalExponent m e) + 1 := by
  let target := spec.targetExponent (totalExponent m e)
  let amount := (e - target).toNat
  have logs := log2_shift positive amount
  change roundWithAccuracy spec s (m <<< amount) (e - amount) .exact = _ at rounded
  have bound := rounded_exponent rounded
  rw [logs] at bound
  have decreased : e - (amount : Int) ≤ target := by dsimp only [amount]; omega
  dsimp only [target, Format.targetExponent, totalExponent, Format.mantissaBits] at *
  omega

theorem rounded_zero_or_finite (spec : Format) (s : Sign) (m : Nat) (e : Int) (a : Accuracy) :
    roundWithAccuracy spec s m e a = .zero s ∨
      ∃ m' e' positive, roundWithAccuracy spec s m e a = .finite s m' e' positive := by
  unfold roundWithAccuracy
  dsimp only
  split
  · exact Or.inl rfl
  · next nonzero => exact Or.inr ⟨_, _, Nat.pos_of_ne_zero nonzero, rfl⟩

theorem round_zero_or_finite (spec : Format) (s : Sign) (m : Nat) (e : Int) :
    round spec s m e = .zero s ∨
      ∃ m' e' positive, round spec s m e = .finite s m' e' positive := by
  exact rounded_zero_or_finite _ _ _ _ _

theorem rounded_fixed (spec : Format) (s : Sign) {m : Nat} {e : Int} (positive : 0 < m)
    (target : spec.targetExponent (totalExponent m e) = e) :
    roundWithAccuracy spec s m e .exact = .finite s m e positive := by
  simp [roundWithAccuracy, shiftToTargetExponent, target, shiftToExponent,
    ExtendedMantissa.ofMantissaAndAccuracy, HShiftRight.hShiftRight, Nat.repeat,
    ExtendedMantissa.roundedMantissa, ExtendedMantissa.accuracy,
    Accuracy.roundToNearestEven, Nat.ne_of_gt positive]

theorem round_fixed (spec : Format) (s : Sign) {m : Nat} {e : Int} (positive : 0 < m)
    (target : spec.targetExponent (totalExponent m e) = e) :
    round spec s m e = .finite s m e positive := by
  simp only [round, target, decreaseExponent, Int.sub_self, Int.toNat_zero,
    Nat.shiftLeft_zero]
  simpa using rounded_fixed spec s positive target

#print axioms rounded_exponent
#print axioms division_mantissa_lt
#print axioms round_exponent

end MerlinVerify.Rules.W8A8IEEERoundingBounds
