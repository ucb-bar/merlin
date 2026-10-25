/- Range of the actual binary64-to-binary32 rounded conversion. The admitted bit interval
is precisely the existing Narrowable premise, including subnormals and positive zero. -/
import MerlinVerify.Rules.W8A8IEEEInteger

namespace MerlinVerify.Rules.W8A8IEEENarrow

open MerlinVerify Float.Model.UnpackedFloat
open Float.Model (Format UnpackedFloat)

theorem unpack_exponent_bound {x : FBits .f64} (bound : W8A8FiniteLaws.Narrowable x) :
    (unpackExponent (spec := Format.binary64) x).toNat ≤ 1149 := by
  have extract := BitVec.extractLsb'_toNat 52 11 x
  simp only [Nat.shiftRight_eq_div_pow] at extract
  change (BitVec.extractLsb' 52 11 x).toNat ≤ 1149
  rw [extract]
  unfold W8A8FiniteLaws.Narrowable at bound
  omega

theorem unpack_cases {x : FBits .f64} (bound : W8A8FiniteLaws.Narrowable x) :
    (∃ s, UnpackedFloat.unpack Format.binary64 x = .zero s) ∨
      ∃ s m e positive, UnpackedFloat.unpack Format.binary64 x = .finite s m e positive ∧
        m < 2^53 ∧ e ≤ 74 := by
  have exponent := unpack_exponent_bound bound
  have mantissa := (unpackMantissa (spec := Format.binary64) x).isLt
  unfold UnpackedFloat.unpack
  dsimp only
  split
  · next exceptional =>
    have bad : (unpackExponent (spec := Format.binary64) x).toNat = 2047 := by rw [exceptional]; rfl
    omega
  · split
    · next subnormal =>
      split
      · exact Or.inl ⟨_, rfl⟩
      · next nonzero =>
        have exponentZero : (unpackExponent (spec := Format.binary64) x).toNat = 0 := by rw [subnormal]; rfl
        have positive : 0 < (unpackMantissa (spec := Format.binary64) x).toNat := by
          apply Nat.pos_of_ne_zero
          intro zero
          exact nonzero (BitVec.eq_of_toNat_eq zero)
        refine Or.inr ⟨_, _, _, positive, rfl, ?_, ?_⟩
        · change (unpackMantissa (spec := Format.binary64) x).toNat < 9007199254740992
          change (unpackMantissa (spec := Format.binary64) x).toNat < 4503599627370496 at mantissa
          omega
        · simp only [Format.exponentBias, Format.binary64] at *
          omega
    · refine Or.inr ⟨_, _, _, by rw [W8A8IEEE32.append_toNat]; change 0 < 1 * 2^52 + _; omega,
        rfl, ?_, ?_⟩
      · rw [W8A8IEEE32.append_toNat]
        change 1 * 2^52 + (unpackMantissa (spec := Format.binary64) x).toNat < 9007199254740992
        change (unpackMantissa (spec := Format.binary64) x).toNat < 4503599627370496 at mantissa
        omega
      · simp only [Format.exponentBias, Format.binary64] at *
        omega

theorem narrow_finite {x : FBits .f64} (bound : W8A8FiniteLaws.Narrowable x) :
    W8A8FiniteLaws.Finite (W8A8ArithmeticModel.narrow x) := by
  rcases unpack_cases bound with ⟨sign, decoded⟩ | ⟨sign, m, e, positive, decoded, mantissa, exponent⟩
  · simp only [W8A8ArithmeticModel.narrow, decoded, W8A8ArithmeticModel.rounded]
    cases sign <;> decide
  · simp only [W8A8ArithmeticModel.narrow, decoded, W8A8ArithmeticModel.rounded]
    rcases W8A8IEEERoundingBounds.round_zero_or_finite Format.binary32 sign m e with
        rounded | ⟨m', e', p', rounded⟩
    · rw [rounded]
      cases sign <;> decide
    · rw [rounded]
      have roundedBound := W8A8IEEERoundingBounds.round_exponent positive rounded
      have logBound := (Nat.log2_lt (Nat.ne_of_gt positive)).2 mantissa
      simp only [Format.targetExponent, Float.Model.totalExponent, Format.mantissaBits,
        Format.minExponent] at roundedBound
      apply W8A8IEEEBounds.pack_finite_exponent
      omega

#print axioms narrow_finite

end MerlinVerify.Rules.W8A8IEEENarrow
