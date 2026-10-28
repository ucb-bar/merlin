/- A separately declared rounded-real sin/cos/exp interpretation and exact exceptional cases.
Noncomputable mathematical candidate; no native fidelity or independent approval is claimed. -/
import MerlinVerify.Rules.W8A8RealRounding

namespace MerlinVerify.Rules.W8A8RoundedReal

open MerlinVerify Rules

noncomputable section

def sine (bits : FBits .f32) : FBits .f32 :=
  if W8A8IEEE32.Finite bits then
    if W8A8ByteFloat.magnitudeWord bits = 0 then bits else roundedReal (Real.sin (value bits))
  else (0x7fc00000 : FBits .f32)

def cosine (bits : FBits .f32) : FBits .f32 :=
  if W8A8IEEE32.Finite bits then roundedReal (Real.cos (value bits)) else (0x7fc00000 : FBits .f32)

def exponential (bits : FBits .f32) : FBits .f32 :=
  if W8A8IEEE32.Finite bits then roundedReal (Real.exp (value bits))
  else if W8A8ByteFloat.isNaN bits then (0x7fc00000 : FBits .f32)
  else if W8A8ByteFloat.negative bits then (0 : FBits .f32) else (0x7f800000 : FBits .f32)

theorem sine_zero : sine (0 : FBits .f32) = (0 : FBits .f32) := by
  unfold sine
  rw [ite_eq_left (show W8A8IEEE32.Finite (0 : FBits .f32) by decide),
    ite_eq_left (show W8A8ByteFloat.magnitudeWord (0 : FBits .f32) = 0 by decide)]

theorem sine_negative_zero : sine (0x80000000 : FBits .f32) = (0x80000000 : FBits .f32) := by
  unfold sine
  rw [ite_eq_left (show W8A8IEEE32.Finite (0x80000000 : FBits .f32) by decide),
    ite_eq_left (show W8A8ByteFloat.magnitudeWord (0x80000000 : FBits .f32) = 0 by decide)]

theorem sine_cap {bits : FBits .f32} (finite : W8A8IEEE32.Finite bits) :
    W8A8IEEEBounds.ExponentLE 127 (sine bits) := by
  rw [sine, ite_eq_left finite]
  split
  · next zero =>
    exact ⟨finite, by simp [W8A8ByteFloat.exponent, zero]⟩
  · exact roundedReal_cap (Real.neg_one_le_sin _) (Real.sin_le_one _)

theorem cosine_cap {bits : FBits .f32} (finite : W8A8IEEE32.Finite bits) :
    W8A8IEEEBounds.ExponentLE 127 (cosine bits) := by
  rw [cosine, ite_eq_left finite]
  exact roundedReal_cap (Real.neg_one_le_cos _) (Real.cos_le_one _)

theorem exponential_negative_infinity : exponential (0xff800000 : FBits .f32) = (0 : FBits .f32) := by
  have finite : ¬ W8A8IEEE32.Finite (0xff800000 : FBits .f32) := by decide
  have nan : W8A8ByteFloat.isNaN (0xff800000 : FBits .f32) = false := by decide
  have sign : W8A8ByteFloat.negative (0xff800000 : FBits .f32) = true := by decide
  unfold exponential
  rw [ite_eq_right finite, nan, sign]
  rfl

theorem exponential_positive_infinity : exponential (0x7f800000 : FBits .f32) = (0x7f800000 : FBits .f32) := by
  have finite : ¬ W8A8IEEE32.Finite (0x7f800000 : FBits .f32) := by decide
  have nan : W8A8ByteFloat.isNaN (0x7f800000 : FBits .f32) = false := by decide
  have sign : W8A8ByteFloat.negative (0x7f800000 : FBits .f32) = false := by decide
  unfold exponential
  rw [ite_eq_right finite, nan, sign]
  rfl

theorem exponential_cap {bits : FBits .f32} (finite : W8A8IEEE32.Finite bits)
    (nonpositive : value bits ≤ 0) :
    W8A8IEEEBounds.ExponentLE 127 (exponential bits) := by
  rw [exponential, ite_eq_left finite]
  apply roundedReal_cap
  · have positive := Real.exp_pos (value bits)
    linarith
  · exact Real.exp_le_one_iff.mpr nonpositive

theorem cosine_zero : cosine (0 : FBits .f32) = (0x3f800000 : FBits .f32) := by
  rw [cosine, ite_eq_left (show W8A8IEEE32.Finite (0 : FBits .f32) by decide),
    value_positive_zero, Real.cos_zero, roundedReal_one]

theorem cosine_negative_zero : cosine (0x80000000 : FBits .f32) = (0x3f800000 : FBits .f32) := by
  rw [cosine, ite_eq_left (show W8A8IEEE32.Finite (0x80000000 : FBits .f32) by decide),
    value_negative_zero, Real.cos_zero, roundedReal_one]

theorem exponential_zero : exponential (0 : FBits .f32) = (0x3f800000 : FBits .f32) := by
  rw [exponential, ite_eq_left (show W8A8IEEE32.Finite (0 : FBits .f32) by decide),
    value_positive_zero, Real.exp_zero, roundedReal_one]

theorem exponential_negative_zero : exponential (0x80000000 : FBits .f32) = (0x3f800000 : FBits .f32) := by
  rw [exponential, ite_eq_left (show W8A8IEEE32.Finite (0x80000000 : FBits .f32) by decide),
    value_negative_zero, Real.exp_zero, roundedReal_one]

theorem sine_nonfinite {bits : FBits .f32} (exceptional : ¬ W8A8IEEE32.Finite bits) :
    sine bits = (0x7fc00000 : FBits .f32) := ite_eq_right exceptional

theorem cosine_nonfinite {bits : FBits .f32} (exceptional : ¬ W8A8IEEE32.Finite bits) :
    cosine bits = (0x7fc00000 : FBits .f32) := ite_eq_right exceptional

theorem exponential_nan {bits : FBits .f32} (nan : W8A8ByteFloat.isNaN bits = true) :
    exponential bits = (0x7fc00000 : FBits .f32) := by
  have exceptional : ¬ W8A8IEEE32.Finite bits := by
    intro finite
    have notNan := W8A8IEEE32.finite_notNaN finite
    rw [W8A8IEEE32.NonNaN, nan] at notNan
    cases notNan
  rw [exponential, ite_eq_right exceptional, nan]
  rfl

theorem exponential_nonnegative {bits : FBits .f32} (finite : W8A8IEEE32.Finite bits)
    (nonpositive : value bits ≤ 0) : W8A8IEEE32.Nonnegative (exponential bits) := by
  have positive := Real.exp_pos (value bits)
  have bounded := Real.exp_le_one_iff.mpr nonpositive
  rw [exponential, ite_eq_left finite, roundedReal_unit (by linarith) bounded]
  exact signedFiniteChoice_nonnegative (le_of_lt positive)

end
end MerlinVerify.Rules.W8A8RoundedReal
