/- Conservative binary32 exponent bounds, proved for the executable rounded model.
The bounds describe bit encodings and include signed zero/subnormal values. They do
not assert native primitive fidelity or a bound on any unbound model parameter. -/
import MerlinVerify.Rules.W8A8IEEE32

namespace MerlinVerify.Rules.W8A8IEEEBounds

open MerlinVerify Float.Model.UnpackedFloat W8A8IEEE32
open Float.Model (Format UnpackedFloat totalExponent)

def ExponentLE (upper : Nat) (x : FBits .f32) : Prop :=
  Finite x ∧ W8A8ByteFloat.exponent x ≤ upper

instance (upper : Nat) (x : FBits .f32) : Decidable (ExponentLE upper x) :=
  inferInstanceAs (Decidable (Finite x ∧ W8A8ByteFloat.exponent x ≤ upper))

theorem shifted_mantissa (em : ExtendedMantissa) (n : Nat) :
    (em >>> n).mantissa = em.mantissa / 2^n := by
  induction n with
  | zero => simp [HShiftRight.hShiftRight, Nat.repeat]
  | succ n ih =>
    change (Nat.repeat ExtendedMantissa.shiftRightOne (n + 1) em).mantissa = _
    change (em >>> n).mantissa / 2 = _
    rw [ih, Nat.div_div_eq_div_mul, Nat.pow_succ]

theorem roundedMantissa_le (em : ExtendedMantissa) :
    em.roundedMantissa ≤ em.mantissa + 1 := by
  unfold ExtendedMantissa.roundedMantissa Accuracy.roundToNearestEven
  split <;> omega

theorem shift_exponent (m : Nat) (e : Int) (a : Accuracy) :
    (shiftToTargetExponent format m e a).2 =
      max e (max ((m.log2 : Int) + e - 23) (-149)) := by
  simp only [shiftToTargetExponent, shiftToExponent, Format.targetExponent,
    totalExponent, Format.mantissaBits, Format.minExponent]
  omega

theorem shifted_target_mantissa_lt (m : Nat) (e : Int) (a : Accuracy) :
    (shiftToTargetExponent format m e a).1.mantissa < 2^24 := by
  have initial : (ExtendedMantissa.ofMantissaAndAccuracy m a).mantissa = m := by
    cases a with
    | exact => rfl
    | inexact cmp => cases cmp <;> rfl
  simp only [shiftToTargetExponent, shiftToExponent, shifted_mantissa, initial,
    Format.targetExponent, totalExponent, Format.mantissaBits, Format.minExponent]
  apply (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).2
  rw [← Nat.pow_add]
  have bound := Nat.lt_log2_self (n := m)
  have exponents : m.log2 + 1 ≤ 24 +
      (max ((m.log2 : Int) + 1 + e - (1 + 23)) (3 - 2^(8 - 1) - (1 + 23)) - e).toNat := by omega
  exact Nat.lt_of_lt_of_le bound (Nat.pow_le_pow_right (by decide) exponents)

theorem rounded_exponent_bound {s : Sign} {m : Nat} {e : Int} {a : Accuracy}
    {s' : Sign} {m' : Nat} {e' : Int} {positive : 0 < m'}
    (hm : m < 2^48)
    (rounded : roundWithAccuracy format s m e a = .finite s' m' e' positive) :
    e' ≤ max (e + 25) (-148) := by
  have firstMantissa := shifted_target_mantissa_lt m e a
  have firstExponent := shift_exponent m e a
  have roundedMantissa := roundedMantissa_le (shiftToTargetExponent format m e a).1
  have secondMantissa : ((shiftToTargetExponent format m e a).1.roundedMantissa).log2 ≤ 24 := by
    by_cases zero : (shiftToTargetExponent format m e a).1.roundedMantissa = 0
    · simp [zero]
    · have upper : (shiftToTargetExponent format m e a).1.roundedMantissa < 2^25 := by omega
      have := (Nat.log2_lt zero).2 upper
      omega
  have logBound : m.log2 ≤ 47 := by
    by_cases zero : m = 0
    · simp [zero]
    · have := (Nat.log2_lt zero).2 hm
      omega
  have secondExponent := shift_exponent
    (shiftToTargetExponent format m e a).1.roundedMantissa
    (shiftToTargetExponent format m e a).2 .exact
  unfold roundWithAccuracy at rounded
  dsimp only at rounded
  split at rounded
  · cases rounded
  · cases rounded
    omega

theorem pack_finite_exponent {s : Sign} {m : Nat} {e : Int} {positive : 0 < m}
    (bound : e ≤ 104) : Finite (W8A8IEEE32.pack (.finite s m e positive)) := by
  change W8A8ByteFloat.exponent (UnpackedFloat.pack format (.finite s m e positive)) < 255
  simp only [UnpackedFloat.pack]
  have biased : (e + (format.exponentBias : Int) + format.mantissaBitsWithoutImplicit).toNat < 255 := by
    change (e + 127 + 23).toNat < 255
    omega
  have noOverflow : ¬ 2^format.exponentBits ≤
      (e + (format.exponentBias : Int) + format.mantissaBitsWithoutImplicit).toNat + 1 := by
    change ¬ 256 ≤ _
    omega
  rw [ite_eq_right noOverflow]
  split
  · simp only [exponent_packComponents,
      BitVec.toNat_ofNat]
    exact Nat.lt_of_le_of_lt (Nat.mod_le _ _) biased
  · change W8A8ByteFloat.exponent (packComponents format s 0 _) < 255
    rw [exponent_packComponents]
    decide

theorem pack_exponent_le {s : Sign} {m upper : Nat} {e : Int} {positive : 0 < m}
    (finite : e ≤ 104) (bound : e + 150 ≤ upper) :
    ExponentLE upper (W8A8IEEE32.pack (.finite s m e positive)) := by
  refine ⟨pack_finite_exponent finite, ?_⟩
  change W8A8ByteFloat.exponent (UnpackedFloat.pack format (.finite s m e positive)) ≤ upper
  simp only [UnpackedFloat.pack]
  split
  · next overflow => change 256 ≤ (e + 127 + 23).toNat + 1 at overflow; omega
  · split
    · rw [exponent_packComponents, BitVec.toNat_ofNat]
      have bounded : (e + 127 + 23).toNat ≤ upper := by omega
      exact Nat.le_trans (Nat.mod_le _ _) bounded
    · rw [exponent_packComponents]
      exact Nat.zero_le _

theorem unpack_finite_bounds {x : FBits .f32} {upper : Nat}
    (h : ExponentLE upper x) {s : Sign} {m : Nat} {e : Int} {positive : 0 < m}
    (decoded : W8A8IEEE32.unpack x = .finite s m e positive) :
    m < 2^24 ∧ e ≤ (max upper 1 : Nat) - (150 : Int) := by
  have cap := h.2
  rw [exponent_eq_unpack] at cap
  unfold W8A8IEEE32.unpack UnpackedFloat.unpack at decoded
  dsimp only at decoded
  split at decoded
  · split at decoded <;> cases decoded
  · split at decoded
    · next zero =>
      split at decoded
      · cases decoded
      · cases decoded
        have mantissa := (unpackMantissa (spec := format) x).isLt
        have exponent : (unpackExponent (spec := format) x).toNat = 0 := by rw [zero]; rfl
        change _ < 16777216 ∧ _ ≤ _
        simp only [format, Format.binary32, Format.exponentBias] at *
        omega
    · cases decoded
      have mantissa := (unpackMantissa (spec := format) x).isLt
      rw [append_toNat]
      change 1 * 2^23 + _ < 16777216 ∧ _ ≤ _
      simp only [format, Format.binary32, Format.exponentBias] at *
      omega

/-- A conservative bound checked from both actual operand exponent bounds. It
allows underflow and both signed zeros; it never assumes finite products close. -/
def mulCap (left right : Nat) : Nat := max (max left 1 + max right 1 - 125) 2

theorem mul_bounded {x y : FBits .f32} {left right : Nat}
    (hx : ExponentLE left x) (hy : ExponentLE right y)
    (safe : max left 1 + max right 1 ≤ 379) :
    ExponentLE (mulCap left right) (W8A8IEEE32.mul x y) := by
  have nx := unpack_notNaN (finite_notNaN hx.1)
  have ny := unpack_notNaN (finite_notNaN hy.1)
  have ix := unpack_notInf hx.1
  have iy := unpack_notInf hy.1
  unfold W8A8IEEE32.mul
  cases dx : W8A8IEEE32.unpack x <;> cases dy : W8A8IEEE32.unpack y
  all_goals try (exact False.elim (nx dx))
  all_goals try (exact False.elim (ny dy))
  all_goals try (exact False.elim (ix _ dx))
  all_goals try (exact False.elim (iy _ dy))
  all_goals try (simp [UnpackedFloat.mul, W8A8IEEE32.pack, UnpackedFloat.pack, packedZero,
    ExponentLE, Finite, W8A8FiniteLaws.Finite, exponent_packComponents]; done)
  case finite.finite sx mx ex px sy my ey py =>
    obtain ⟨mxBound, exBound⟩ := unpack_finite_bounds hx dx
    obtain ⟨myBound, eyBound⟩ := unpack_finite_bounds hy dy
    have product : mx * my < 2^48 := by
      have := Nat.mul_lt_mul_of_lt_of_le mxBound (Nat.le_of_lt myBound) (by decide : 0 < 2^24)
      simpa using this
    simp only [UnpackedFloat.mul]
    cases result : roundWithAccuracy format (sx * sy) (mx * my) (ex + ey) .exact with
    | notANumber => exact False.elim (round_notNaN _ _ _ _ result)
    | infinity sign =>
      unfold roundWithAccuracy at result
      dsimp only at result
      split at result <;> cases result
    | zero sign =>
      simp [W8A8IEEE32.pack, UnpackedFloat.pack, packedZero, ExponentLE, Finite,
        W8A8FiniteLaws.Finite, exponent_packComponents]
    | finite sign mantissa exponent positive =>
      have bound := rounded_exponent_bound product result
      apply pack_exponent_le (by omega)
      simp only [mulCap]
      omega

theorem mul_finite {x y : FBits .f32} {left right : Nat}
    (hx : ExponentLE left x) (hy : ExponentLE right y)
    (safe : max left 1 + max right 1 ≤ 379) : Finite (W8A8IEEE32.mul x y) :=
  (mul_bounded hx hy safe).1

/-- Rounded addition of a signed zero cannot turn a finite value into infinity.
Signed-zero result bits are deliberately not identified with the input's bits. -/
theorem add_zero_bounded {x : FBits .f32} {upper : Nat} (hx : ExponentLE upper x) (sign : Sign) :
    ExponentLE (max upper 1) (W8A8IEEE32.add x (W8A8IEEE32.pack (.zero sign))) := by
  have nx := unpack_notNaN (finite_notNaN hx.1)
  have ix := unpack_notInf hx.1
  have unpackZero : W8A8IEEE32.unpack (W8A8IEEE32.pack (.zero sign)) = .zero sign := by
    cases sign <;> rfl
  unfold W8A8IEEE32.add
  rw [unpackZero]
  cases decoded : W8A8IEEE32.unpack x with
  | notANumber => exact False.elim (nx decoded)
  | infinity s => exact False.elim (ix s decoded)
  | zero s =>
    cases s <;> cases sign <;> simp [UnpackedFloat.add, W8A8IEEE32.pack, UnpackedFloat.pack,
      packedZero, ExponentLE, Finite, W8A8FiniteLaws.Finite, exponent_packComponents]
  | finite s m e positive =>
    simp only [UnpackedFloat.add]
    have bounds := unpack_finite_bounds hx decoded
    have finiteBound := unpack_finite_bounds (show ExponentLE 254 x from ⟨hx.1, by have := hx.1; omega⟩) decoded
    exact pack_exponent_le (by omega) (by omega)

theorem add_zero_left_bounded {x : FBits .f32} {upper : Nat} (hx : ExponentLE upper x) (sign : Sign) :
    ExponentLE (max upper 1) (W8A8IEEE32.add (W8A8IEEE32.pack (.zero sign)) x) := by
  have commute : W8A8IEEE32.add (W8A8IEEE32.pack (.zero sign)) x =
      W8A8IEEE32.add x (W8A8IEEE32.pack (.zero sign)) := by
    have zero : W8A8IEEE32.unpack (W8A8IEEE32.pack (.zero sign)) = .zero sign := by cases sign <;> rfl
    simp only [W8A8IEEE32.add, zero]
    cases decoded : W8A8IEEE32.unpack x <;> try rfl
    next s => cases s <;> cases sign <;> rfl
  rw [commute]
  exact add_zero_bounded hx sign

theorem zero_representation {bits : FBits .f32} (zero : W8A8ByteFloat.magnitudeWord bits = 0) :
    ∃ sign, bits = W8A8IEEE32.pack (.zero sign) := by
  have bounded := bits.isLt
  change bits.toNat < 4294967296 at bounded
  unfold W8A8ByteFloat.magnitudeWord at zero
  have either : bits.toNat = 0 ∨ bits.toNat = 2147483648 := by omega
  rcases either with positive | negative
  · exact ⟨.positive, BitVec.eq_of_toNat_eq positive⟩
  · exact ⟨.negative, BitVec.eq_of_toNat_eq negative⟩

#print axioms mul_bounded
#print axioms mul_finite

end MerlinVerify.Rules.W8A8IEEEBounds
