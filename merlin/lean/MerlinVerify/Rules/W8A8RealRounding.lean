/- Mathematical binary32 nearest-even rounding, signed zeros, underflow and overflow.
Noncomputable mathematical candidate; no native fidelity or independent approval is claimed. -/
import MerlinVerify.Rules.W8A8NearestReal

namespace MerlinVerify.Rules.W8A8RoundedReal

open MerlinVerify Rules

noncomputable section

local instance : Fintype (FBits .f32) :=
  Fintype.ofEquiv (Fin (2^32))
    ⟨BitVec.ofFin, BitVec.toFin, fun _ => rfl, fun _ => rfl⟩

def signedFiniteChoice (x : ℝ) : FBits .f32 :=
  if W8A8OrderedMagnitude.units (tieChoice x) = 0 then
    if x < 0 then (0x80000000 : FBits .f32) else (0 : FBits .f32)
  else tieChoice x

theorem signedFiniteChoice_value (x : ℝ) : value (signedFiniteChoice x) = value (tieChoice x) := by
  unfold signedFiniteChoice
  split
  · next zero =>
    rw [units_zero_value zero]
    split <;> norm_num [value, magnitude, W8A8OrderedMagnitude.units, W8A8ByteFloat.negative,
      W8A8ByteFloat.exponent, W8A8ByteFloat.fraction, W8A8ByteFloat.magnitudeWord]
  · rfl

theorem signedFiniteChoice_spec (x : ℝ) : signedFiniteChoice x ∈ finiteWords ∧
    ∀ other ∈ finiteWords, |value (signedFiniteChoice x) - x| ≤ |value other - x| := by
  constructor
  · unfold signedFiniteChoice
    split
    · split <;> exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, by decide⟩
    · exact (tieChoice_spec x).1
  · rw [signedFiniteChoice_value]
    exact (tieChoice_spec x).2

theorem signedFiniteChoice_even {x : ℝ} (available : (evenWords x).Nonempty) :
    (signedFiniteChoice x).toNat % 2 = 0 := by
  unfold signedFiniteChoice
  split
  · split <;> decide
  · exact tieChoice_even available

theorem signedFiniteChoice_zero : signedFiniteChoice 0 = (0 : FBits .f32) := by
  have compared := (tieChoice_spec 0).2 0 (Finset.mem_filter.mpr ⟨Finset.mem_univ _, by decide⟩)
  have valueZero : value (tieChoice 0) = 0 := by
    have zero : value ((0 : FBits .f32)) = 0 := by
      norm_num [value, magnitude, W8A8OrderedMagnitude.units, W8A8ByteFloat.negative,
        W8A8ByteFloat.exponent, W8A8ByteFloat.fraction, W8A8ByteFloat.magnitudeWord]
    rw [zero, sub_zero, sub_self, abs_zero] at compared
    exact abs_nonpos_iff.mp compared
  simp [signedFiniteChoice, value_zero_units valueZero]

theorem signedFiniteChoice_one : signedFiniteChoice 1 = (0x3f800000 : FBits .f32) := by
  have compared := (tieChoice_spec 1).2 (0x3f800000 : FBits .f32)
    (Finset.mem_filter.mpr ⟨Finset.mem_univ _, by decide⟩)
  have exactValue : value (tieChoice 1) = 1 := by
    rw [value_one, sub_self, abs_zero] at compared
    exact sub_eq_zero.mp (abs_nonpos_iff.mp compared)
  have exactBits := value_one_encoding exactValue
  rw [signedFiniteChoice, exactBits]
  have notZero : W8A8OrderedMagnitude.units (0x3f800000 : FBits .f32) ≠ 0 := by decide
  exact ite_eq_right notZero

theorem signedFiniteChoice_cap {x : ℝ} (lower : -1 ≤ x) (upper : x ≤ 1) :
    W8A8IEEEBounds.ExponentLE 127 (signedFiniteChoice x) := by
  have spec := signedFiniteChoice_spec x
  have bounded := minimizer_within_unit spec.2 lower upper
  have absolute : |value (signedFiniteChoice x)| ≤ 1 := abs_le.mpr bounded
  rw [abs_value] at absolute
  have unitsReal : (W8A8OrderedMagnitude.units (signedFiniteChoice x) : ℝ) ≤ (2:ℝ)^149 :=
    (div_le_one (by positivity)).mp absolute
  have unitsNat : W8A8OrderedMagnitude.units (signedFiniteChoice x) ≤ (2:ℕ)^149 := by
    exact_mod_cast unitsReal
  have finite : W8A8IEEE32.Finite (signedFiniteChoice x) := by
    simpa only [finiteWords, Finset.mem_filter, Finset.mem_univ, true_and] using spec.1
  exact ⟨finite, W8A8OrderedMagnitude.exponent_le_of_units
    (power := 150) (lt_of_le_of_lt unitsNat (by decide))⟩

def overflowPoint : ℝ := (2:ℝ)^128 - (2:ℝ)^103

theorem one_lt_overflowPoint : 1 < overflowPoint := by norm_num [overflowPoint]

def roundedReal (x : ℝ) : FBits .f32 :=
  if overflowPoint ≤ |x| then
    if x < 0 then (0xff800000 : FBits .f32) else (0x7f800000 : FBits .f32)
  else signedFiniteChoice x

theorem roundedReal_unit {x : ℝ} (lower : -1 ≤ x) (upper : x ≤ 1) :
    roundedReal x = signedFiniteChoice x := by
  have smaller : |x| < overflowPoint := lt_of_le_of_lt (abs_le.mpr ⟨lower, upper⟩) one_lt_overflowPoint
  exact ite_eq_right (not_le.mpr smaller)

theorem roundedReal_cap {x : ℝ} (lower : -1 ≤ x) (upper : x ≤ 1) :
    W8A8IEEEBounds.ExponentLE 127 (roundedReal x) := by
  rw [roundedReal_unit lower upper]
  exact signedFiniteChoice_cap lower upper

theorem roundedReal_zero : roundedReal 0 = (0 : FBits .f32) := by
  rw [roundedReal_unit (by norm_num) (by norm_num), signedFiniteChoice_zero]

theorem roundedReal_one : roundedReal 1 = (0x3f800000 : FBits .f32) := by
  rw [roundedReal_unit (by norm_num) (by norm_num), signedFiniteChoice_one]

theorem roundedReal_positive_overflow {x : ℝ} (overflow : overflowPoint ≤ x) :
    roundedReal x = (0x7f800000 : FBits .f32) := by
  have positive : 0 < x := lt_of_lt_of_le (lt_trans (by norm_num) one_lt_overflowPoint) overflow
  rw [roundedReal, abs_of_pos positive, ite_eq_left overflow, ite_eq_right (not_lt.mpr (le_of_lt positive))]

theorem roundedReal_negative_overflow {x : ℝ} (overflow : x ≤ -overflowPoint) :
    roundedReal x = (0xff800000 : FBits .f32) := by
  have negative : x < 0 := by have bound := one_lt_overflowPoint; linarith
  rw [roundedReal, abs_of_neg negative, ite_eq_left (show overflowPoint ≤ -x by linarith),
    ite_eq_left negative]

theorem signedFiniteChoice_nonnegative {x : ℝ} (nonnegative : 0 ≤ x) :
    W8A8IEEE32.Nonnegative (signedFiniteChoice x) := by
  have spec := signedFiniteChoice_spec x
  have finite : W8A8IEEE32.Finite (signedFiniteChoice x) := by
    simpa only [finiteWords, Finset.mem_filter, Finset.mem_univ, true_and] using spec.1
  refine ⟨W8A8IEEE32.finite_notNaN finite, ?_⟩
  by_cases zero : W8A8OrderedMagnitude.units (tieChoice x) = 0
  · rw [signedFiniteChoice, ite_eq_left zero, ite_eq_right (not_lt.mpr nonnegative)]
    decide
  · have nearestNonnegative := minimizer_nonnegative (tieChoice_spec x).2 nonnegative
    have positive : 0 < magnitude (tieChoice x) := by
      unfold magnitude
      exact div_pos (by exact_mod_cast Nat.pos_of_ne_zero zero) (by positivity)
    have sign : W8A8ByteFloat.negative (tieChoice x) = false := by
      cases signed : W8A8ByteFloat.negative (tieChoice x) with
      | false => rfl
      | true =>
        rw [value, signed] at nearestNonnegative
        simp only [↓reduceIte] at nearestNonnegative
        linarith
    rw [signedFiniteChoice, ite_eq_right zero]
    exact sign

def halfLeastSubnormal : ℝ := 1 / (2:ℝ)^150

theorem zero_distance_minimal {x : ℝ} (tiny : |x| ≤ halfLeastSubnormal) (bits : FBits .f32) :
    |x| ≤ |value bits - x| := by
  by_cases zero : W8A8OrderedMagnitude.units bits = 0
  · rw [units_zero_value zero, zero_sub, abs_neg]
  · have unitsLower : (1:ℝ) ≤ (W8A8OrderedMagnitude.units bits : ℝ) := by
      exact_mod_cast (Nat.one_le_iff_ne_zero.mpr zero)
    have spacing : 2 * halfLeastSubnormal ≤ magnitude bits := by
      have exactSpacing : (1:ℝ) / (2:ℝ)^149 = 2 * halfLeastSubnormal := by
        norm_num [halfLeastSubnormal]
      rw [← exactSpacing]
      exact div_le_div_of_nonneg_right unitsLower (by positivity)
    have triangle := abs_add_le (value bits - x) x
    rw [sub_add_cancel, abs_value] at triangle
    linarith

theorem tiny_tieChoice_zero {x : ℝ} (tiny : |x| ≤ halfLeastSubnormal) :
    tieChoice x = (0 : FBits .f32) := by
  have zeroFinite : (0 : FBits .f32) ∈ finiteWords :=
    Finset.mem_filter.mpr ⟨Finset.mem_univ _, by decide⟩
  have zeroNearest : (0 : FBits .f32) ∈ nearestWords x := by
    unfold nearestWords
    refine Finset.mem_filter.mpr ⟨zeroFinite, ?_⟩
    apply le_antisymm
    · rw [value_positive_zero, zero_sub, abs_neg]
      exact zero_distance_minimal tiny _
    · exact (nearest_spec x).2 _ zeroFinite
  have zeroEven : (0 : FBits .f32) ∈ evenWords x := by
    unfold evenWords
    exact Finset.mem_filter.mpr ⟨zeroNearest, by decide⟩
  have available : (evenWords x).Nonempty := ⟨_, zeroEven⟩
  have zeroPreferred : (0 : FBits .f32) ∈ preferredWords x := by
    rw [preferredWords, ite_eq_left available]
    exact zeroEven
  apply BitVec.eq_of_toNat_eq
  have minimal := (tieChoice_min x).2 _ zeroPreferred
  change (tieChoice x).toNat ≤ 0 at minimal
  exact Nat.eq_zero_of_le_zero minimal

theorem positive_half_subnormal_tie :
    roundedReal halfLeastSubnormal = (0 : FBits .f32) := by
  have tiny : |halfLeastSubnormal| ≤ halfLeastSubnormal := by
    rw [abs_of_nonneg (by unfold halfLeastSubnormal; positivity)]
  have lower : -1 ≤ halfLeastSubnormal := by norm_num [halfLeastSubnormal]
  have upper : halfLeastSubnormal ≤ 1 := by norm_num [halfLeastSubnormal]
  rw [roundedReal_unit lower upper, signedFiniteChoice, tiny_tieChoice_zero tiny,
    ite_eq_left (show W8A8OrderedMagnitude.units (0 : FBits .f32) = 0 by decide),
    ite_eq_right (show ¬ halfLeastSubnormal < 0 by norm_num [halfLeastSubnormal])]

theorem negative_half_subnormal_tie :
    roundedReal (-halfLeastSubnormal) = (0x80000000 : FBits .f32) := by
  have tiny : |-halfLeastSubnormal| ≤ halfLeastSubnormal := by
    rw [abs_neg, abs_of_nonneg (by unfold halfLeastSubnormal; positivity)]
  have lower : -1 ≤ -halfLeastSubnormal := by norm_num [halfLeastSubnormal]
  have upper : -halfLeastSubnormal ≤ 1 := by norm_num [halfLeastSubnormal]
  rw [roundedReal_unit lower upper, signedFiniteChoice, tiny_tieChoice_zero tiny,
    ite_eq_left (show W8A8OrderedMagnitude.units (0 : FBits .f32) = 0 by decide),
    ite_eq_left (show -halfLeastSubnormal < 0 by norm_num [halfLeastSubnormal])]

end
end MerlinVerify.Rules.W8A8RoundedReal
