/- All finite nearest words, even significand preference and deterministic ties.
Noncomputable mathematical candidate; no native fidelity or independent approval is claimed. -/
import MerlinVerify.Rules.W8A8RealValue
import Mathlib.Data.Finset.Max

namespace MerlinVerify.Rules.W8A8RoundedReal

open MerlinVerify Rules

noncomputable section

local instance : Fintype (FBits .f32) :=
  Fintype.ofEquiv (Fin (2^32))
    ⟨BitVec.ofFin, BitVec.toFin, fun _ => rfl, fun _ => rfl⟩

def finiteWords : Finset (FBits .f32) := Finset.univ.filter W8A8IEEE32.Finite

theorem finiteWords_nonempty : finiteWords.Nonempty := by
  exact ⟨0, Finset.mem_filter.mpr ⟨Finset.mem_univ _, by decide⟩⟩

theorem nearest_exists (x : ℝ) : ∃ bits ∈ finiteWords,
    ∀ other ∈ finiteWords, |value bits - x| ≤ |value other - x| :=
  finiteWords.exists_min_image (fun bits => |value bits - x|) finiteWords_nonempty

def nearestCandidate (x : ℝ) : FBits .f32 := Classical.choose (nearest_exists x)

theorem nearest_spec (x : ℝ) : nearestCandidate x ∈ finiteWords ∧
    ∀ other ∈ finiteWords, |value (nearestCandidate x) - x| ≤ |value other - x| :=
  Classical.choose_spec (nearest_exists x)

theorem nearest_within_unit {x : ℝ} (lower : -1 ≤ x) (upper : x ≤ 1) :
    -1 ≤ value (nearestCandidate x) ∧ value (nearestCandidate x) ≤ 1 := by
  have one : (0x3f800000 : FBits .f32) ∈ finiteWords :=
    Finset.mem_filter.mpr ⟨Finset.mem_univ _, by decide⟩
  have negOne : (0xbf800000 : FBits .f32) ∈ finiteWords :=
    Finset.mem_filter.mpr ⟨Finset.mem_univ _, by decide⟩
  have below := (nearest_spec x).2 _ negOne
  have above := (nearest_spec x).2 _ one
  rw [value_neg_one, abs_of_nonpos (show -1 - x ≤ 0 from by linarith)] at below
  rw [value_one, abs_of_nonneg (show 0 ≤ 1 - x from by linarith)] at above
  constructor
  · by_contra outside
    rw [abs_of_neg (show value (nearestCandidate x) - x < 0 from by linarith)] at below
    linarith
  · by_contra outside
    rw [abs_of_pos (show 0 < value (nearestCandidate x) - x from by linarith)] at above
    linarith

theorem nearest_cap {x : ℝ} (lower : -1 ≤ x) (upper : x ≤ 1) :
    W8A8IEEEBounds.ExponentLE 127 (nearestCandidate x) := by
  have bounded := nearest_within_unit lower upper
  have absolute : |value (nearestCandidate x)| ≤ 1 := abs_le.mpr bounded
  rw [abs_value] at absolute
  have unitsReal : (W8A8OrderedMagnitude.units (nearestCandidate x) : ℝ) ≤ (2:ℝ)^149 :=
    (div_le_one (by positivity)).mp absolute
  have unitsNat : W8A8OrderedMagnitude.units (nearestCandidate x) ≤ (2:ℕ)^149 := by
    exact_mod_cast unitsReal
  have finite : W8A8IEEE32.Finite (nearestCandidate x) := by
    simpa only [finiteWords, Finset.mem_filter, Finset.mem_univ, true_and] using (nearest_spec x).1
  refine ⟨finite, ?_⟩
  exact W8A8OrderedMagnitude.exponent_le_of_units
    (power := 150) (lt_of_le_of_lt unitsNat (by decide))

def nearestWords (x : ℝ) : Finset (FBits .f32) :=
  finiteWords.filter (fun bits => |value bits - x| = |value (nearestCandidate x) - x|)

theorem nearestWords_nonempty (x : ℝ) : (nearestWords x).Nonempty := by
  exact ⟨nearestCandidate x, Finset.mem_filter.mpr ⟨(nearest_spec x).1, rfl⟩⟩

theorem nearestWords_spec {x : ℝ} {bits : FBits .f32} (member : bits ∈ nearestWords x) :
    bits ∈ finiteWords ∧ ∀ other ∈ finiteWords, |value bits - x| ≤ |value other - x| := by
  obtain ⟨finite, distance⟩ := Finset.mem_filter.mp member
  exact ⟨finite, fun other admissible => distance ▸ (nearest_spec x).2 other admissible⟩

def evenWords (x : ℝ) : Finset (FBits .f32) :=
  (nearestWords x).filter (fun bits => bits.toNat % 2 = 0)

def preferredWords (x : ℝ) : Finset (FBits .f32) :=
  if (evenWords x).Nonempty then evenWords x else nearestWords x

theorem preferredWords_nonempty (x : ℝ) : (preferredWords x).Nonempty := by
  unfold preferredWords
  split
  · assumption
  · exact nearestWords_nonempty x

-- Keep elaboration at the proved set interfaces instead of unfolding the finite universe.

attribute [irreducible] nearestWords evenWords preferredWords

theorem preferredWords_nearest {x : ℝ} {bits : FBits .f32}
    (member : bits ∈ preferredWords x) : bits ∈ nearestWords x := by
  unfold preferredWords at member
  split at member
  · unfold evenWords at member
    exact (Finset.mem_filter.mp member).1
  · exact member

def tieChoice (x : ℝ) : FBits .f32 :=
  Classical.choose ((preferredWords x).exists_min_image BitVec.toNat (preferredWords_nonempty x))

theorem tieChoice_min (x : ℝ) : tieChoice x ∈ preferredWords x ∧
    ∀ other ∈ preferredWords x, (tieChoice x).toNat ≤ other.toNat :=
  Classical.choose_spec ((preferredWords x).exists_min_image BitVec.toNat (preferredWords_nonempty x))

attribute [irreducible] tieChoice

theorem tieChoice_spec (x : ℝ) : tieChoice x ∈ finiteWords ∧
    ∀ other ∈ finiteWords, |value (tieChoice x) - x| ≤ |value other - x| :=
  nearestWords_spec (x := x) (bits := tieChoice x)
    (preferredWords_nearest (x := x) (bits := tieChoice x) (tieChoice_min x).1)

theorem tieChoice_even {x : ℝ} (available : (evenWords x).Nonempty) :
    (tieChoice x).toNat % 2 = 0 := by
  have member : tieChoice x ∈ preferredWords x := (tieChoice_min x).1
  rw [preferredWords, ite_eq_left available] at member
  unfold evenWords at member
  exact (Finset.mem_filter.mp member).2

theorem tieChoice_unique {x : ℝ} {bits : FBits .f32}
    (member : bits ∈ preferredWords x)
    (minimal : ∀ other ∈ preferredWords x, bits.toNat ≤ other.toNat) :
    bits = tieChoice x := by
  apply BitVec.eq_of_toNat_eq
  have left : bits.toNat ≤ (tieChoice x).toNat := minimal _ (tieChoice_min x).1
  have right : (tieChoice x).toNat ≤ bits.toNat := (tieChoice_min x).2 _ member
  exact Nat.le_antisymm left right

theorem minimizer_within_unit {x : ℝ} {bits : FBits .f32}
    (minimal : ∀ other ∈ finiteWords, |value bits - x| ≤ |value other - x|)
    (lower : -1 ≤ x) (upper : x ≤ 1) : -1 ≤ value bits ∧ value bits ≤ 1 := by
  have one : (0x3f800000 : FBits .f32) ∈ finiteWords :=
    Finset.mem_filter.mpr ⟨Finset.mem_univ _, by decide⟩
  have negOne : (0xbf800000 : FBits .f32) ∈ finiteWords :=
    Finset.mem_filter.mpr ⟨Finset.mem_univ _, by decide⟩
  have below := minimal _ negOne
  have above := minimal _ one
  rw [value_neg_one, abs_of_nonpos (show -1 - x ≤ 0 from by linarith)] at below
  rw [value_one, abs_of_nonneg (show 0 ≤ 1 - x from by linarith)] at above
  constructor
  · by_contra outside
    rw [abs_of_neg (show value bits - x < 0 from by linarith)] at below
    linarith
  · by_contra outside
    rw [abs_of_pos (show 0 < value bits - x from by linarith)] at above
    linarith

theorem minimizer_nonnegative {x : ℝ} {bits : FBits .f32}
    (minimal : ∀ other ∈ finiteWords, |value bits - x| ≤ |value other - x|)
    (nonnegative : 0 ≤ x) : 0 ≤ value bits := by
  have compared := minimal (0 : FBits .f32)
    (Finset.mem_filter.mpr ⟨Finset.mem_univ _, by decide⟩)
  rw [value_positive_zero, zero_sub, abs_neg, abs_of_nonneg nonnegative] at compared
  by_contra below
  have negative : value bits - x < 0 := by linarith
  rw [abs_of_neg negative] at compared
  linarith

end
end MerlinVerify.Rules.W8A8RoundedReal
