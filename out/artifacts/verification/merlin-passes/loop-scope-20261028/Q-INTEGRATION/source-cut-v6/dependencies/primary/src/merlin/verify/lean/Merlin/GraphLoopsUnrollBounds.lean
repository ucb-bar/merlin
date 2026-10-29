/- Signed bounds for actual inline and bulk/tail ordered unrolling.
Only control/index arithmetic is used; body arithmetic remains opaque. -/
import Merlin.GraphLoopsUnrollGrouping

set_option autoImplicit false

namespace Merlin.GraphLoopsUnroll

open GraphLoopsSegment

structure Bounds where
  bits : Nat
  lower : Nat
  upper : Nat
  step : Nat
  factor : Nat
  count : Nat
  groups : Nat
  remainder : Nat
  deriving DecidableEq, Repr

def Bounds.Valid (b : Bounds) : Prop :=
  (b.bits = 32 ∨ b.bits = 64) ∧ b.lower ≤ b.upper ∧ 0 < b.step ∧ 1 < b.factor ∧
  CountCorrect b.lower b.upper b.step b.count ∧ b.groups * b.factor + b.remainder = b.count ∧
  b.remainder < b.factor ∧ b.upper + b.step * b.factor < 2 ^ (b.bits - 1)

instance (b : Bounds) : Decidable b.Valid :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))

theorem CountCorrect.index_bound {lower upper step count index : Nat}
    (correct : CountCorrect lower upper step count) (inside : index < count) : lower + step * index < upper := by
  cases count with
  | zero => omega
  | succ count =>
    have monotone := Nat.mul_le_mul_left step (show index ≤ count by omega)
    have last := correct.1
    omega

theorem CountCorrect.one_past {lower upper step count : Nat}
    (correct : CountCorrect lower upper step count) (ordered : lower ≤ upper) : lower + step * count ≤ upper + step := by
  cases count with
  | zero => simpa only [Nat.mul_zero, Nat.add_zero] using (show lower ≤ upper + step by omega)
  | succ count =>
    have last := correct.1
    rw [Nat.mul_succ]
    omega

theorem Bounds.constants {b : Bounds} (valid : b.Valid) :
    FitsSigned b.bits b.lower ∧ FitsSigned b.bits b.upper ∧ FitsSigned b.bits b.step ∧
    FitsSigned b.bits (b.step * b.factor) ∧ FitsSigned b.bits (b.lower + b.step * (b.groups * b.factor)) := by
  rcases valid with ⟨_, ordered, positive, factor, correct, partition, _, bound⟩
  have stepLe : b.step ≤ b.step * b.factor := by
    simpa only [Nat.mul_one] using Nat.mul_le_mul_left b.step (show 1 ≤ b.factor by omega)
  have bulkLe := Nat.mul_le_mul_left b.step (show b.groups * b.factor ≤ b.count by omega)
  have past := correct.one_past ordered
  exact ⟨fits_signed_of_lt (by omega), fits_signed_of_lt (by omega), fits_signed_of_lt (by omega),
    fits_signed_of_lt (by omega), fits_signed_of_lt (by omega)⟩

theorem Bounds.lane_ordinal {b : Bounds} (valid : b.Valid) {group lane : Nat}
    (insideGroup : group < b.groups) (insideLane : lane < b.factor) : group * b.factor + lane < b.count := by
  have next := Nat.mul_le_mul_right b.factor (show group + 1 ≤ b.groups by omega)
  rw [Nat.add_mul, Nat.one_mul] at next
  have partition := valid.2.2.2.2.2.1
  omega

theorem Bounds.lane_control {b : Bounds} (valid : b.Valid) {group lane : Nat}
    (insideGroup : group < b.groups) (insideLane : lane < b.factor) :
    b.lower + (b.step * b.factor) * group + b.step * lane = b.lower + b.step * (group * b.factor + lane) ∧
    FitsSigned b.bits (b.step * lane) ∧
    FitsSigned b.bits (b.lower + (b.step * b.factor) * group + b.step * lane) := by
  have equal : b.lower + (b.step * b.factor) * group + b.step * lane = b.lower + b.step * (group * b.factor + lane) := by
    simp only [Nat.mul_add, Nat.mul_comm, Nat.mul_left_comm, Nat.add_assoc]
  have ordinal := Bounds.lane_ordinal valid insideGroup insideLane
  have live := valid.2.2.2.2.1.index_bound ordinal
  have laneLe := Nat.mul_le_mul_left b.step (show lane ≤ b.factor by omega)
  have bound := valid.2.2.2.2.2.2.2
  refine ⟨equal, fits_signed_of_lt (by omega), ?_⟩
  rw [equal]
  exact fits_signed_of_lt (by omega)

theorem Bounds.source_control {b : Bounds} (valid : b.Valid) {index : Nat} (inside : index < b.upper) :
    FitsSigned b.bits index ∧ FitsSigned b.bits (index + b.step) := by
  have stepLe : b.step ≤ b.step * b.factor := by
    simpa only [Nat.mul_one] using Nat.mul_le_mul_left b.step (show 1 ≤ b.factor from Nat.le_of_lt valid.2.2.2.1)
  have bound := valid.2.2.2.2.2.2.2
  exact ⟨fits_signed_of_lt (by omega), fits_signed_of_lt (by omega)⟩

/-- Includes the last bulk increment, which may lie above the original upper
bound when the source's last iteration is not aligned with that bound. -/
theorem Bounds.bulk_control {b : Bounds} (valid : b.Valid) {group : Nat} (inside : group ≤ b.groups) :
    FitsSigned b.bits (b.lower + (b.step * b.factor) * group) := by
  have monotone := Nat.mul_le_mul_left (b.step * b.factor) inside
  have endpoint := (Bounds.constants valid).2.2.2.2
  have rearrange : (b.step * b.factor) * b.groups = b.step * (b.groups * b.factor) := by
    simp only [Nat.mul_comm, Nat.mul_left_comm]
  rw [rearrange] at monotone
  unfold FitsSigned at endpoint ⊢
  omega

theorem Bounds.inline_control {b : Bounds} (valid : b.Valid) {ordinal : Nat} (inside : ordinal < b.count) :
    FitsSigned b.bits (b.lower + b.step * ordinal) ∧
    FitsSigned b.bits (b.lower + b.step * ordinal + b.step) :=
  Bounds.source_control valid (valid.2.2.2.2.1.index_bound inside)

end Merlin.GraphLoopsUnroll
