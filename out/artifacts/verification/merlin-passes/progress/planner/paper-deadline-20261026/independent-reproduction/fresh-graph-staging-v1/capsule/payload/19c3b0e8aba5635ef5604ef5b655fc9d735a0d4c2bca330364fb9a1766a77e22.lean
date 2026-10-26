/- Signed index safety derived from actual strip-mining control parameters.
This does not establish body memory safety or a counted Kernel/SCF bridge. -/
import Merlin.GraphLoopsSegment

set_option autoImplicit false

namespace Merlin.GraphLoopsSegment

structure Bounds where
  bits : Nat
  lower : Nat
  upper : Nat
  step : Nat
  tile : Nat
  deriving DecidableEq, Repr

def Bounds.Valid (bounds : Bounds) : Prop :=
  (bounds.bits = 32 ∨ bounds.bits = 64) ∧ bounds.lower ≤ bounds.upper ∧
  0 < bounds.step ∧ 1 < bounds.tile ∧
  bounds.upper + bounds.step * bounds.tile < 2 ^ (bounds.bits - 1) ∧
  bounds.upper + bounds.step < 2 ^ (bounds.bits - 1)

instance (bounds : Bounds) : Decidable bounds.Valid :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))

/-- Exact signed interval, with a natural power cast to Int. -/
def FitsSigned (bits value : Nat) : Prop :=
  -((2 ^ (bits - 1) : Nat) : Int) ≤ (value : Int) ∧
    (value : Int) < ((2 ^ (bits - 1) : Nat) : Int)

theorem fits_signed_of_lt {bits value : Nat} (bound : value < 2 ^ (bits - 1)) : FitsSigned bits value := by
  unfold FitsSigned
  omega

theorem Bounds.constants {bounds : Bounds} (valid : bounds.Valid) :
    FitsSigned bounds.bits bounds.lower ∧ FitsSigned bounds.bits bounds.upper ∧
    FitsSigned bounds.bits bounds.step ∧ FitsSigned bounds.bits (bounds.step * bounds.tile) := by
  rcases valid with ⟨_, ordered, _, _, blockBound, stepBound⟩
  exact ⟨fits_signed_of_lt (by omega), fits_signed_of_lt (by omega),
    fits_signed_of_lt (by omega), fits_signed_of_lt (by omega)⟩

/-- Both the generated bound add and final outer increment fit; minui is
applied to nonnegative, fitting values, and its result fits as well. -/
theorem Bounds.outer_control {bounds : Bounds} (valid : bounds.Valid) {outer : Nat}
    (inside : outer < bounds.upper) :
    FitsSigned bounds.bits outer ∧ FitsSigned bounds.bits (outer + bounds.step * bounds.tile) ∧
    FitsSigned bounds.bits (min (outer + bounds.step * bounds.tile) bounds.upper) := by
  have blockBound := valid.2.2.2.2.1
  have minimum := Nat.min_le_right (outer + bounds.step * bounds.tile) bounds.upper
  exact ⟨fits_signed_of_lt (by omega), fits_signed_of_lt (by omega), fits_signed_of_lt (by omega)⟩

/-- Every live inner IV and its final increment fit, including partial tails. -/
theorem Bounds.inner_control {bounds : Bounds} (valid : bounds.Valid) {outer inner : Nat}
    (inside : inner < min (outer + bounds.step * bounds.tile) bounds.upper) :
    FitsSigned bounds.bits inner ∧ FitsSigned bounds.bits (inner + bounds.step) := by
  have stepBound := valid.2.2.2.2.2
  have upper := Nat.min_le_right (outer + bounds.step * bounds.tile) bounds.upper
  exact ⟨fits_signed_of_lt (by omega), fits_signed_of_lt (by omega)⟩

theorem Execution.indices {σ : Type} {body : Nat → σ → σ → Prop} {upper step lower : Nat}
    {initial final : σ} {trace : List Nat} (run : Execution body upper step lower initial trace final) :
    ∀ index ∈ trace, lower ≤ index ∧ index < upper := by
  induction run with
  | done => simp
  | @more lower initial middle final trace inside transition remaining induction =>
    intro index member
    rcases List.mem_cons.mp member with rfl | later
    · exact ⟨Nat.le_refl _, inside⟩
    · have bounds := induction index later
      exact ⟨by omega, bounds.2⟩

theorem Execution.signed_indices {σ : Type} {body : Nat → σ → σ → Prop} {bounds : Bounds}
    (valid : bounds.Valid) {initial final : σ} {trace : List Nat}
    (run : Execution body bounds.upper bounds.step bounds.lower initial trace final) :
    ∀ index ∈ trace, FitsSigned bounds.bits index ∧ FitsSigned bounds.bits (index + bounds.step) := by
  intro index member
  have inside := (run.indices index member).2
  have stepBound := valid.2.2.2.2.2
  exact ⟨fits_signed_of_lt (by omega), fits_signed_of_lt (by omega)⟩

end Merlin.GraphLoopsSegment
