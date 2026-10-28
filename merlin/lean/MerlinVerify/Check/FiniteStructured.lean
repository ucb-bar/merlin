/- Elementwise finite/normal facts and checked reduction invariants. No arithmetic operation
is presumed to preserve these facts merely because its result is initialized. -/
import MerlinVerify.Check.FiniteImplication
import MerlinVerify.Check.DefinedContextGeneric

namespace MerlinVerify.Check.FiniteFacts.Structured

open MerlinVerify Definedness

theorem elementwise : Refined.Elementwise model := by
  intro info tensor
  cases info with
  | any => exact ⟨fun _ _ _ => trivial, fun _ => trivial⟩
  | finite => rfl
  | normal => rfl

def widen : Class → Class
  | .normal => .finite
  | cls => cls

theorem widen_sound (info : Class) (value : RVal) (holds : model.holds info value) :
    model.holds (widen info) value := by
  cases info with
  | any => exact holds
  | finite => exact holds
  | normal => exact Implication.classes_sound (source := .normal) (target := .finite) rfl holds

def invariants (ib : Nat) : Contextual.Structured.Invariants model ib where
  widen := widen
  widen_sound := widen_sound
  implies := Implication.check
  implies_sound _ _ checked _ holds := Implication.check_sound checked holds

#print axioms elementwise
#print axioms widen_sound
#print axioms invariants

end MerlinVerify.Check.FiniteFacts.Structured
