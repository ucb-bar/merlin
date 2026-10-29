/- Necessary-premise countermodels. They are preserved refusals/domains, not
extra axioms or alternative semantics silently selected by the validator. -/
import Merlin.GraphLoopsShapeWitness

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Countermodels

def type : ViewType := ⟨[2, 3], 0, [3, 1], "i32", 0⟩
def first : Descriptor := ⟨1, 100, 0, [2, 3], [3, 1], "i32", 0⟩
def second : Descriptor := ⟨2, 100, 0, [2, 3], [3, 1], "i32", 0⟩

theorem same_type_different_allocation :
    Conforms type first ∧ Conforms type second ∧ address first [0, 0] ≠ address second [0, 0] := by decide

theorem same_type_cannot_change_buffer : ¬ Cast type type first second := by
  intro changed
  have identity := cast_descriptor changed
  have wrong : second.allocation ≠ first.allocation := by decide
  exact wrong (congrArg Descriptor.allocation identity)

def LoggedCast (before after : Nat) : Prop := Cast type type first first ∧ after = before + 1
def ErasedCast (before after : Nat) : Prop := after = before

/-- A logging/trapping intrinsic is not interchangeable with deletion merely
because a host source/type hash matches. Local nonobservability is essential. -/
theorem logging_deletion : LoggedCast 0 1 ∧ ¬ ErasedCast 0 1 := by
  refine ⟨⟨identical_cast type first (by decide), rfl⟩, ?_⟩
  change (1 : Nat) ≠ 0
  decide

def TrappingCast (_before _after : Nat) : Prop := False

theorem trapping_deletion : (¬ ∃ after, TrappingCast 0 after) ∧ ErasedCast 0 0 := by
  refine ⟨?_, rfl⟩
  rintro ⟨_after, impossible⟩
  exact impossible

def badShape : Descriptor := ⟨1, 100, 0, [4, 3], [3, 1], "i32", 0⟩

theorem inconsistent_metadata :
    ¬ Conforms type badShape ∧ badShape.shape[0]? = some 4 ∧ type.shape[0]? = some 2 := by decide

/-- Static shape equality also does not justify changing an aligned base. -/
def shiftedBase : Descriptor := ⟨1, 101, 0, [2, 3], [3, 1], "i32", 0⟩

theorem same_type_different_base :
    Conforms type shiftedBase ∧ address shiftedBase [0, 0] ≠ address first [0, 0] := by decide

end Merlin.GraphLoopsShape.Countermodels
