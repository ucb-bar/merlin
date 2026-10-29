/- Identity normalization is a separate boundary, never positive rewrite coverage.
The complete parsed bindings agree and the module contains no shape rewrite sites.
The existing typed module theorem still supplies the semantic mode bridge. -/
import Merlin.GraphLoopsShapeModel

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Identity

def Valid (p : Model.Projection) : Prop :=
  p.module.Valid p.bits ∧ (p.module.functions.map (fun f => Model.changed f.code)).sum = 0 ∧
  p.sourceBinding = p.targetBinding ∧ p.sourceBinding ≠ ""

instance (p : Model.Projection) : Decidable (Valid p) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

def check (p : Model.Projection) : Bool := decide (Valid p)

theorem check_sound {p : Model.Projection} (accepted : check p = true) : Valid p :=
  of_decide_eq_true accepted

theorem checked_complete {Value World : Type} {p : Model.Projection} (accepted : check p = true)
    (meaning : Interpretation Value World) (name : String) (inputs : List (Runtime Value))
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    complete meaning p.module false name inputs before outputs after ↔
      complete meaning p.module true name inputs before outputs after :=
  Module.checked_complete p.module p.bits (check_sound accepted).1 meaning name inputs before outputs after

theorem checked_memory {Value Public : Type} {p : Model.Projection} (accepted : check p = true)
    (base : Interpretation Value (MemoryWorld Value Public)) (classify : String → MemoryOpcode)
    (name : String) (inputs : List (Runtime Value)) (before : MemoryWorld Value Public)
    (outputs : List (Runtime Value)) (after : MemoryWorld Value Public) :
    complete (withMemory base classify) p.module false name inputs before outputs after ↔
      complete (withMemory base classify) p.module true name inputs before outputs after :=
  Module.checked_memory p.module p.bits (check_sound accepted).1 base classify name inputs before outputs after

theorem checked_context {Value World Outcome : Type} {p : Model.Projection} (accepted : check p = true)
    (meaning : Interpretation Value World) (context : (String → RegionMeaning Value World) → Outcome) :
    context (complete meaning p.module false) = context (complete meaning p.module true) := by
  have same : complete meaning p.module false = complete meaning p.module true := by
    funext name inputs before outputs after
    exact propext (checked_complete accepted meaning name inputs before outputs after)
  rw [same]

end Merlin.GraphLoopsShape.Identity
