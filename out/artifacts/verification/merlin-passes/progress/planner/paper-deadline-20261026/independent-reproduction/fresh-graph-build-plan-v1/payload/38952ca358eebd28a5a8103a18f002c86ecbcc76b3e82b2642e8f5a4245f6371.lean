/- Actual whole-module shape blueprint checker. The complete raw-IR projection
is trusted Python. Its strings bind syntax; they do NOT prove descriptor truth. -/
import Merlin.GraphLoopsShapeMemory

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Model

def changed : Code → Nat
  | .returned _ => 0
  | .cast _ _ _ _ rest => 1 + changed rest
  | .dimension _ _ _ _ _ _ rest => 1 + changed rest
  | .constant _ _ rest => changed rest
  | .action _ _ _ rest => changed rest
  | .region _ _ _ _ body rest => changed body + changed rest

structure Projection where
  bits : Nat
  module : Module
  sourceBinding : String
  targetBinding : String
  deriving DecidableEq, Repr

def Valid (p : Projection) : Prop :=
  p.module.Valid p.bits ∧ 0 < (p.module.functions.map (fun function => changed function.code)).sum ∧
  p.sourceBinding ≠ "" ∧ p.targetBinding ≠ ""

instance (p : Projection) : Decidable (Valid p) := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

def check (p : Projection) : Bool := decide (Valid p)

theorem check_sound {p : Projection} (accepted : check p = true) : Valid p := of_decide_eq_true accepted

/-- Both shape rewrite forms, all functions, nested regions, aliases, returns
and entire modeled worlds. Call meanings unfold internally; external/region
primitive and runtime-descriptor fidelity remain explicit local contracts. -/
theorem checked_complete {Value World : Type} {p : Projection} (accepted : check p = true)
    (meaning : Interpretation Value World) (name : String) (inputs : List (Runtime Value))
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    complete meaning p.module false name inputs before outputs after ↔
      complete meaning p.module true name inputs before outputs after :=
  Module.checked_complete p.module p.bits (check_sound accepted).1 meaning name inputs before outputs after

theorem checked_memory {Value Public : Type} {p : Projection} (accepted : check p = true)
    (base : Interpretation Value (MemoryWorld Value Public)) (classify : String → MemoryOpcode)
    (name : String) (inputs : List (Runtime Value)) (before : MemoryWorld Value Public)
    (outputs : List (Runtime Value)) (after : MemoryWorld Value Public) :
    complete (withMemory base classify) p.module false name inputs before outputs after ↔
      complete (withMemory base classify) p.module true name inputs before outputs after :=
  Module.checked_memory p.module p.bits (check_sound accepted).1 base classify name inputs before outputs after

/-- Any semantic caller of the full function environment observes equal complete
outcomes. Syntax bytes, native errors, allocator ABI and hardware are not callers
of this declared denotation. No complete-program agreement premise is used. -/
theorem checked_context {Value World Outcome : Type} {p : Projection} (accepted : check p = true)
    (meaning : Interpretation Value World) (context : (String → RegionMeaning Value World) → Outcome) :
    context (complete meaning p.module false) = context (complete meaning p.module true) := by
  have same : complete meaning p.module false = complete meaning p.module true := by
    funext name inputs before outputs after
    exact propext (checked_complete accepted meaning name inputs before outputs after)
  rw [same]

end Merlin.GraphLoopsShape.Model
