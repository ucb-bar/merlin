/- Concrete view/heap interpretation for shape normalization. Heap addresses are
logical element-cell addresses; native descriptor/byte ABI conformance is a
separate local contract. Allocation/other actions may remain partial/observable. -/
import Merlin.GraphLoopsShapeModule

set_option autoImplicit false

namespace Merlin.GraphLoopsShape

abbrev MemoryWorld (Value Public : Type) := Merlin.GraphLoopsReuse.Memory Value Public

inductive MemoryOpcode where
  | load
  | store
  | loop
  | other
  deriving DecidableEq, Repr

/-- Actual allocation identity, base, offset and strides determine the read.
Descriptor conformance is carried by the typed runtime view, not guessed. -/
def Load {Value Public : Type} (inputs : List (Runtime Value)) (before : MemoryWorld Value Public)
    (outputs : List (Runtime Value)) (after : MemoryWorld Value Public) : Prop :=
  ∃ type descriptor valid coordinates value,
    inputs = Runtime.view type descriptor valid :: coordinates.map Runtime.index ∧
    Merlin.GraphLoopsReuse.Geometry.Coordinate descriptor.shape coordinates ∧
    before.heap descriptor.allocation (descriptor.alignedBase + descriptor.offset + linearOffset descriptor.strides coordinates) = some value ∧
    outputs = [Runtime.scalar type.element value] ∧ after = before

def Store {Value Public : Type} (inputs : List (Runtime Value)) (before : MemoryWorld Value Public)
    (outputs : List (Runtime Value)) (after : MemoryWorld Value Public) : Prop :=
  ∃ type descriptor valid coordinates value,
    inputs = Runtime.scalar type.element value :: Runtime.view type descriptor valid :: coordinates.map Runtime.index ∧
    Merlin.GraphLoopsReuse.Geometry.Coordinate descriptor.shape coordinates ∧ outputs = [] ∧
    after = ⟨before.observed, Merlin.GraphLoopsReuse.replaceCell before.heap descriptor.allocation
      (descriptor.alignedBase + descriptor.offset + linearOffset descriptor.strides coordinates) value⟩

def loopBody {Value World : Type} (body : RegionMeaning Value World) (index : Nat)
    (before after : List (Runtime Value) × World) : Prop := body (.index index :: before.1) before.2 after.1 after.2

/-- SCF executes the same ordered body relation, forwarding EVERY carry and the
entire memory/public world. No iteration or tensor-size expansion occurs here. -/
def SCF {Value World : Type} (body : RegionMeaning Value World) (inputs : List (Runtime Value))
    (before : World) (outputs : List (Runtime Value)) (after : World) : Prop :=
  ∃ lower upper step carried, inputs = .index lower :: .index upper :: .index step :: carried ∧ 0 < step ∧
    ∃ trace, Merlin.GraphLoopsSegment.Execution (loopBody body) upper step lower (carried, before) trace (outputs, after)

def withMemory {Value Public : Type} (base : Interpretation Value (MemoryWorld Value Public))
    (classify : String → MemoryOpcode) : Interpretation Value (MemoryWorld Value Public) where
  action := fun key inputs before outputs after => match classify key with
    | .load => Load inputs before outputs after
    | .store => Store inputs before outputs after
    | _ => base.action key inputs before outputs after
  region := fun key inputs body before outputs after => match classify key with
    | .loop => SCF body inputs before outputs after
    | _ => base.region key inputs body before outputs after

/-- This specialization preserves the exact heap and public world of actual
loads/stores/SCF plus arbitrary shared partial/observable surrounding actions. -/
theorem Module.checked_memory {Value Public : Type} (module : Module) (bits : Nat) (valid : module.Valid bits)
    (base : Interpretation Value (MemoryWorld Value Public)) (classify : String → MemoryOpcode)
    (name : String) (inputs : List (Runtime Value)) (before : MemoryWorld Value Public)
    (outputs : List (Runtime Value)) (after : MemoryWorld Value Public) :
    complete (withMemory base classify) module false name inputs before outputs after ↔
      complete (withMemory base classify) module true name inputs before outputs after :=
  Module.checked_complete module bits valid (withMemory base classify) name inputs before outputs after

/-- Separate zero-extent domain: no coordinate of a shape containing zero is
in-bounds. Metadata rewrites need no positive-buffer or element-read premise. -/
theorem zero_has_no_coordinate {shape coordinates : List Nat} (zero : 0 ∈ shape) :
    ¬ Merlin.GraphLoopsReuse.Geometry.Coordinate shape coordinates := by
  intro valid
  induction shape generalizing coordinates with
  | nil => cases zero
  | cons extent rest induction =>
    cases coordinates with
    | nil => cases valid
    | cons index indices =>
      rcases List.mem_cons.mp zero with equal | member
      · subst extent
        exact Nat.not_lt_zero index valid.1
      · exact induction member valid.2

end Merlin.GraphLoopsShape
