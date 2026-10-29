/- UNCOMPILED successor. Constructorwise public frame laws. No arbitrary opaque
operation is assumed unable to inspect private state; only the listed concrete
partial primitives have this property. Allocation/release/calls are excluded. -/
import Merlin.GraphLoopsMemorySCFAfterContextFrame

set_option autoImplicit false
namespace Merlin.GraphLoopsMemorySCF.After
open GraphLoopsShape

inductive PublicOpcode where
  | binary (element : String)
  | literal (element : String)
  | index (opcode : IndexOpcode)
  | load
  | store
  deriving DecidableEq, Repr

def publicOpcode (indices : IndexOpcodes) (opcodes : Epoch.Opcodes) (literals : LiteralOpcodes)
    (key : String) : Option PublicOpcode :=
  match literalAt literals key with
  | some element => some (.literal element)
  | none => match Epoch.memoryOpcode opcodes key with
    | .load => some .load
    | .store => some .store
    | _ => match indexOpcodeAt indices key with
      | some opcode => some (.index opcode)
      | none => match Epoch.opcodeAt opcodes key with
        | some (.binary element) => some (.binary element)
        | _ => none

def PublicOpcode.action {Value User : Type} (opcode : PublicOpcode) (bits : Nat) (key : String)
    (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value) :
    RegionMeaning Value (CallerWorld Value User) :=
  match opcode with
  | .binary element => Epoch.Binary element (scalar key)
  | .literal element => LiteralAction element key literal
  | .index opcode => IndexAction bits opcode
  | .load => Load
  | .store => Store

theorem public_action_selected {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (opcodes : Epoch.Opcodes) (literals : LiteralOpcodes) (key : String) (opcode : PublicOpcode)
    (selected : publicOpcode indices opcodes literals key = some opcode)
    (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value) :
    (entryMeaning (User := User) bits indices opcodes scalar literals literal).action key =
      opcode.action bits key scalar literal := by
  cases literalFound : literalAt literals key with
  | some element =>
    have chosen : opcode = .literal element := by
      simpa only [publicOpcode, literalFound, Option.some.injEq] using selected.symm
    subst opcode
    simp only [entryMeaning, withLiterals, literalFound, PublicOpcode.action]
  | none =>
    cases old : Epoch.opcodeAt opcodes key with
    | none =>
      cases generated : indexOpcodeAt indices key <;>
        simp_all [publicOpcode, Epoch.memoryOpcode, entryMeaning, withLiterals, finalMeaning,
          withMemory, withIndex, Epoch.scalarAllocationBase, PublicOpcode.action]
    | some oldOpcode =>
      cases oldOpcode <;> cases generated : indexOpcodeAt indices key <;>
        simp_all [publicOpcode, Epoch.memoryOpcode, entryMeaning, withLiterals, finalMeaning,
          withMemory, withIndex, Epoch.scalarAllocationBase, PublicOpcode.action]

/-- Same explicit arguments/results, related public heaps; private clocks and
private cells are unconstrained. Outputs are matched, not assumed deterministic.
-/
theorem PublicOpcode.transfer {Value User : Type} (opcode : PublicOpcode) (bits bound : Nat) (key : String)
    (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value)
    (inputs : List (Runtime Value)) (outside : ExternalValues bound inputs)
    (before other after : CallerWorld Value User) (same : PublicRelated bound before other)
    (outputs : List (Runtime Value)) (run : opcode.action bits key scalar literal inputs before outputs after) :
    ∃ final, opcode.action bits key scalar literal inputs other outputs final ∧ PublicRelated bound after final := by
  cases opcode with
  | binary element =>
    obtain ⟨left, right, result, arguments, computed, returned, rfl⟩ := run
    exact ⟨other, ⟨left, right, result, arguments, computed, returned, rfl⟩, same⟩
  | literal element =>
    obtain ⟨arguments, value, computed, returned, rfl⟩ := run
    exact ⟨other, ⟨arguments, value, computed, returned, rfl⟩, same⟩
  | index opcode =>
    cases inputs with
    | nil => exact False.elim run
    | cons first remaining =>
      cases remaining with
      | nil => exact False.elim run
      | cons second rest =>
        cases rest with
        | cons => exact False.elim run
        | nil =>
          cases first <;> cases second
          all_goals first | exact False.elim run | skip
          obtain ⟨leftBound, rightBound, resultBound, returned, rfl⟩ := run
          exact ⟨other, ⟨leftBound, rightBound, resultBound, returned, rfl⟩, same⟩
  | load =>
    obtain ⟨type, descriptor, valid, coordinates, value, arguments, inside, loaded, returned, rfl⟩ := run
    have external : ¬ descriptor.allocation < bound := outside (.view type descriptor valid) (by rw [arguments]; simp)
    refine ⟨other, ⟨type, descriptor, valid, coordinates, value, arguments, inside, ?_, returned, rfl⟩, same⟩
    rw [← same.read descriptor.allocation _ external]
    exact loaded
  | store =>
    obtain ⟨type, descriptor, valid, coordinates, value, arguments, inside, returned, rfl⟩ := run
    refine ⟨⟨other.observed, GraphLoopsReuse.replaceCell other.heap descriptor.allocation
      (descriptor.alignedBase + descriptor.offset + linearOffset descriptor.strides coordinates) value⟩,
      ⟨type, descriptor, valid, coordinates, value, arguments, inside, returned, rfl⟩, ?_⟩
    exact same.store descriptor.allocation _ value

end Merlin.GraphLoopsMemorySCF.After
