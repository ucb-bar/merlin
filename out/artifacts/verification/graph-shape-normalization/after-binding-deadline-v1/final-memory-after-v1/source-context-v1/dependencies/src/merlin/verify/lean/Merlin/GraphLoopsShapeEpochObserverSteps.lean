/- A retained call is connected to the actual internal caller-observer body,
then to the old public-state instruction. No whole-call equality is assumed. -/
import Merlin.GraphLoopsShapeEpochScalarSteps
import Merlin.GraphLoopsShapeEpochCallers

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

theorem view_object_injective {Value : Type} (first second : Nat) (shape : List Nat) (element : String)
    (same : (view first shape element : Runtime Value) = view second shape element) : first = second := by
  have descriptors := (Runtime.view.inj same).2
  exact congrArg Descriptor.allocation descriptors

theorem clock_caller_store {Value User : Type} (bound next object index : Nat) (value : Value)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User)) (outside : ¬ object < bound) :
    clockWorld bound next { before with observed := { before.observed with
      callerHeap := GraphLoopsReuse.replaceCell before.observed.callerHeap object index value } } =
      ⟨(clockWorld bound next before).observed,
        GraphLoopsReuse.replaceCell (clockWorld bound next before).heap object index value⟩ :=
  caller_store_world bound object index value (atClock next before) outside

/-- Every positive callee depth interprets the checked concrete observer body.
Its loaded input/order, caller-cell update and typed returned result are exact. -/
theorem observer_atom {Value User : Type} (bound next : Nat) (specifications : Specifications)
    (scalar : BinaryMeaning Value) (mapping : Aliases) (key callKey name : String)
    (input buffer result : Binding) (program : Touch) (checked : decide program.Facts = true)
    (selected : specAt specifications key = some (.observer program.binaryKey input buffer result program.element))
    (module : Module) (found : functionAt module name = some (program.function name))
    (separate : program.input.name ≠ program.buffer.name)
    (loadNotCall : callTarget module.calls program.loadKey = none)
    (binaryNotCall : callTarget module.calls program.binaryKey = none)
    (storeNotCall : callTarget module.calls program.storeKey = none)
    (base : Interpretation Value (MemoryWorld Value (AllocationPublic User)))
    (load : base.action program.loadKey = Load)
    (binary : base.action program.binaryKey = Binary program.element (scalar program.binaryKey))
    (store : base.action program.storeKey = Store)
    (meaning : Interpretation Value (MemoryWorld Value (AllocationPublic User))) (depth : Nat)
    (callee : meaning.action callKey = GraphLoopsShape.invoke base module false (depth + 1) name)
    (objects : GraphLoopsReuse.Objects) (configs : List GraphLoopsMemorySCF.Configuration)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User))
    (object : Nat) (outside : ¬ object < bound) (value : Value)
    (inputRead : before.observed.registers input.name = .scalar program.element value)
    (bufferRead : before.observed.registers buffer.name = view object [1] program.element)
    (env : Env Value) (world : MemoryWorld Value (AllocationPublic User)) :
    (Atom.action callKey [input, buffer] [result]).step meaning before.observed.registers
      (clockWorld bound next before) env world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs (interpretation bound specifications mapping scalar) false false
        (.retained key) before after ∧ env = after.observed.registers ∧ world = clockWorld bound next after := by
  have arguments : readArguments false [] before.observed.registers [input, buffer] =
      [.scalar program.element value, view object [1] program.element] := by
    change [before.observed.registers input.name, before.observed.registers buffer.name] = _
    rw [inputRead, bufferRead]
  simp only [Atom.step, callee, arguments,
    program.call checked module name found separate loadNotCall binaryNotCall storeNotCall base
      (scalar program.binaryKey) load binary store depth object value,
    retained_step_iff bound specifications scalar mapping key _ selected objects configs]
  constructor
  · rintro ⟨values, ⟨loaded, updated, read, computed, rfl, world⟩, typed, env⟩
    have callerRead : before.observed.callerHeap object 0 = some loaded :=
      (read_caller bound object 0 before.heap before.observed.callerHeap outside).symm.trans read
    have holds : result.Holds (.scalar program.element updated) := by cases typed with | cons holds tail => exact holds
    let observed : PublicState Value User :=
      { writeRegister before.observed result.name (.scalar program.element updated) with
        callerHeap := GraphLoopsReuse.replaceCell before.observed.callerHeap object 0 updated }
    refine ⟨⟨observed, before.heap⟩, ⟨observed, ?_, rfl⟩, env, ?_⟩
    · exact ⟨object, value, loaded, updated, outside, inputRead, bufferRead, callerRead, computed, holds, rfl⟩
    · exact world.trans (clock_caller_store bound next object 0 updated before outside).symm
  · rintro ⟨after, ⟨observed, ⟨actualObject, actualValue, loaded, updated, actualOutside, actualInput, actualBuffer,
      read, computed, holds, rfl⟩, rfl⟩, rfl, rfl⟩
    have objectSame : actualObject = object := view_object_injective actualObject object [1] program.element
      (actualBuffer.symm.trans bufferRead)
    have valueSame : actualValue = value := (Runtime.scalar.inj (actualInput.symm.trans inputRead)).2
    subst actualObject
    subst actualValue
    refine ⟨[.scalar program.element updated], ⟨loaded, updated, ?_, computed, rfl, ?_⟩, .cons holds .nil, rfl⟩
    · exact (read_caller bound object 0 before.heap before.observed.callerHeap outside).trans read
    · exact clock_caller_store bound next object 0 updated before outside

/-- A rejected unconditional top-level statement prevents every complete
execution, regardless of surrounding partial effects or later observations. -/
theorem sequence_refused {Value World : Type} (meaning : Interpretation Value World) (atoms : List Atom)
    (atom : Atom) (member : atom ∈ atoms)
    (refused : ∀ env before next after, ¬ atom.step meaning env before next after)
    (env finalEnv : Env Value) (before after : World) : ¬ SequenceRun meaning atoms env before finalEnv after := by
  intro run
  induction run with
  | done => cases member
  | more first suffix induction =>
    rcases List.mem_cons.mp member with rfl | later
    · exact refused _ _ _ _ first
    · exact induction later

end Merlin.GraphLoopsShape.Epoch
