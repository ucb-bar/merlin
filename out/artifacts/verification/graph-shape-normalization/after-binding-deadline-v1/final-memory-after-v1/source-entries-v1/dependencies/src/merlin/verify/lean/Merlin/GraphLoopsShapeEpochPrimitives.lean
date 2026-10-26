/- A single explicit interpretation for retained epoch instructions. Each key
selects checked typed SSA operands/results; callers see caller memory only.
Arithmetic stays a shared partial binary relation. Initializers use the existing
operational SCF interpretation instead of an opaque initializer primitive. -/
import Merlin.GraphLoopsShapeEpochAllocation
import Merlin.GraphLoopsShapeEpochObserver
import Merlin.GraphLoopsShapeEpochSequence

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

structure Access where
  object : Nat
  buffer : Binding
  indices : List Binding
  shape : List Nat
  element : String
  deriving DecidableEq, Repr

inductive Spec where
  | constant (result : Binding) (value : Nat)
  | binary (operation : String) (left right result : Binding) (element : String)
  | load (access : Access) (result : Binding)
  | store (access : Access) (value : Binding)
  | observer (operation : String) (input buffer result : Binding) (element : String)
  | returned (values : List Binding)
  deriving DecidableEq, Repr

abbrev Specifications := List (String × Spec)

def specAt : Specifications → String → Option Spec
  | [], _ => none
  | (key, spec) :: rest, actual => if actual = key then some spec else specAt rest actual

abbrev BinaryMeaning (Value : Type) := String → Value → Value → Value → Prop

def writeRegister {Value User : Type} (state : PublicState Value User) (name : Nat) (value : Runtime Value) :
    PublicState Value User := { state with registers := put state.registers name value }

def ReadCoordinates {Value : Type} (access : Access) (env : Env Value) (coordinates : List Nat) : Prop :=
  env access.buffer.name = view access.object access.shape access.element ∧
    readArguments false [] env access.indices = coordinates.map Runtime.index ∧
    GraphLoopsReuse.Geometry.Coordinate access.shape coordinates

/-- These predicates are formulas over actual operand reads, not summaries
postulated about the resulting whole program. Scalar result typing is explicit. -/
def Retained {Value User : Type} (bound : Nat) (scalar : BinaryMeaning Value) (spec : Spec)
    (before after : PublicState Value User) : Prop :=
  match spec with
  | .constant result value => after = writeRegister before result.name (.index value)
  | .binary key left right result element => ∃ a b value,
      before.registers left.name = .scalar element a ∧ before.registers right.name = .scalar element b ∧
      scalar key a b value ∧ result.Holds (.scalar element value) ∧
      after = writeRegister before result.name (.scalar element value)
  | .observer key input buffer result element => ∃ object value loaded updated,
      ¬ object < bound ∧ before.registers input.name = .scalar element value ∧
      before.registers buffer.name = view object [1] element ∧ before.callerHeap object 0 = some loaded ∧
      scalar key loaded value updated ∧ result.Holds (.scalar element updated) ∧
      after = { writeRegister before result.name (.scalar element updated) with
        callerHeap := GraphLoopsReuse.replaceCell before.callerHeap object 0 updated }
  | .returned values => after = { before with returned := readArguments false [] before.registers values }
  | _ => False

def primitives {Value User : Type} (bound : Nat) (specifications : Specifications) (scalar : BinaryMeaning Value) :
    GraphLoopsReuse.Primitives Value (PublicState Value User) where
  initialization := fun _ _ _ => False
  address := fun key before index => ∃ access result coordinates,
    specAt specifications key = some (.load access result) ∧ ReadCoordinates access before.registers coordinates ∧
      index = GraphLoopsReuse.Geometry.flatten access.shape coordinates
  load := fun key value before after => ∃ access result,
    specAt specifications key = some (.load access result) ∧ result.Holds (.scalar access.element value) ∧
      after = writeRegister before result.name (.scalar access.element value)
  store := fun key before index value => ∃ access operand coordinates,
    specAt specifications key = some (.store access operand) ∧ ReadCoordinates access before.registers coordinates ∧
      before.registers operand.name = .scalar access.element value ∧
      index = GraphLoopsReuse.Geometry.flatten access.shape coordinates
  dimension := fun _ _ _ _ => False
  retained := fun key before after => ∃ spec, specAt specifications key = some spec ∧ Retained bound scalar spec before after

def interpretation {Value User : Type} (bound : Nat) (specifications : Specifications)
    (mapping : Aliases) (scalar : BinaryMeaning Value) : GraphLoopsMemorySCF.Interpretation Value (PublicState Value User) where
  pure := fun _ _ _ => False
  captures := fun state => eraseMapped (resolve mapping) state.registers
  retained := primitives bound specifications scalar

/-- No numeric totality follows from this model: a refused scalar operation
refuses the retained instruction even if its operands and output type are valid. -/
theorem binary_requires_success {Value User : Type} (bound : Nat) (scalar : BinaryMeaning Value)
    (key : String) (left right result : Binding) (element : String) (before after : PublicState Value User)
    (run : Retained bound scalar (.binary key left right result element) before after) :
    ∃ a b value, scalar key a b value := by
  obtain ⟨a, b, value, _, _, computed, _, _⟩ := run
  exact ⟨a, b, value, computed⟩

theorem constant_register {Value User : Type} (bound : Nat) (specifications : Specifications)
    (scalar : BinaryMeaning Value) (mapping : Aliases) (key : String) (result : Binding) (value : Nat)
    (selected : specAt specifications key = some (.constant result value))
    (objects : GraphLoopsReuse.Objects) (configs : List GraphLoopsMemorySCF.Configuration)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User)) :
    GraphLoopsMemorySCF.Step objects configs (interpretation bound specifications mapping scalar) false false (.retained key)
      before ⟨writeRegister before.observed result.name (.index value), before.heap⟩ := by
  apply GraphLoopsMemorySCF.Step.ordinary (by trivial)
  apply GraphLoopsReuse.Step.retained
  exact ⟨.constant result value, selected, rfl⟩

/-- The constant's old retained instruction has no hidden alternative effect. -/
theorem constant_step_iff {Value User : Type} (bound : Nat) (specifications : Specifications)
    (scalar : BinaryMeaning Value) (mapping : Aliases) (key : String) (result : Binding) (value : Nat)
    (selected : specAt specifications key = some (.constant result value))
    (objects : GraphLoopsReuse.Objects) (configs : List GraphLoopsMemorySCF.Configuration)
    (before after : GraphLoopsReuse.Memory Value (PublicState Value User)) :
    GraphLoopsMemorySCF.Step objects configs (interpretation bound specifications mapping scalar) false false (.retained key)
      before after ↔ after = ⟨writeRegister before.observed result.name (.index value), before.heap⟩ := by
  constructor
  · intro run
    cases run with
    | ordinary ordinary run =>
      cases run with
      | retained key initial observed action =>
        obtain ⟨spec, found, action⟩ := action
        have same : spec = .constant result value := Option.some.inj (found.symm.trans selected)
        subst spec
        change observed = writeRegister before.observed result.name (.index value) at action
        cases action
        rfl
  · rintro rfl
    exact constant_register bound specifications scalar mapping key result value selected objects configs before

end Merlin.GraphLoopsShape.Epoch
