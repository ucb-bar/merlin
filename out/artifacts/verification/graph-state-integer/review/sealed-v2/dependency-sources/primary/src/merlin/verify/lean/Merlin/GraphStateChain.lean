/- Actual schedule composition has two separate routes. The direct route
rechecks the complete original/final boundary using GraphState.check_sound.
This module supplies a supported semantic transport route: finite checked
bijections, fixed physical cells, and action conjugacy DERIVED from a common
total primitive interpretation with ordered operand/result slots. Actual typed
IR projection and the primitive interpretation's operational fidelity remain
explicit host premises. This does not identify arbitrary Action models. -/
import Merlin.GraphStateModels
import Init.Data.List.Find

namespace Merlin.GraphState.Chain
open StateEffects
set_option autoImplicit false

def rename (mapping : List Nat) (key : Nat) : Nat := mapping[key]?.getD key

structure Renaming where
  forward : List Nat
  backward : List Nat
  physical : List Nat
  deriving DecidableEq, Repr

def valid (mapping : Renaming) : Prop :=
  mapping.forward.length = mapping.backward.length ∧
  (∀ key ∈ List.range mapping.forward.length,
    rename mapping.backward (rename mapping.forward key) = key ∧
    rename mapping.forward (rename mapping.backward key) = key) ∧
  (∀ key ∈ mapping.physical, rename mapping.forward key = key)

instance (mapping : Renaming) : Decidable (valid mapping) := by unfold valid; infer_instance

def checkRenaming (mapping : Renaming) : Bool := decide (valid mapping)

theorem rename_outside (mapping : List Nat) (key : Nat) (outside : mapping.length ≤ key) :
    rename mapping key = key := by
  simp [rename,List.getElem?_eq_none outside]

theorem inverse_forward {mapping : Renaming} (accepted : checkRenaming mapping = true) (key : Nat) :
    rename mapping.backward (rename mapping.forward key) = key := by
  have checked : valid mapping := of_decide_eq_true accepted
  by_cases inside : key < mapping.forward.length
  · exact (checked.2.1 key (List.mem_range.mpr inside)).1
  · have outside := Nat.le_of_not_gt inside
    rw [rename_outside mapping.forward key outside]
    exact rename_outside mapping.backward key (by simpa only [←checked.1] using outside)

theorem inverse_backward {mapping : Renaming} (accepted : checkRenaming mapping = true) (key : Nat) :
    rename mapping.forward (rename mapping.backward key) = key := by
  have checked : valid mapping := of_decide_eq_true accepted
  by_cases inside : key < mapping.forward.length
  · exact (checked.2.1 key (List.mem_range.mpr inside)).2
  · have outside := Nat.le_of_not_gt inside
    rw [rename_outside mapping.backward key (by simpa only [←checked.1] using outside)]
    exact rename_outside mapping.forward key outside

theorem forward_injective {mapping : Renaming} (accepted : checkRenaming mapping = true) :
    Function.Injective (rename mapping.forward) := by
  intro first second equal
  calc first = rename mapping.backward (rename mapping.forward first) := (inverse_forward accepted first).symm
       _ = rename mapping.backward (rename mapping.forward second) := congrArg _ equal
       _ = second := inverse_forward accepted second

def pull {Value : Type} (mapping : Nat → Nat) (state : State Value) : State Value := fun key => state (mapping key)

theorem pull_inverse {mapping : Renaming} (accepted : checkRenaming mapping = true)
    {Value : Type} (state : State Value) :
    pull (rename mapping.forward) (pull (rename mapping.backward) state) = state := by
  funext key
  exact congrArg state (inverse_forward accepted key)

theorem physical_observation {mapping : Renaming} (accepted : checkRenaming mapping = true)
    {Value : Type} (state : State Value) (key : Nat) (physical : key ∈ mapping.physical) :
    pull (rename mapping.forward) state key = state key := by
  have checked : valid mapping := of_decide_eq_true accepted
  exact congrArg state (checked.2.2 key physical)

/-- Core includes actual opcode/types/metadata and stable physical slot roles.
The ordered SSA addresses are separate, so operand order is not footprint sort
order. A common interpreter is arbitrary, total and pure at this boundary. -/
structure Core where
  operation : String
  operandTypes : List String
  resultTypes : List String
  attributes : String
  physicalInputs : List String
  physicalOutputs : List String
  deriving DecidableEq, Repr

structure Primitive where
  core : Core
  inputs : List Nat
  outputs : List Nat
  deriving DecidableEq, Repr

def reindex (mapping : Nat → Nat) (primitive : Primitive) : Primitive :=
  {primitive with inputs := primitive.inputs.map mapping,outputs := primitive.outputs.map mapping}

def step {Value : Type} (interpret : Core → List Value → Nat → Value)
    (primitive : Primitive) (state : State Value) : State Value := fun key =>
  match primitive.outputs.idxOf? key with
  | none => state key
  | some slot => interpret primitive.core (primitive.inputs.map state) slot

def action {Value : Type} (interpret : Core → List Value → Nat → Value)
    (primitive : Primitive) : Action ⟨primitive.inputs,primitive.outputs⟩ Value where
  step := step interpret primitive
  frame := by
    intro state key outside
    simp only [step,List.idxOf?_eq_none_iff.mpr outside]
  locality := by
    intro first second agree key inside
    have inputsEqual : primitive.inputs.map first = primitive.inputs.map second := by
      apply List.map_congr_left
      exact agree
    cases found : primitive.outputs.idxOf? key with
    | none => exact False.elim ((List.idxOf?_eq_none_iff.mp found) inside)
    | some slot => simp only [step,found,inputsEqual]

theorem idxOf_map_injective (mapping : Nat → Nat) (injective : Function.Injective mapping)
    (keys : List Nat) (key : Nat) : (keys.map mapping).idxOf? (mapping key) = keys.idxOf? key := by
  induction keys with
  | nil => rfl
  | cons head tail ih =>
    by_cases same : head = key
    · subst head; simp [List.idxOf?_cons]
    · have different : mapping head ≠ mapping key := fun equal => same (injective equal)
      simp only [List.map_cons,List.idxOf?_cons,beq_eq_false_iff_ne.mpr same,
        beq_eq_false_iff_ne.mpr different,Bool.false_eq_true,ite_false,ih]

/-- Conjugacy is a theorem about the SAME primitive interpreter and actual
ordered mapped slots, not an assumption equating arbitrary stage functions. -/
theorem step_transport {Value : Type} (interpret : Core → List Value → Nat → Value)
    (mapping : Nat → Nat) (injective : Function.Injective mapping) (primitive : Primitive)
    (state : State Value) :
    pull mapping (step interpret (reindex mapping primitive) state) =
      step interpret primitive (pull mapping state) := by
  funext key
  simp only [pull,step,reindex,idxOf_map_injective mapping injective,List.map_map,Function.comp_def]
  rfl

theorem execute_transport {Value : Type} (interpret : Core → List Value → Nat → Value)
    (mapping : Nat → Nat) (injective : Function.Injective mapping) (primitives : List Primitive) :
    ∀ state, pull mapping (execute ((primitives.map (reindex mapping)).map (step interpret)) state) =
      execute (primitives.map (step interpret)) (pull mapping state) := by
  induction primitives with
  | nil => intros; rfl
  | cons primitive rest ih =>
    intro state
    simpa only [List.map_cons,execute,List.foldl_cons,step_transport interpret mapping injective] using
      ih (step interpret (reindex mapping primitive) state)

/-- Source-relative logical SSA identity includes its actual type and slot.
Physical resource/observer identity is exact and cannot be treated as a temporary. -/
inductive Cell where
  | temporary (logicalIdentity type : String)
  | argument (logicalIdentity type : String)
  | physical (identity : String)
  deriving DecidableEq, Repr

def physicalFixed (mapping : Renaming) (source : List Cell) (key : Nat) : Bool :=
  match source[key]? with
  | some (.physical _) => decide (key ∈ mapping.physical ∧ rename mapping.forward key = key)
  | _ => true

def checkCells (mapping : Renaming) (source target : List Cell) : Bool :=
  decide (source.length = mapping.forward.length ∧ target.length = source.length ∧
    source.Nodup ∧ target.Nodup ∧
    (∀ key ∈ List.range source.length, target[rename mapping.forward key]? = source[key]?) ∧
    (∀ key ∈ List.range source.length, physicalFixed mapping source key = true))

def permute (source : List Primitive) (order : List Nat) : List Primitive :=
  order.filterMap (fun index => source[index]?)

def checkTransport (mapping : Renaming) (source target : List Primitive) (order : List Nat)
    (sourceCells targetCells : List Cell) : Bool :=
  checkRenaming mapping && checkCells mapping sourceCells targetCells &&
    decide (order.length = source.length ∧ order.Nodup ∧
      (∀ index ∈ order, index < source.length) ∧
      target = (permute source order).map (reindex (rename mapping.forward)))

/-- Acceptance binds actual target primitives to the SAME opcode/types/metadata
and ordered source-relative slots, and transports complete execution. Sorting
or equating independent footprint numbers supplies no such conclusion. -/
theorem checkTransport_sound {mapping : Renaming} {source target : List Primitive} {order : List Nat}
    {sourceCells targetCells : List Cell}
    (accepted : checkTransport mapping source target order sourceCells targetCells = true)
    {Value : Type} (interpret : Core → List Value → Nat → Value) (initial : State Value) :
    pull (rename mapping.forward) (execute (target.map (step interpret)) initial) =
      execute ((permute source order).map (step interpret)) (pull (rename mapping.forward) initial) := by
  have checks : (checkRenaming mapping = true ∧ checkCells mapping sourceCells targetCells = true) ∧
      decide (order.length = source.length ∧ order.Nodup ∧
        (∀ index ∈ order, index < source.length) ∧
        target = (permute source order).map (reindex (rename mapping.forward))) = true := by
    simpa only [checkTransport,Bool.and_eq_true] using accepted
  have same := (of_decide_eq_true checks.2).2.2.2
  rw [same]
  exact execute_transport interpret (rename mapping.forward) (forward_injective checks.1.1) _ initial

def supports (effects : Footprint) (primitive : Primitive) : Bool :=
  decide ((∀ key ∈ primitive.inputs, key ∈ effects.reads) ∧
    (∀ key ∈ primitive.outputs, key ∈ effects.writes) ∧
    (∀ key ∈ effects.writes, key ∈ primitive.outputs))

/-- Actual sorted footprints and ordered primitive slots can have different
orders. Membership coverage, including EXACT writes, is checked explicitly. -/
def actualAction {Value : Type} (interpret : Core → List Value → Nat → Value)
    (effects : Footprint) (primitive : Primitive) (accepted : supports effects primitive = true) : Action effects Value where
  step := step interpret primitive
  frame := by
    have checked := of_decide_eq_true accepted
    intro initial key outside
    exact (action interpret primitive).frame initial key (fun inside => outside (checked.2.1 key inside))
  locality := by
    have checked := of_decide_eq_true accepted
    intro first second agree key inside
    exact (action interpret primitive).locality first second
      (fun read inReads => agree read (checked.1 read inReads)) key (checked.2.2 key inside)

def checkPrimitives : List Footprint → List Primitive → Bool
  | [],[] => true
  | effects::rest,primitive::tail => supports effects primitive && checkPrimitives rest tail
  | _,_ => false

def run {Value : Type} (interpret : Core → List Value → Nat → Value)
    (primitives : List Primitive) (index : Nat) : State Value → State Value :=
  match primitives[index]? with
  | some primitive => step interpret primitive
  | none => id

/-- Totality, locality and complete frames are DERIVED for the common primitive
model. Its fidelity to actual instructions remains a separate operational premise. -/
theorem checkPrimitives_models {Value : Type} (interpret : Core → List Value → Nat → Value)
    {footprints : List Footprint} {primitives : List Primitive}
    (accepted : checkPrimitives footprints primitives = true) : Models footprints (run interpret primitives) := by
  induction footprints generalizing primitives with
  | nil => intro index effects impossible; simp at impossible
  | cons effects rest ih =>
    cases primitives with
    | nil => cases accepted
    | cons primitive tail =>
      have parts : supports effects primitive = true ∧ checkPrimitives rest tail = true := by
        simpa only [checkPrimitives,Bool.and_eq_true] using accepted
      intro index actual found
      cases index with
      | zero =>
        have same : effects = actual := by simpa using found
        subst actual
        exact ⟨actualAction interpret effects primitive parts.1,rfl⟩
      | succ index =>
        obtain ⟨actual,stepEqual⟩ := ih parts.2 index actual (by simpa using found)
        exact ⟨actual,by simpa only [run,List.getElem?_cons_succ] using stepEqual⟩

#print axioms forward_injective
#print axioms pull_inverse
#print axioms physical_observation
#print axioms action
#print axioms step_transport
#print axioms execute_transport
#print axioms checkTransport_sound
#print axioms checkPrimitives_models
end Merlin.GraphState.Chain
