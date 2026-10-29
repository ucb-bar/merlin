/- Typed logical-epoch to physical-slot register relation. UNCOMPILED.
Only private allocation labels change; offsets, strides, shapes, element kinds,
address spaces, scalar/index values and caller allocation identities remain.
The physical heap already belongs to the old candidate execution; this module
does not rename that heap or assume whole-program equality. -/
import Merlin.GraphLoopsMemorySCFAfterInitializer
import Merlin.GraphLoopsShapeEpochState

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

def slotAllocation (objects : GraphLoopsReuse.Objects) (allocation : Nat) : Nat :=
  if allocation < objects.length then GraphLoopsReuse.slotOf objects allocation else allocation

def slotRuntime {Value : Type} (objects : GraphLoopsReuse.Objects) : Runtime Value → Runtime Value
  | .index value => .index value
  | .scalar element value => .scalar element value
  | .view type descriptor valid =>
      .view type { descriptor with allocation := slotAllocation objects descriptor.allocation } valid

def slotEnv {Value : Type} (objects : GraphLoopsReuse.Objects) (env : GraphLoopsShape.Env Value) : GraphLoopsShape.Env Value :=
  fun name => slotRuntime objects (env name)

theorem slot_kind {Value : Type} (objects : GraphLoopsReuse.Objects) (value : Runtime Value) :
    (slotRuntime objects value).kind = value.kind := by cases value <;> rfl

theorem slot_holds {Value : Type} (objects : GraphLoopsReuse.Objects) (binding : Binding)
    (value : Runtime Value) (holds : binding.Holds value) : binding.Holds (slotRuntime objects value) := by
  refine ⟨(slot_kind objects value).trans holds.1, ?_⟩
  intro index known
  rw [holds.2 index known]
  rfl

theorem slot_typed {Value : Type} (objects : GraphLoopsReuse.Objects) (context : Context)
    (env : GraphLoopsShape.Env Value) (typed : Typed context env) : Typed context (slotEnv objects env) :=
  fun binding member => slot_holds objects binding (env binding.name) (typed binding member)

theorem slot_put {Value : Type} (objects : GraphLoopsReuse.Objects) (env : GraphLoopsShape.Env Value)
    (name : Nat) (value : Runtime Value) :
    slotEnv objects (GraphLoopsShape.put env name value) = GraphLoopsShape.put (slotEnv objects env) name (slotRuntime objects value) := by
  funext other
  by_cases equal : other = name <;> simp only [slotEnv, GraphLoopsShape.put, equal, ↓reduceIte]

theorem slot_assign {Value : Type} (objects : GraphLoopsReuse.Objects) (env : GraphLoopsShape.Env Value)
    (bindings : List Binding) (values : List (Runtime Value)) :
    slotEnv objects (GraphLoopsShape.assign env bindings values) =
      GraphLoopsShape.assign (slotEnv objects env) bindings (values.map (slotRuntime objects)) := by
  induction bindings generalizing env values with
  | nil => rfl
  | cons binding rest induction =>
    cases values with
    | nil => rfl
    | cons value tail =>
      simp only [GraphLoopsShape.assign, List.map_cons]
      rw [induction, slot_put]

theorem slot_arguments {Value : Type} (objects : GraphLoopsReuse.Objects) (env : GraphLoopsShape.Env Value)
    (bindings : List Binding) (aliases : Aliases) :
    readArguments false aliases (slotEnv objects env) bindings =
      (readArguments false aliases env bindings).map (slotRuntime objects) := by
  simp only [readArguments, List.map_map, Function.comp_def, slotEnv]

theorem slot_private_view {Value : Type} (objects : GraphLoopsReuse.Objects) (object : Nat)
    (inside : object < objects.length) (shape : List Nat) (element : String) :
    slotRuntime objects (Epoch.view (Value := Value) object shape element) =
      Epoch.view (GraphLoopsReuse.slotOf objects object) shape element := by
  simp only [Epoch.view, Epoch.descriptor, slotRuntime, slotAllocation, inside, ↓reduceIte]

theorem slot_caller_view {Value : Type} (objects : GraphLoopsReuse.Objects) (object : Nat)
    (outside : ¬ object < objects.length) (shape : List Nat) (element : String) :
    slotRuntime objects (Epoch.view (Value := Value) object shape element) = Epoch.view object shape element := by
  simp only [Epoch.view, Epoch.descriptor, slotRuntime, slotAllocation, outside, ↓reduceIte]

/-- Initializer scalar/index capture erasure commutes with the descriptor map.
Thus the old initializer environment is preserved, not separately assumed. -/
theorem erase_slot {Value : Type} (objects : GraphLoopsReuse.Objects) (value : Runtime Value) :
    Epoch.erase (slotRuntime objects value) = Epoch.erase value := by cases value <;> rfl

theorem erase_slot_env {Value : Type} (objects : GraphLoopsReuse.Objects) (env : GraphLoopsShape.Env Value) :
    Epoch.eraseEnv (slotEnv objects env) = Epoch.eraseEnv env := by
  funext name
  exact erase_slot objects (env name)

/-- Reused views can be introduced by an identity cast in an expanded typed
target, then deleted by the existing proved normalizer. This permits actual
raw-SSA equality under a bijection, without assuming a noninjective name map. -/
theorem alias_same_slot {Value : Type} (objects : GraphLoopsReuse.Objects) (first second : Nat)
    (firstInside : first < objects.length) (secondInside : second < objects.length)
    (same : GraphLoopsReuse.slotOf objects first = GraphLoopsReuse.slotOf objects second)
    (shape : List Nat) (element : String) :
    slotRuntime objects (Epoch.view (Value := Value) first shape element) =
      slotRuntime objects (Epoch.view second shape element) := by
  rw [slot_private_view objects first firstInside, slot_private_view objects second secondInside, same]

end Merlin.GraphLoopsMemorySCF.After
