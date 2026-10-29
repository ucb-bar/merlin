/- UNCOMPILED. Scalar/index results and outside-caller descriptors are invariant
under the private epoch→slot register map. Retained operations see exact inputs
and full worlds; their partial effects/numerics are never replaced by equations.
-/
import Merlin.GraphLoopsMemorySCFAfterAllocationSteps

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

def NoViewKind : Kind → Prop
  | .view _ => False
  | _ => True

def NoViewBindings (bindings : List Binding) : Prop :=
  ∀ binding ∈ bindings, NoViewKind binding.kind

instance (kind : Kind) : Decidable (NoViewKind kind) := by cases kind <;> simp only [NoViewKind] <;> infer_instance

instance (bindings : List Binding) : Decidable (NoViewBindings bindings) := by
  unfold NoViewBindings
  infer_instance

theorem scalar_slot_fixed {Value : Type} (objects : GraphLoopsReuse.Objects)
    (binding : Binding) (value : Runtime Value) (typed : binding.Holds value)
    (scalar : NoViewKind binding.kind) : slotRuntime objects value = value := by
  have kind := typed.1
  rw [← kind] at scalar
  cases value with
  | index => rfl
  | scalar => rfl
  | view => exact False.elim scalar

theorem scalar_values_fixed {Value : Type} (objects : GraphLoopsReuse.Objects)
    (bindings : List Binding) (values : List (Runtime Value)) (typed : ValuesHold bindings values)
    (scalar : NoViewBindings bindings) : values.map (slotRuntime objects) = values := by
  induction typed with
  | nil => rfl
  | @cons binding value rest remaining holds typed induction =>
    rw [List.map_cons, scalar_slot_fixed objects binding value holds (scalar binding (by simp))]
    rw [induction (fun item member => scalar item (List.mem_cons_of_mem _ member))]

/-- No private descriptor may escape through this scalar/index return domain.
View-valued returns need a separate observational relation and are not silently
identified by the noninjective slot map.
-/
theorem scalar_returns_fixed {Value : Type} (objects : GraphLoopsReuse.Objects)
    (env : GraphLoopsShape.Env Value) (returned : List Binding)
    (typed : ∀ binding ∈ returned, binding.Holds (env binding.name))
    (scalar : NoViewBindings returned) :
    readArguments false [] (slotEnv objects env) returned = readArguments false [] env returned := by
  apply List.map_congr_left
  intro binding member
  exact scalar_slot_fixed objects binding (env binding.name) (typed binding member) (scalar binding member)

theorem caller_argument_fixed {Value : Type} (objects : GraphLoopsReuse.Objects)
    (env : GraphLoopsShape.Env Value) (binding : Binding) (allocation : Nat)
    (outside : ¬ allocation < objects.length) (shape : List Nat) (element : String)
    (pinned : env binding.name = Epoch.view allocation shape element) :
    slotEnv objects env binding.name = env binding.name := by
  change slotRuntime objects (env binding.name) = env binding.name
  rw [pinned, slot_caller_view objects allocation outside]

/-- A local action relation is unchanged when its explicit arguments are
fixed and its typed results are scalar/index. ALL world effects remain in the
same action relation; no purity, totality or arithmetic law is postulated.
-/
theorem retained_action_fragment {Value World : Type}
    (objects : GraphLoopsReuse.Objects) (meaning : GraphLoopsShape.Interpretation Value World)
    (key : String) (operands results : List Binding) (env : GraphLoopsShape.Env Value)
    (fixedArguments : ∀ binding ∈ operands, slotRuntime objects (env binding.name) = env binding.name)
    (scalarResults : NoViewBindings results)
    (before : World) (next : GraphLoopsShape.Env Value) (after : World) :
    (Epoch.Atom.action key operands results).step meaning (slotEnv objects env) before next after ↔
      ∃ logicalNext, (Epoch.Atom.action key operands results).step meaning env before logicalNext after ∧
        next = slotEnv objects logicalNext := by
  have arguments : readArguments false [] (slotEnv objects env) operands = readArguments false [] env operands := by
    apply List.map_congr_left
    intro binding member
    exact fixedArguments binding member
  simp only [Epoch.Atom.step, arguments]
  constructor
  · rintro ⟨values, action, typed, assigned⟩
    refine ⟨GraphLoopsShape.assign env results values, ⟨values, action, typed, rfl⟩, ?_⟩
    rw [slot_assign, scalar_values_fixed objects results values typed scalarResults]
    exact assigned
  · rintro ⟨logicalNext, ⟨values, action, typed, rfl⟩, assigned⟩
    refine ⟨values, action, typed, ?_⟩
    simpa only [slot_assign, scalar_values_fixed objects results values typed scalarResults] using assigned

theorem constant_fragment {Value World : Type} (objects : GraphLoopsReuse.Objects)
    (meaning : GraphLoopsShape.Interpretation Value World) (result : Binding) (value : Nat)
    (env : GraphLoopsShape.Env Value) (before : World) (next : GraphLoopsShape.Env Value) (after : World) :
    (Epoch.Atom.constant result value).step meaning (slotEnv objects env) before next after ↔
      ∃ logicalNext, (Epoch.Atom.constant result value).step meaning env before logicalNext after ∧
        next = slotEnv objects logicalNext := by
  simp only [Epoch.Atom.step, slot_put, slotRuntime]
  constructor
  · rintro ⟨assigned, world⟩
    refine ⟨GraphLoopsShape.put env result.name (.index value), ⟨rfl, world⟩, ?_⟩
    simpa only [slot_put, slotRuntime] using assigned
  · rintro ⟨logicalNext, ⟨rfl, world⟩, assigned⟩
    exact ⟨by simpa only [slot_put, slotRuntime] using assigned, world⟩

end Merlin.GraphLoopsMemorySCF.After
