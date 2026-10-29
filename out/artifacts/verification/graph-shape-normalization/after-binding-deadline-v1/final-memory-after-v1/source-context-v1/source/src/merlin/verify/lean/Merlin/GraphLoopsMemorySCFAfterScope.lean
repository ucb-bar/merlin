/- UNCOMPILED. The expanded target has fresh tile/control bindings absent from
logical epoch registers. Compare scoped source bindings, not entire environments.
Future private descriptor ghosts likewise need not exist before their definition.
-/
import Merlin.GraphLoopsMemorySCFAfterObservers
import Merlin.GraphLoopsMemorySCFAfterTiled

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

def RegistersRelated {Value : Type} (objects : GraphLoopsReuse.Objects) (context : Context)
    (logical actual : GraphLoopsShape.Env Value) : Prop :=
  ∀ binding ∈ context, actual binding.name = slotRuntime objects (logical binding.name)

theorem RegistersRelated.typed {Value : Type} {objects : GraphLoopsReuse.Objects} {context : Context}
    {logical actual : GraphLoopsShape.Env Value} (related : RegistersRelated objects context logical actual)
    (typed : Typed context logical) : Typed context actual := by
  intro binding member
  rw [related binding member]
  exact slot_holds objects binding (logical binding.name) (typed binding member)

theorem RegistersRelated.scratch {Value : Type} {objects : GraphLoopsReuse.Objects} {context : Context}
    {logical actual : GraphLoopsShape.Env Value} (related : RegistersRelated objects context logical actual)
    (name : Nat) (value : Runtime Value) (fresh : name ∉ names context) :
    RegistersRelated objects context logical (GraphLoopsShape.put actual name value) := by
  intro binding member
  have different : binding.name ≠ name := fun equal => fresh (equal ▸ List.mem_map_of_mem member)
  exact (ite_eq_right different).trans (related binding member)

theorem RegistersRelated.define {Value : Type} {objects : GraphLoopsReuse.Objects} {context : Context}
    {logical actual : GraphLoopsShape.Env Value} (related : RegistersRelated objects context logical actual)
    (binding : Binding) (value : Runtime Value) (fresh : binding.name ∉ names context) :
    RegistersRelated objects (binding :: context) (GraphLoopsShape.put logical binding.name value)
      (GraphLoopsShape.put actual binding.name (slotRuntime objects value)) := by
  intro other member
  rcases List.mem_cons.mp member with rfl | prior
  · simp only [GraphLoopsShape.put, ↓reduceIte]
  · have different : other.name ≠ binding.name := fun equal => fresh (equal ▸ List.mem_map_of_mem prior)
    simp only [GraphLoopsShape.put, different, ↓reduceIte]
    exact related other prior

theorem RegistersRelated.assign {Value : Type} {objects : GraphLoopsReuse.Objects} {context : Context}
    {logical actual : GraphLoopsShape.Env Value} (related : RegistersRelated objects context logical actual)
    (bindings : List Binding) (values : List (Runtime Value)) (fresh : Fresh context bindings)
    (typed : ValuesHold bindings values) :
    RegistersRelated objects (extendContext context bindings) (GraphLoopsShape.assign logical bindings values)
      (GraphLoopsShape.assign actual bindings (values.map (slotRuntime objects))) := by
  induction typed generalizing context logical actual with
  | nil => exact related
  | @cons binding value rest remaining holds typed induction =>
    exact induction (related.define binding value fresh.1) fresh.2

/-- A logical allocation view was preseeded as a ghost in the old epoch model.
The actual alias/allocation defines its SSA name now, extending the scoped
relation without requiring a matching preexisting actual ghost.
-/
theorem RegistersRelated.define_existing {Value : Type} {objects : GraphLoopsReuse.Objects} {context : Context}
    {logical actual : GraphLoopsShape.Env Value} (related : RegistersRelated objects context logical actual)
    (binding : Binding) (value : Runtime Value) (fresh : binding.name ∉ names context)
    (existing : logical binding.name = value) :
    RegistersRelated objects (binding :: context) logical
      (GraphLoopsShape.put actual binding.name (slotRuntime objects value)) := by
  have result := related.define binding value fresh
  rw [Epoch.put_existing _ _ _ existing] at result
  exact result

theorem RegistersRelated.private_view {Value : Type} {objects : GraphLoopsReuse.Objects} {context : Context}
    {logical actual : GraphLoopsShape.Env Value} (related : RegistersRelated objects context logical actual)
    (binding : Binding) (bound : binding ∈ context) (object : Nat) (inside : object < objects.length)
    (shape : List Nat) (element : String) (pinned : logical binding.name = Epoch.view object shape element) :
    actual binding.name = Epoch.view (GraphLoopsReuse.slotOf objects object) shape element := by
  rw [related binding bound, pinned, slot_private_view objects object inside]

theorem RegistersRelated.caller_view {Value : Type} {objects : GraphLoopsReuse.Objects} {context : Context}
    {logical actual : GraphLoopsShape.Env Value} (related : RegistersRelated objects context logical actual)
    (binding : Binding) (bound : binding ∈ context) (object : Nat) (outside : ¬ object < objects.length)
    (shape : List Nat) (element : String) (pinned : logical binding.name = Epoch.view object shape element) :
    actual binding.name = logical binding.name := by
  rw [related binding bound, pinned, slot_caller_view objects object outside]

theorem RegistersRelated.scalar {Value : Type} {objects : GraphLoopsReuse.Objects} {context : Context}
    {logical actual : GraphLoopsShape.Env Value} (related : RegistersRelated objects context logical actual)
    (binding : Binding) (bound : binding ∈ context) (typed : binding.Holds (logical binding.name))
    (scalar : NoViewKind binding.kind) : actual binding.name = logical binding.name := by
  rw [related binding bound, scalar_slot_fixed objects binding (logical binding.name) typed scalar]

theorem RegistersRelated.returns {Value : Type} {objects : GraphLoopsReuse.Objects} {context : Context}
    {logical actual : GraphLoopsShape.Env Value} (related : RegistersRelated objects context logical actual)
    (typed : Typed context logical) (returned : List Binding)
    (bound : ∀ binding ∈ returned, binding ∈ context) (scalar : NoViewBindings returned) :
    readArguments false [] actual returned = readArguments false [] logical returned := by
  apply List.map_congr_left
  intro binding member
  exact related.scalar binding (bound binding member) (typed binding (bound binding member)) (scalar binding member)

/-- Reconstruct ALL initializer operands from the old scoped logical invariant
and the actual target register relation. The tile constant is separately bound
by the newly emitted constant instruction; no complete-environment equality is
needed, and an unbound private buffer is not rescued by a future ghost.
-/
theorem initializer_inputs_from_related {Value : Type} (objects : GraphLoopsReuse.Objects)
    (link : Epoch.InitializerLink) (facts : link.Facts) (program : TiledInitializer)
    (sameUniform : program.uniform = link.canonical)
    (context : Context) (views : List Epoch.PrivateView) (bound : link.Bound context views)
    (bufferBound : link.canonical.buffer ∈ context) (inside : link.config.object < objects.length)
    (logical actual : GraphLoopsShape.Env Value) (valid : Epoch.RegistersValid context views logical)
    (related : RegistersRelated objects context logical actual)
    (tileValue : actual program.tile.name = .index link.config.tile) :
    ∃ value, program.Inputs actual (GraphLoopsReuse.slotOf objects link.config.object)
      link.rows link.columns link.config.tile link.element value := by
  obtain ⟨value, seed, buffer, zero, rows, columns, one⟩ := link.inputs facts context views bound logical valid
  have actualSeed : actual link.canonical.seed.name = .scalar link.element value := by
    rw [related _ bound.1, seed]
    rfl
  have actualBuffer := related.private_view link.canonical.buffer bufferBound link.config.object inside
    [link.rows, link.columns] link.element buffer
  have actualZero : actual link.canonical.zero.name = .index 0 := by
    rw [related _ bound.2.1, zero]
    rfl
  have actualRows : actual link.canonical.rows.name = .index link.rows := by
    rw [related _ bound.2.2.1, rows]
    rfl
  have actualColumns : actual link.canonical.columns.name = .index link.columns := by
    rw [related _ bound.2.2.2.1, columns]
    rfl
  have actualOne : actual link.canonical.one.name = .index 1 := by
    rw [related _ bound.2.2.2.2.1, one]
    rfl
  refine ⟨value, ?_⟩
  constructor
  · simpa only [sameUniform] using actualSeed
  · simpa only [sameUniform] using actualBuffer
  · simpa only [sameUniform] using actualZero
  · simpa only [sameUniform] using actualRows
  · simpa only [sameUniform] using actualColumns
  · simpa only [sameUniform] using actualOne
  · exact tileValue

end Merlin.GraphLoopsMemorySCF.After
