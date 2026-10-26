/- UNCOMPILED successor. Caller-visible relation deliberately ignores private
allocation clocks and private cells. It does not equate complete MemoryWorlds.
Only constructed public-address actions may use this relation. -/
import Merlin.GraphLoopsMemorySCFAfterEntriesWitness

set_option autoImplicit false
namespace Merlin.GraphLoopsMemorySCF.After
open GraphLoopsShape

abbrev CallerWorld (Value User : Type) := MemoryWorld Value (Epoch.AllocationPublic User)

def PublicRelated {Value User : Type} (bound : Nat) (first second : CallerWorld Value User) : Prop :=
  first.observed.observed = second.observed.observed ∧
    Epoch.visibleHeap bound first.heap = Epoch.visibleHeap bound second.heap

theorem PublicRelated.refl {Value User : Type} (bound : Nat) (world : CallerWorld Value User) :
    PublicRelated bound world world := ⟨rfl, rfl⟩

theorem PublicRelated.symm {Value User : Type} {bound : Nat} {first second : CallerWorld Value User}
    (same : PublicRelated bound first second) : PublicRelated bound second first := ⟨same.1.symm, same.2.symm⟩

theorem PublicRelated.trans {Value User : Type} {bound : Nat} {first second third : CallerWorld Value User}
    (left : PublicRelated bound first second) (right : PublicRelated bound second third) :
    PublicRelated bound first third := ⟨left.1.trans right.1, left.2.trans right.2⟩

theorem PublicRelated.read {Value User : Type} {bound : Nat} {first second : CallerWorld Value User}
    (same : PublicRelated bound first second) (object index : Nat) (outside : ¬ object < bound) :
    first.heap object index = second.heap object index := by
  have cell := congrFun (congrFun same.2 object) index
  simpa only [Epoch.visibleHeap, outside, ↓reduceIte] using cell

theorem PublicRelated.store {Value User : Type} {bound : Nat} {first second : CallerWorld Value User}
    (same : PublicRelated bound first second) (object index : Nat) (value : Value) :
    PublicRelated bound
      ⟨first.observed, GraphLoopsReuse.replaceCell first.heap object index value⟩
      ⟨second.observed, GraphLoopsReuse.replaceCell second.heap object index value⟩ := by
  refine ⟨same.1, ?_⟩
  funext other cell
  by_cases private : other < bound
  · simp only [Epoch.visibleHeap, private, ↓reduceIte]
  · by_cases objectEqual : other = object
    · subst other
      by_cases cellEqual : cell = index
      · simp [Epoch.visibleHeap, private, GraphLoopsReuse.replaceCell, GraphLoopsReuse.replaceObject, cellEqual]
      · simpa [Epoch.visibleHeap, private, GraphLoopsReuse.replaceCell, GraphLoopsReuse.replaceObject, cellEqual]
          using same.read object cell private
    · simpa [Epoch.visibleHeap, private, GraphLoopsReuse.replaceCell, GraphLoopsReuse.replaceObject, objectEqual]
        using same.read other cell private

theorem observation_related {Value User : Type} (bound : Nat) (first second : CallerWorld Value User)
    (outputs other : List (Runtime Value)) :
    Epoch.observeWorld bound outputs first = Epoch.observeWorld bound other second ↔
      outputs = other ∧ PublicRelated bound first second := by
  constructor
  · intro same
    exact ⟨congrArg Epoch.Observation.returned same,
      congrArg Epoch.Observation.user same, congrArg Epoch.Observation.heap same⟩
  · rintro ⟨rfl, user, heap⟩
    cases first with | mk firstPublic firstHeap =>
      cases second with | mk secondPublic secondHeap =>
        simp only [Epoch.observeWorld, user, heap]

def ExternalValues {Value : Type} (bound : Nat) (values : List (Runtime Value)) : Prop :=
  ∀ value ∈ values, ExternalInput bound value

def ExternalEnv {Value : Type} (bound : Nat) (env : GraphLoopsShape.Env Value) : Prop :=
  ∀ name, ExternalInput bound (env name)

theorem external_empty {Value : Type} (bound : Nat) : ExternalEnv bound (emptyEnv (Value := Value)) := by
  intro name; trivial

theorem external_put {Value : Type} (bound : Nat) (env : GraphLoopsShape.Env Value) (safe : ExternalEnv bound env)
    (name : Nat) (value : Runtime Value) (outside : ExternalInput bound value) :
    ExternalEnv bound (put env name value) := by
  intro other
  by_cases equal : other = name <;> simp only [put, equal, ↓reduceIte]
  · exact outside
  · exact safe other

theorem external_assign {Value : Type} (bound : Nat) (env : GraphLoopsShape.Env Value) (safe : ExternalEnv bound env)
    (bindings : List Binding) (values : List (Runtime Value)) (outside : ExternalValues bound values) :
    ExternalEnv bound (assign env bindings values) := by
  induction bindings generalizing env values with
  | nil => exact safe
  | cons binding rest induction =>
    cases values with
    | nil => exact safe
    | cons value remaining =>
      exact induction (put env binding.name value)
        (external_put bound env safe binding.name value (outside value List.mem_cons_self)) remaining
        (fun item member => outside item (List.mem_cons_of_mem _ member))

theorem external_arguments {Value : Type} (bound : Nat) (env : GraphLoopsShape.Env Value) (safe : ExternalEnv bound env)
    (aliases : Aliases) (bindings : List Binding) : ExternalValues bound (readArguments false aliases env bindings) := by
  intro value member
  obtain ⟨binding, _, rfl⟩ := List.mem_map.mp member
  exact safe binding.name

theorem nonview_external {Value : Type} (bound : Nat) (binding : Binding) (value : Runtime Value)
    (typed : binding.Holds value) (scalar : NoViewKind binding.kind) : ExternalInput bound value := by
  have kind := typed.1
  cases value with
  | index => trivial
  | scalar => trivial
  | view type descriptor valid => rw [← kind] at scalar; exact False.elim scalar

theorem nonview_values_external {Value : Type} (bound : Nat) (bindings : List Binding) (values : List (Runtime Value))
    (typed : ValuesHold bindings values) (scalar : NoViewBindings bindings) : ExternalValues bound values := by
  induction typed with
  | nil => intro value member; cases member
  | @cons binding value rest remaining holds typed induction =>
    intro item member
    rcases List.mem_cons.mp member with rfl | later
    · exact nonview_external bound binding value holds (scalar binding List.mem_cons_self)
    · exact induction (fun binding member => scalar binding (List.mem_cons_of_mem _ member)) item later

theorem cast_external {Value : Type} (bound : Nat) (source target : ViewType) (input output : Runtime Value)
    (safe : ExternalInput bound input) (cast : castResult source target input = some output) : ExternalInput bound output := by
  cases input with
  | index => simp only [castResult] at cast
  | scalar => simp only [castResult] at cast
  | view actual descriptor valid =>
    simp only [castResult] at cast
    split at cast
    · cases Option.some.inj cast
      exact safe
    · contradiction

end Merlin.GraphLoopsMemorySCF.After
