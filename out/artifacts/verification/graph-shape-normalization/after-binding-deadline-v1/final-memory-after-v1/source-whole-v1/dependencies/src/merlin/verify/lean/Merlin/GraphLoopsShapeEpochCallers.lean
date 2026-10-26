/- Caller descriptor invariants across scoped scalar definitions. Caller objects
are outside the private logical allocation partition. They may alias one another;
no caller/caller disjointness or native pointer identity is silently assumed. -/
import Merlin.GraphLoopsShapeEpochState

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

structure CallerView where
  binding : Binding
  element : String
  deriving DecidableEq, Repr

def CallersValid {Value : Type} (bound : Nat) (callers : List CallerView) (env : Env Value) : Prop :=
  ∀ caller ∈ callers, ∃ object, ¬ object < bound ∧ env caller.binding.name = view object [1] caller.element

def CallersBound (context : Context) (callers : List CallerView) : Prop :=
  ∀ caller ∈ callers, caller.binding ∈ context

instance callersBoundDecidable (context : Context) (callers : List CallerView) : Decidable (CallersBound context callers) :=
  inferInstanceAs (Decidable (∀ caller ∈ callers, caller.binding ∈ context))

theorem fresh_separates (context results : Context) (fresh : Fresh context results) (name : Nat)
    (bound : name ∈ names context) : name ∉ names results := by
  induction results generalizing context with
  | nil => simp [names]
  | cons result rest induction =>
    have different : name ≠ result.name := fun equal => fresh.1 (equal ▸ bound)
    have remaining := induction (result :: context) fresh.2 (by simp only [names, List.map_cons, List.mem_cons]; exact Or.inr bound)
    simpa only [names, List.map_cons, List.mem_cons, not_or] using And.intro different remaining

theorem callers_assign {Value : Type} (bound : Nat) (context : Context) (callers : List CallerView)
    (env : Env Value) (valid : CallersValid bound callers env) (callersInContext : CallersBound context callers)
    (results : List Binding) (fresh : Fresh context results) (values : List (Runtime Value)) :
    CallersValid bound callers (assign env results values) := by
  intro caller member
  obtain ⟨object, outside, read⟩ := valid caller member
  refine ⟨object, outside, ?_⟩
  rw [assign_at_other env results values caller.binding.name
    (fresh_separates context results fresh caller.binding.name (List.mem_map_of_mem (callersInContext caller member)))]
  exact read

theorem indices_of_typed {Value : Type} (context : Context) (env : Env Value) (typed : Typed context env)
    (indices : List Binding) (bound : AllBound context indices)
    (kinds : ∀ binding ∈ indices, binding.kind = .index) :
    ∃ coordinates : List Nat, readArguments false [] env indices = coordinates.map Runtime.index := by
  induction indices with
  | nil => exact ⟨[], rfl⟩
  | cons index rest induction =>
    have actualKind : (env index.name).kind = .index := (typed index (bound index (by simp))).1.trans (kinds index (by simp))
    have runtimeIndex : ∃ value, env index.name = .index value := by
      cases read : env index.name with
      | index value => exact ⟨value, rfl⟩
      | scalar => rw [read] at actualKind; cases actualKind
      | view => rw [read] at actualKind; cases actualKind
    obtain ⟨value, read⟩ := runtimeIndex
    obtain ⟨coordinates, tail⟩ := induction (fun binding member => bound binding (List.mem_cons_of_mem _ member))
      (fun binding member => kinds binding (List.mem_cons_of_mem _ member))
    refine ⟨value :: coordinates, ?_⟩
    change env index.name :: readArguments false [] env rest = .index value :: coordinates.map Runtime.index
    rw [read, tail]

end Merlin.GraphLoopsShape.Epoch
