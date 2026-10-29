/- Typed lexical SSA and forward aliases for deleted descriptor operations.
Distinct source/candidate register stacks are related only at bound names.
Retained operations receive explicit values, never the private SSA environment. -/
import Merlin.GraphLoopsShape

set_option autoImplicit false

namespace Merlin.GraphLoopsShape

inductive Kind where
  | index
  | scalar (type : String)
  | view (type : ViewType)
  deriving DecidableEq, Repr

inductive Runtime (Value : Type) where
  | index (value : Nat)
  | scalar (type : String) (value : Value)
  | view (type : ViewType) (descriptor : Descriptor) (valid : Conforms type descriptor)

def Runtime.kind {Value : Type} : Runtime Value → Kind
  | .index _ => .index
  | .scalar type _ => .scalar type
  | .view type _ _ => .view type

structure Binding where
  name : Nat
  kind : Kind
  known : Option Nat
  deriving DecidableEq, Repr

def Binding.Holds {Value : Type} (binding : Binding) (value : Runtime Value) : Prop :=
  value.kind = binding.kind ∧ ∀ index, binding.known = some index → value = .index index

abbrev Env (Value : Type) := Nat → Runtime Value
abbrev Context := List Binding
abbrev Aliases := List (Nat × Nat)

def names (context : Context) : List Nat := context.map Binding.name

def resolve : Aliases → Nat → Nat
  | [], name => name
  | (source, target) :: rest, name => if name = source then target else resolve rest name

def put {Value : Type} (env : Env Value) (name : Nat) (value : Runtime Value) : Env Value :=
  fun other => if other = name then value else env other

def Typed {Value : Type} (context : Context) (env : Env Value) : Prop :=
  ∀ binding ∈ context, binding.Holds (env binding.name)

/-- Images must remain actual previously bound source names. Thus a fresh SSA
result cannot overwrite a retained target register used by another alias. -/
def Scope (context : Context) (aliases : Aliases) : Prop :=
  (names context).Nodup ∧ ∀ binding ∈ context, resolve aliases binding.name ∈ names context

instance (context : Context) (aliases : Aliases) : Decidable (Scope context aliases) :=
  inferInstanceAs (Decidable (_ ∧ _))

def Agreement {Value : Type} (context : Context) (aliases : Aliases) (source target : Env Value) : Prop :=
  Typed context source ∧ ∀ binding ∈ context, source binding.name = target (resolve aliases binding.name)

theorem Runtime.view_of_kind {Value : Type} {value : Runtime Value} {type : ViewType}
    (kind : value.kind = .view type) : ∃ descriptor valid, value = .view type descriptor valid := by
  cases value with
  | index => cases kind
  | scalar => cases kind
  | view actual descriptor valid =>
    have same : actual = type := Kind.view.inj kind
    subst actual
    exact ⟨descriptor, valid, rfl⟩

theorem known_index {Value : Type} {binding : Binding} {value : Runtime Value} {index : Nat}
    (holds : binding.Holds value) (known : binding.known = some index) : value = .index index := holds.2 index known

theorem Scope.extend {context : Context} {aliases : Aliases} {binding : Binding} {target : Nat}
    (scope : Scope context aliases) (fresh : binding.name ∉ names context)
    (image : target = binding.name ∨ target ∈ names context) :
    Scope (binding :: context) ((binding.name, target) :: aliases) := by
  refine ⟨List.nodup_cons.mpr ⟨fresh, scope.1⟩, ?_⟩
  intro current member
  rcases List.mem_cons.mp member with rfl | previous
  · simp only [resolve, ↓reduceIte, names, List.map_cons, List.mem_cons]
    exact image
  · have different : current.name ≠ binding.name := by
      intro same
      exact fresh (same ▸ List.mem_map_of_mem previous)
    simp only [resolve, ite_eq_right different, names, List.map_cons, List.mem_cons]
    exact Or.inr (scope.2 current previous)

/-- Retained definitions extend both stacks, but only after freshness rules
out overwriting any previous source name OR target alias image. -/
theorem Agreement.define {Value : Type} {context : Context} {aliases : Aliases} {source target : Env Value}
    (same : Agreement context aliases source target) (scope : Scope context aliases) (binding : Binding)
    (fresh : binding.name ∉ names context) (value : Runtime Value) (typed : binding.Holds value) :
    Agreement (binding :: context) ((binding.name, binding.name) :: aliases)
      (put source binding.name value) (put target binding.name value) := by
  have different : ∀ current ∈ context, current.name ≠ binding.name := by
    intro current member equal
    exact fresh (equal ▸ List.mem_map_of_mem member)
  refine ⟨?_, ?_⟩
  · intro current member
    rcases List.mem_cons.mp member with rfl | previous
    · simpa only [put, ↓reduceIte] using typed
    · simpa only [put, ite_eq_right (different current previous)] using same.1 current previous
  · intro current member
    rcases List.mem_cons.mp member with rfl | previous
    · simp only [resolve, put, ↓reduceIte]
    · have imageDifferent : resolve aliases current.name ≠ binding.name := by
        intro equal
        exact fresh (equal ▸ scope.2 current previous)
      simpa only [resolve, ite_eq_right (different current previous), put, ite_eq_right imageDifferent] using same.2 current previous

/-- Deleting an identity cast extends ONLY the source stack. The candidate
uses the original value through the checked forward alias. -/
theorem Agreement.alias {Value : Type} {context : Context} {aliases : Aliases} {source target : Env Value}
    (same : Agreement context aliases source target) {origin : Binding} (member : origin ∈ context)
    (binding : Binding) (fresh : binding.name ∉ names context) (typed : binding.Holds (source origin.name)) :
    Agreement (binding :: context) ((binding.name, resolve aliases origin.name) :: aliases)
      (put source binding.name (source origin.name)) target := by
  refine ⟨?_, ?_⟩
  · intro current initialized
    rcases List.mem_cons.mp initialized with rfl | previous
    · simpa only [put, ↓reduceIte] using typed
    · have different : current.name ≠ binding.name := fun equal => fresh (equal ▸ List.mem_map_of_mem previous)
      simpa only [put, ite_eq_right different] using same.1 current previous
  · intro current initialized
    rcases List.mem_cons.mp initialized with rfl | previous
    · simpa only [put, resolve, ↓reduceIte] using same.2 origin member
    · have different : current.name ≠ binding.name := fun equal => fresh (equal ▸ List.mem_map_of_mem previous)
      simpa only [put, resolve, ite_eq_right different] using same.2 current previous

theorem Agreement.arguments {Value : Type} {context : Context} {aliases : Aliases} {source target : Env Value}
    (same : Agreement context aliases source target) (operands : List Binding)
    (bound : ∀ binding ∈ operands, binding ∈ context) :
    operands.map (fun binding => source binding.name) = operands.map (fun binding => target (resolve aliases binding.name)) := by
  apply List.map_congr_left
  intro binding member
  exact same.2 binding (bound binding member)

end Merlin.GraphLoopsShape
