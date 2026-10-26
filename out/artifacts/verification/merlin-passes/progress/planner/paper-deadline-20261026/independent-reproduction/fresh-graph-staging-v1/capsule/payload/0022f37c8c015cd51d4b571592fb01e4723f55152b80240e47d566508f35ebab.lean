/- Multiple typed definitions and intrinsic descriptor operations. -/
import Merlin.GraphLoopsShapeSSA

set_option autoImplicit false

namespace Merlin.GraphLoopsShape

def extendContext : Context → List Binding → Context
  | context, [] => context
  | context, binding :: rest => extendContext (binding :: context) rest

def extendAliases : Aliases → List Binding → Aliases
  | aliases, [] => aliases
  | aliases, binding :: rest => extendAliases ((binding.name, binding.name) :: aliases) rest

def assign {Value : Type} : Env Value → List Binding → List (Runtime Value) → Env Value
  | env, binding :: rest, value :: remaining => assign (put env binding.name value) rest remaining
  | env, _, _ => env

def Fresh : Context → List Binding → Prop
  | _, [] => True
  | context, binding :: rest => binding.name ∉ names context ∧ Fresh (binding :: context) rest

def freshDecidable (context : Context) : (bindings : List Binding) → Decidable (Fresh context bindings)
  | [] => isTrue True.intro
  | binding :: rest =>
    letI := freshDecidable (binding :: context) rest
    inferInstanceAs (Decidable (_ ∧ _))

instance (context : Context) (bindings : List Binding) : Decidable (Fresh context bindings) := freshDecidable context bindings

inductive ValuesHold {Value : Type} : List Binding → List (Runtime Value) → Prop where
  | nil : ValuesHold [] []
  | cons {binding : Binding} {value : Runtime Value} {rest : List Binding} {remaining : List (Runtime Value)}
      (holds : binding.Holds value) (typed : ValuesHold rest remaining) : ValuesHold (binding :: rest) (value :: remaining)

theorem Scope.define_many {context : Context} {aliases : Aliases} (scope : Scope context aliases)
    (bindings : List Binding) (fresh : Fresh context bindings) :
    Scope (extendContext context bindings) (extendAliases aliases bindings) := by
  induction bindings generalizing context aliases with
  | nil => exact scope
  | cons binding rest induction => exact induction (scope.extend fresh.1 (Or.inl rfl)) fresh.2

theorem Agreement.define_many {Value : Type} {context : Context} {aliases : Aliases} {source target : Env Value}
    (same : Agreement context aliases source target) (scope : Scope context aliases)
    {bindings : List Binding} {values : List (Runtime Value)} (typed : ValuesHold bindings values)
    (fresh : Fresh context bindings) :
    Agreement (extendContext context bindings) (extendAliases aliases bindings)
      (assign source bindings values) (assign target bindings values) := by
  induction typed generalizing context aliases source target with
  | nil => exact same
  | @cons binding value rest remaining holds typed induction =>
    exact induction (same.define scope binding fresh.1 value holds) (scope.extend fresh.1 (Or.inl rfl)) fresh.2

def readArguments {Value : Type} (mode : Bool) (aliases : Aliases) (env : Env Value) (bindings : List Binding) : List (Runtime Value) :=
  bindings.map (fun binding => env (if mode then resolve aliases binding.name else binding.name))

theorem Agreement.read_arguments {Value : Type} {context : Context} {aliases : Aliases} {source target : Env Value}
    (same : Agreement context aliases source target) (operands : List Binding)
    (bound : ∀ binding ∈ operands, binding ∈ context) :
    readArguments false aliases source operands = readArguments true aliases target operands := by
  exact same.arguments operands bound

def castResult {Value : Type} (source target : ViewType) (value : Runtime Value) : Option (Runtime Value) :=
  match value with
  | .view actual _ _ => if actual = source ∧ source = target then some value else none
  | _ => none

def dimensionResult {Value : Type} (type : ViewType) (view axis : Runtime Value) : Option Nat :=
  match view, axis with
  | .view actual descriptor _, .index index => if actual = type then descriptor.shape[index]? else none
  | _, _ => none

theorem cast_result {Value : Type} {value : Runtime Value} {type : ViewType}
    (typed : value.kind = .view type) : castResult type type value = some value := by
  obtain ⟨descriptor, valid, rfl⟩ := Runtime.view_of_kind typed
  simp only [castResult, and_self, ↓reduceIte]

theorem dimension_result {Value : Type} {value : Runtime Value} {type : ViewType} {bits axis result : Nat}
    (typed : value.kind = .view type) (valid : ValidDimension bits type axis result) :
    dimensionResult type value (.index axis) = some result := by
  obtain ⟨descriptor, conforms, rfl⟩ := Runtime.view_of_kind typed
  simp only [dimensionResult, ↓reduceIte, conforms.1]
  exact valid.2.1

theorem Binding.holds_index {Value : Type} (binding : Binding) (index : Nat)
    (kind : binding.kind = .index) (known : binding.known = some index) : binding.Holds (Runtime.index (Value := Value) index) := by
  refine ⟨kind.symm, ?_⟩
  intro other equal
  have same : other = index := (Option.some.inj (equal.symm.trans known))
  subst other
  rfl

end Merlin.GraphLoopsShape
