/- Complete lexical graph blocks with nested region semantics. Operators see
explicit inputs and scoped region functions, never private SSA stacks. A region
function restores its captured stack and returns only declared values/world. -/
import Merlin.GraphLoopsShapeBindings

set_option autoImplicit false

namespace Merlin.GraphLoopsShape

inductive Code where
  | returned (values : List Binding)
  | cast (origin result : Binding) (source target : ViewType) (rest : Code)
  | dimension (view axis result : Binding) (type : ViewType) (axisIndex value : Nat) (rest : Code)
  | constant (result : Binding) (value : Nat) (rest : Code)
  | action (key : String) (operands results : List Binding) (rest : Code)
  | region (key : String) (operands parameters results : List Binding) (body rest : Code)
  deriving DecidableEq, Repr

abbrev RegionMeaning (Value World : Type) := List (Runtime Value) → World → List (Runtime Value) → World → Prop

structure Interpretation (Value World : Type) where
  action : String → List (Runtime Value) → World → List (Runtime Value) → World → Prop
  region : String → List (Runtime Value) → RegionMeaning Value World → World → List (Runtime Value) → World → Prop

def AllBound (context : Context) (operands : List Binding) : Prop := ∀ binding ∈ operands, binding ∈ context
def Plain (bindings : List Binding) : Prop := ∀ binding ∈ bindings, binding.known = none

instance (context : Context) (operands : List Binding) : Decidable (AllBound context operands) :=
  inferInstanceAs (Decidable (∀ binding ∈ operands, binding ∈ context))
instance (bindings : List Binding) : Decidable (Plain bindings) :=
  inferInstanceAs (Decidable (∀ binding ∈ bindings, binding.known = none))

def CastFacts (context : Context) (origin result : Binding) (source target : ViewType) : Prop :=
  origin ∈ context ∧ origin.kind = .view source ∧ source = target ∧ result.kind = .view target ∧
  result.known = none ∧ result.name ∉ names context

def DimensionFacts (bits : Nat) (context : Context) (view axis result : Binding) (type : ViewType) (axisIndex value : Nat) : Prop :=
  view ∈ context ∧ axis ∈ context ∧ view.kind = .view type ∧ axis.known = some axisIndex ∧
  result.kind = .index ∧ result.known = some value ∧ ValidDimension bits type axisIndex value ∧ result.name ∉ names context

def ConstantFacts (bits : Nat) (context : Context) (result : Binding) (value : Nat) : Prop :=
  result.kind = .index ∧ result.known = some value ∧ value < 2 ^ (bits - 1) ∧ result.name ∉ names context

instance (context : Context) (origin result : Binding) (source target : ViewType) :
    Decidable (CastFacts context origin result source target) := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))
instance (bits : Nat) (context : Context) (view axis result : Binding) (type : ViewType) (axisIndex value : Nat) :
    Decidable (DimensionFacts bits context view axis result type axisIndex value) := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))
instance (bits : Nat) (context : Context) (result : Binding) (value : Nat) : Decidable (ConstantFacts bits context result value) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

/-- Every name, known constant and scope obligation is recomputed from the full
blueprint. Source metadata alone is not an operational descriptor witness. -/
def Valid (bits : Nat) (expected : List Kind) (context : Context) (aliases : Aliases) (code : Code) : Prop :=
  Scope context aliases ∧ match code with
  | .returned values => AllBound context values ∧ values.map Binding.kind = expected
  | .cast origin result source target rest => CastFacts context origin result source target ∧
      Valid bits expected (result :: context) ((result.name, resolve aliases origin.name) :: aliases) rest
  | .dimension view axis result type axisIndex value rest => DimensionFacts bits context view axis result type axisIndex value ∧
      Valid bits expected (result :: context) ((result.name, result.name) :: aliases) rest
  | .constant result value rest => ConstantFacts bits context result value ∧
      Valid bits expected (result :: context) ((result.name, result.name) :: aliases) rest
  | .action _ operands results rest => AllBound context operands ∧ Fresh context results ∧ Plain results ∧
      Valid bits expected (extendContext context results) (extendAliases aliases results) rest
  | .region _ operands parameters results body rest => AllBound context operands ∧ Fresh context parameters ∧ Plain parameters ∧
      Valid bits (results.map Binding.kind) (extendContext context parameters) (extendAliases aliases parameters) body ∧
      Fresh context results ∧ Plain results ∧
      Valid bits expected (extendContext context results) (extendAliases aliases results) rest

def validDecidable (bits : Nat) (expected : List Kind) (context : Context) (aliases : Aliases) :
    (code : Code) → Decidable (Valid bits expected context aliases code)
  | .returned _ => inferInstanceAs (Decidable (_ ∧ _ ∧ _))
  | .cast origin result _ _ rest =>
    letI := validDecidable bits expected (result :: context) ((result.name, resolve aliases origin.name) :: aliases) rest
    inferInstanceAs (Decidable (_ ∧ _ ∧ _))
  | .dimension _ _ result _ _ _ rest =>
    letI := validDecidable bits expected (result :: context) ((result.name, result.name) :: aliases) rest
    inferInstanceAs (Decidable (_ ∧ _ ∧ _))
  | .constant result _ rest =>
    letI := validDecidable bits expected (result :: context) ((result.name, result.name) :: aliases) rest
    inferInstanceAs (Decidable (_ ∧ _ ∧ _))
  | .action _ _ results rest =>
    letI := validDecidable bits expected (extendContext context results) (extendAliases aliases results) rest
    inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _))
  | .region _ _ parameters results body rest =>
    letI := validDecidable bits (results.map Binding.kind) (extendContext context parameters) (extendAliases aliases parameters) body
    letI := validDecidable bits expected (extendContext context results) (extendAliases aliases results) rest
    inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))

instance (bits : Nat) (expected : List Kind) (context : Context) (aliases : Aliases) (code : Code) :
    Decidable (Valid bits expected context aliases code) := validDecidable bits expected context aliases code

/-- Mode false interprets original casts/dim metadata reads; mode true interprets
actual cast deletion/static index materialization. Logical blueprint recursion
bounds syntax traversal, not runtime iterations or effects inside region meaning. -/
def Eval {Value World : Type} (meaning : Interpretation Value World) (mode : Bool) (aliases : Aliases) (code : Code)
    (env : Env Value) (before : World) (returned : List (Runtime Value)) (after : World) : Prop :=
  match code with
  | .returned values => returned = readArguments mode aliases env values ∧ after = before
  | .cast origin result source target rest =>
      if mode then Eval meaning mode ((result.name, resolve aliases origin.name) :: aliases) rest env before returned after else
        ∃ value, castResult source target (env origin.name) = some value ∧
          Eval meaning mode ((result.name, resolve aliases origin.name) :: aliases) rest (put env result.name value) before returned after
  | .dimension view axis result type _ value rest =>
      if mode then Eval meaning mode ((result.name, result.name) :: aliases) rest (put env result.name (.index value)) before returned after else
        ∃ actual, dimensionResult type (env view.name) (env axis.name) = some actual ∧
          Eval meaning mode ((result.name, result.name) :: aliases) rest (put env result.name (.index actual)) before returned after
  | .constant result value rest =>
      Eval meaning mode ((result.name, result.name) :: aliases) rest (put env result.name (.index value)) before returned after
  | .action key operands results rest =>
      ∃ values middle, meaning.action key (readArguments mode aliases env operands) before values middle ∧ ValuesHold results values ∧
        Eval meaning mode (extendAliases aliases results) rest (assign env results values) middle returned after
  | .region key operands parameters results body rest =>
      ∃ values middle, meaning.region key (readArguments mode aliases env operands)
        (fun inputs initial outputs final => ValuesHold parameters inputs ∧
          Eval meaning mode (extendAliases aliases parameters) body (assign env parameters inputs) initial outputs final)
        before values middle ∧ ValuesHold results values ∧
        Eval meaning mode (extendAliases aliases results) rest (assign env results values) middle returned after

end Merlin.GraphLoopsShape
