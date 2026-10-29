/- ONE operation-level initializer interpretation with scoped SSA values.
Pure operators may be partial; their only inputs are explicit scalar/index
operands. Stores emit ordered writes; heap/public framing is proved separately.
This does not add a route to the frozen MerlinVerify evaluator. -/
import Merlin.GraphLoopsReuseCountermodels
import Merlin.GraphLoopsUnrollGrouping

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF

inductive Datum (Value : Type) where
  | index (value : Nat)
  | scalar (value : Value)

abbrev Env (Value : Type) := Nat → Option (Datum Value)
abbrev Write (Value : Type) := Nat × Value
abbrev Log (Value : Type) := List (Write Value)

def put {Value : Type} (env : Env Value) (name : Nat) (value : Datum Value) : Env Value :=
  fun other => if other = name then some value else env other

def assign {Value : Type} : Env Value → List Nat → List (Datum Value) → Env Value
  | env, name :: names, value :: values => assign (put env name value) names values
  | env, _, _ => env

def read {Value : Type} (env : Env Value) : List Nat → Option (List (Datum Value))
  | [] => some []
  | name :: names => do
      let value ← env name
      let remaining ← read env names
      pure (value :: remaining)

structure PureOp where
  key : String
  operands : List Nat
  results : List Nat
  deriving DecidableEq, Repr

abbrev PureMeaning (Value : Type) := String → List (Datum Value) → List (Datum Value) → Prop

inductive EvalPure {Value : Type} (meaning : PureMeaning Value) : List PureOp → Env Value → Env Value → Prop where
  | done (env : Env Value) : EvalPure meaning [] env env
  | more {op : PureOp} {rest : List PureOp} {before after : Env Value} {arguments values : List (Datum Value)}
      (inputs : read before op.operands = some arguments) (compute : meaning op.key arguments values)
      (arity : values.length = op.results.length)
      (remaining : EvalPure meaning rest (assign before op.results values) after) : EvalPure meaning (op :: rest) before after

def definitions (ops : List PureOp) : List Nat := ops.flatMap PureOp.results

theorem assign_frame {Value : Type} (env : Env Value) (names : List Nat) (values : List (Datum Value)) (name : Nat)
    (outside : name ∉ names) : assign env names values name = env name := by
  induction names generalizing env values with
  | nil => rfl
  | cons first rest induction =>
    cases values with
    | nil => rfl
    | cons value values =>
      have different : name ≠ first := fun same => outside (same ▸ List.mem_cons_self)
      have restOutside : name ∉ rest := fun member => outside (List.mem_cons_of_mem first member)
      change assign (put env first value) rest values name = env name
      rw [induction _ _ restOutside]
      exact ite_eq_right different

theorem EvalPure.frame {Value : Type} {meaning : PureMeaning Value} {ops : List PureOp} {before after : Env Value}
    (run : EvalPure meaning ops before after) {name : Nat} (outside : name ∉ definitions ops) : after name = before name := by
  induction run with
  | done => rfl
  | @more op rest before after arguments values inputs compute arity remaining induction =>
    have outsideFirst : name ∉ op.results := fun member => outside (List.mem_append_left _ member)
    have outsideRest : name ∉ definitions rest := fun member => outside (List.mem_append_right _ member)
    exact (induction outsideRest).trans (assign_frame before op.results values name outsideFirst)

structure Leaf where
  ops : List PureOp
  value : Nat
  indices : List Nat
  deriving DecidableEq, Repr

def LeafResult {Value : Type} (meaning : PureMeaning Value) (leaf : Leaf) (env : Env Value)
    (coordinates : List Nat) (value : Value) : Prop :=
  ∃ after, EvalPure meaning leaf.ops env after ∧ after leaf.value = some (.scalar value) ∧
    read after leaf.indices = some (coordinates.map Datum.index)

structure Initializer where
  shape : List Nat
  ivs : List Nat
  outerPrefix : List PureOp
  leaf : Leaf
  deriving DecidableEq, Repr

/-- Closed locals restore the exact incoming environment after each body.
The log is the only state changed inside initialization. -/
structure ScopedLog (Value : Type) where
  env : Env Value
  writes : Log Value

/-- The store reads every actual coordinate from its SSA operands. Its address
is row-major; source/target buffer binding is supplied by the checked epoch map. -/
def Emit {Value : Type} (meaning : PureMeaning Value) (leaf : Leaf) (shape : List Nat)
    (before : ScopedLog Value) (after : ScopedLog Value) : Prop :=
  ∃ coordinates value, LeafResult meaning leaf before.env coordinates value ∧
    GraphLoopsReuse.Geometry.Coordinate shape coordinates ∧
    after = ⟨before.env, before.writes ++ [(GraphLoopsReuse.Geometry.flatten shape coordinates, value)]⟩

end Merlin.GraphLoopsMemorySCF
