/- Operational scalar SSA language. Pure primitives are deterministic TOTAL
value functions, with no persistent-state access. Other retained operations
may fail and expose a terminal world/error. Only returned values and complete
persistent outcome are observable; deleted private SSA slots need not agree. -/
import Init

namespace Merlin.GraphState.Scalar
set_option autoImplicit false

inductive Ty where
  | bits (width : Nat)
  | index
  | opaque (identity : Nat)
  deriving DecidableEq, BEq, ReflBEq, LawfulBEq, Repr

inductive Prim where
  | constant (width : Nat) (value : Int)
  | add (width : Nat)
  | sub (width : Nat)
  | mul (width : Nat)
  | sext (source target : Nat)
  | zext (source target : Nat)
  | trunc (source target : Nat)
  deriving DecidableEq, BEq, ReflBEq, LawfulBEq, Repr

def Prim.inputs : Prim → List Ty
  | .constant _ _ => []
  | .add width | .sub width | .mul width => [.bits width,.bits width]
  | .sext source _ | .zext source _ | .trunc source _ => [.bits source]

def Prim.output : Prim → Ty
  | .constant width _ | .add width | .sub width | .mul width => .bits width
  | .sext _ target | .zext _ target | .trunc _ target => .bits target

def widthValid (width : Nat) : Bool := decide (1 ≤ width ∧ width ≤ 256)

def Prim.valid : Prim → Bool
  | .constant width _ | .add width | .sub width | .mul width => widthValid width
  | .sext source target | .zext source target =>
    widthValid source && widthValid target && decide (source < target)
  | .trunc source target => widthValid source && widthValid target && decide (target < source)

/-- Logical expressions reference only inputs and retained result binders,
never arbitrary independently numbered source/destination SSA cells. -/
inductive Expr where
  | var (index : Nat)
  | zero (primitive : Prim)
  | one (primitive : Prim) (argument : Expr)
  | two (primitive : Prim) (left right : Expr)
  deriving DecidableEq, BEq, ReflBEq, LawfulBEq, Repr

def evalExpr {Value : Type} (default : Value) (primitive : Prim → List Value → Value)
    (environment : List Value) : Expr → Value
  | .var index => environment[index]?.getD default
  | .zero op => primitive op []
  | .one op argument => primitive op [evalExpr default primitive environment argument]
  | .two op left right => primitive op [evalExpr default primitive environment left,evalExpr default primitive environment right]

def shift (amount : Nat) : Expr → Expr
  | .var index => .var (amount+index)
  | .zero op => .zero op
  | .one op argument => .one op (shift amount argument)
  | .two op left right => .two op (shift amount left) (shift amount right)

def reads {α : Type} (default : α) (environment : List α) (indices : List Nat) : List α :=
  indices.map (fun index => environment[index]?.getD default)

inductive Program where
  | ret (values : List Nat)
  | pure (primitive : Prim) (arguments : List Nat) (next : Program)
  | opaque (identity : Nat) (arguments : List Nat) (results : List Ty) (next : Program)
  | loop (lower upper stride : Nat) (initial : List Nat) (carries : List Ty) (body next : Program)
  deriving DecidableEq, BEq, ReflBEq, LawfulBEq, Repr

inductive Code where
  | ret (values : List Expr)
  | opaque (identity : Nat) (arguments : List Expr) (results : List Ty) (next : Code)
  | loop (lower upper stride : Expr) (initial : List Expr) (carries : List Ty) (body next : Code)
  deriving DecidableEq, BEq, ReflBEq, LawfulBEq, Repr

structure Signature where
  inputs : List Ty
  outputs : List Ty
  deriving DecidableEq, BEq, ReflBEq, LawfulBEq, Repr

def signature (registry : List Signature) (identity : Nat) : Signature :=
  registry[identity]?.getD {inputs := [],outputs := []}

inductive Fault (Error : Type) where
  | external (error : Error)
  | arity
  deriving DecidableEq, Repr

abbrev Outcome (Value World Error : Type) := Except (Fault Error × World) (List Value × World)

/-- A retained call consumes its exact ordered operands and persistent state.
Successful result arity is an explicit LOCAL operation contract. No whole-source
success assumption is made. Failures include the complete terminal world. -/
structure Model (Value World Error : Type) (registry : List Signature) where
  default : Value
  primitive : Prim → List Value → Value
  index : Int → Value
  iterations : Value → Value → Value → List Int
  effect : Nat → List Value → World → Except (Error × World) (List Value × World)
  arity : ∀ identity arguments world values final,
    effect identity arguments world = .ok (values,final) →
    values.length = (signature registry identity).outputs.length

def opaqueStep {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) (identity : Nat) (arguments : List Value) (world : World) :
    Outcome Value World Error :=
  match model.effect identity arguments world with
  | .error (error,final) => .error (.external error,final)
  | .ok result => .ok result

def loopEval {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error) :
    List Int → List Value → World → Outcome Value World Error
  | iterations,values,world =>
    if values.length = arity then
      match iterations with
      | [] => .ok (values,world)
      | index::later => (body index values world).bind (fun result => loopEval arity body later result.1 result.2)
    else .error (.arity,world)

def evalProgram {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) : Program → List Value → World → Outcome Value World Error
  | .ret values,environment,world => .ok (reads model.default environment values,world)
  | .pure op arguments next,environment,world =>
    evalProgram model next (model.primitive op (reads model.default environment arguments)::environment) world
  | .opaque identity arguments _ next,environment,world =>
    (opaqueStep model identity (reads model.default environment arguments) world).bind
      (fun result => evalProgram model next (result.1++environment) result.2)
  | .loop lower upper stride initial carries body next,environment,world =>
    let iterations := model.iterations (environment[lower]?.getD model.default)
      (environment[upper]?.getD model.default) (environment[stride]?.getD model.default)
    (loopEval carries.length
      (fun index values current => evalProgram model body (model.index index::values++environment) current)
      iterations (reads model.default environment initial) world).bind
        (fun result => evalProgram model next (result.1++environment) result.2)

def evalCode {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) : Code → List Value → World → Outcome Value World Error
  | .ret values,environment,world => .ok (values.map (evalExpr model.default model.primitive environment),world)
  | .opaque identity arguments _ next,environment,world =>
    (opaqueStep model identity (arguments.map (evalExpr model.default model.primitive environment)) world).bind
      (fun result => evalCode model next (result.1++environment) result.2)
  | .loop lower upper stride initial carries body next,environment,world =>
    let iterations := model.iterations (evalExpr model.default model.primitive environment lower)
      (evalExpr model.default model.primitive environment upper) (evalExpr model.default model.primitive environment stride)
    (loopEval carries.length
      (fun index values current => evalCode model body (model.index index::values++environment) current)
      iterations (initial.map (evalExpr model.default model.primitive environment)) world).bind
        (fun result => evalCode model next (result.1++environment) result.2)

/-- Complete live/persistent observations. Source and target private SSA stacks
are independent and need not be pointwise equal. -/
def Preserves (registry : List Signature) (inputCount : Nat) (before after : Program) : Prop :=
  ∀ (Value World Error : Type) (model : Model Value World Error registry) inputs world,
    inputs.length = inputCount →
    evalProgram model before inputs world = evalProgram model after inputs world

end Merlin.GraphState.Scalar
