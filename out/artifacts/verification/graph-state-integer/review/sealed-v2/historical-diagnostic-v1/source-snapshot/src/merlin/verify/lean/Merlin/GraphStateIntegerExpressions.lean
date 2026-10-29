/- Kernel-checked local integer rewrite sequences over the existing Scalar.Expr.
Every neutral deletion requires the actual retained expression's runtime word
type. Python may propose a sequence; only the following proved rules accept it. -/
import Merlin.GraphStateIntegerOperations

namespace Merlin.GraphState.Integer
set_option autoImplicit false
open Scalar

inductive ExprTyped (environment : List Ty) : Expr → Ty → Prop where
  | var {index : Nat} {type : Ty} (found : environment[index]? = some type) :
      ExprTyped environment (.var index) type
  | zero {op : Prim} (valid : op.valid = true) (arity : op.inputs = []) :
      ExprTyped environment (.zero op) op.output
  | one {op : Prim} {argument : Expr} {type : Ty}
      (valid : op.valid = true) (arity : op.inputs = [type])
      (operand : ExprTyped environment argument type) :
      ExprTyped environment (.one op argument) op.output
  | two {op : Prim} {left right : Expr} {first second : Ty}
      (valid : op.valid = true) (arity : op.inputs = [first,second])
      (lhs : ExprTyped environment left first) (rhs : ExprTyped environment right second) :
      ExprTyped environment (.two op left right) op.output

theorem ExprTyped.canonical {environment : List Ty} {expression : Expr} {type : Ty}
    (typed : ExprTyped environment expression type) (values : List Int)
    (entry : Values Canonical environment values) :
    Canonical type (evalExpr 0 primitive values expression) := by
  cases typed with
  | var found => exact entry.lookup found 0
  | zero valid arity => exact primitive_canonical _ _
  | one valid arity operand => exact primitive_canonical _ _
  | two valid arity lhs rhs => exact primitive_canonical _ _

inductive ExprsTyped (environment : List Ty) : List Expr → List Ty → Prop where
  | nil : ExprsTyped environment [] []
  | cons {head : Expr} {tail : List Expr} {type : Ty} {types : List Ty}
      (first : ExprTyped environment head type) (rest : ExprsTyped environment tail types) :
      ExprsTyped environment (head::tail) (type::types)

theorem ExprsTyped.canonical {environment types : List Ty} {expressions : List Expr}
    (typed : ExprsTyped environment expressions types) (values : List Int)
    (entry : Values Canonical environment values) :
    Values Canonical types (expressions.map (evalExpr 0 primitive values)) := by
  induction typed with
  | nil => exact .nil
  | cons first rest ih => exact .cons (first.canonical values entry) ih

def Same (environment : List Ty) (first second : Expr) : Prop :=
  ∀ values, Values Canonical environment values →
    evalExpr 0 primitive values first = evalExpr 0 primitive values second

theorem Same.refl (environment : List Ty) (expression : Expr) : Same environment expression expression :=
  fun _ _ => rfl

theorem Same.symm {environment : List Ty} {first second : Expr} (same : Same environment first second) :
    Same environment second first := fun values entry => (same values entry).symm

theorem Same.trans {environment : List Ty} {first middle last : Expr}
    (left : Same environment first middle) (right : Same environment middle last) :
    Same environment first last := fun values entry => (left values entry).trans (right values entry)

theorem Same.one {environment : List Ty} {first second : Expr} (op : Prim)
    (same : Same environment first second) : Same environment (.one op first) (.one op second) := by
  intro values entry
  simp only [evalExpr,same values entry]

theorem Same.two {environment : List Ty} {first second left right : Expr} (op : Prim)
    (lhs : Same environment first second) (rhs : Same environment left right) :
    Same environment (.two op first left) (.two op second right) := by
  intro values entry
  simp only [evalExpr,lhs values entry,rhs values entry]

theorem Same.constant (environment : List Ty) (width : Nat) (first second : Int)
    (bits : word width first = word width second) :
    Same environment (.zero (.constant width first)) (.zero (.constant width second)) := by
  intro values entry
  exact bits

theorem Same.fold_add (environment : List Ty) (width : Nat) (first second result : Int)
    (bits : word width (first+second) = word width result) :
    Same environment (.two (.add width) (.zero (.constant width first)) (.zero (.constant width second)))
      (.zero (.constant width result)) := by
  intro values entry
  exact (word_add width first second).trans bits

theorem Same.fold_sub (environment : List Ty) (width : Nat) (first second result : Int)
    (bits : word width (first-second) = word width result) :
    Same environment (.two (.sub width) (.zero (.constant width first)) (.zero (.constant width second)))
      (.zero (.constant width result)) := by
  intro values entry
  exact (word_sub width first second).trans bits

theorem Same.fold_mul (environment : List Ty) (width : Nat) (first second result : Int)
    (bits : word width (first*second) = word width result) :
    Same environment (.two (.mul width) (.zero (.constant width first)) (.zero (.constant width second)))
      (.zero (.constant width result)) := by
  intro values entry
  exact (word_mul width first second).trans bits

theorem Same.add_zero {environment : List Ty} {value : Expr} (width : Nat)
    (typed : ExprTyped environment value (.bits width)) :
    Same environment (.two (.add width) value (.zero (.constant width 0))) value := by
  intro values entry
  exact Integer.add_zero width _ (typed.canonical values entry)

theorem Same.zero_add {environment : List Ty} {value : Expr} (width : Nat)
    (typed : ExprTyped environment value (.bits width)) :
    Same environment (.two (.add width) (.zero (.constant width 0)) value) value := by
  intro values entry
  exact Integer.zero_add width _ (typed.canonical values entry)

theorem Same.sub_zero {environment : List Ty} {value : Expr} (width : Nat)
    (typed : ExprTyped environment value (.bits width)) :
    Same environment (.two (.sub width) value (.zero (.constant width 0))) value := by
  intro values entry
  exact Integer.sub_zero width _ (typed.canonical values entry)

theorem Same.mul_one {environment : List Ty} {value : Expr} (width : Nat)
    (typed : ExprTyped environment value (.bits width)) :
    Same environment (.two (.mul width) value (.zero (.constant width 1))) value := by
  intro values entry
  exact Integer.mul_one width _ (typed.canonical values entry)

theorem Same.one_mul {environment : List Ty} {value : Expr} (width : Nat)
    (typed : ExprTyped environment value (.bits width)) :
    Same environment (.two (.mul width) (.zero (.constant width 1)) value) value := by
  intro values entry
  exact Integer.one_mul width _ (typed.canonical values entry)

def Sames (environment : List Ty) (first second : List Expr) : Prop :=
  ∀ values, Values Canonical environment values →
    first.map (evalExpr 0 primitive values) = second.map (evalExpr 0 primitive values)

theorem Sames.nil (environment : List Ty) : Sames environment [] [] := fun _ _ => rfl

theorem Sames.cons {environment : List Ty} {first second : Expr} {left right : List Expr}
    (head : Same environment first second) (tail : Sames environment left right) :
    Sames environment (first::left) (second::right) := by
  intro values entry
  simp only [List.map_cons,head values entry,tail values entry]

def ExactModel {World Error : Type} {registry : List Signature} (model : Model Int World Error registry) : Prop :=
  model.default = 0 ∧ model.primitive = primitive

theorem Same.model {World Error : Type} {registry : List Signature}
    {environment : List Ty} {first second : Expr} (same : Same environment first second)
    (model : Model Int World Error registry) (exact : ExactModel model)
    (values : List Int) (entry : Values Canonical environment values) :
    evalExpr model.default model.primitive values first = evalExpr model.default model.primitive values second := by
  rw [exact.1,exact.2]
  exact same values entry

#print axioms Same.fold_add
#print axioms Same.fold_sub
#print axioms Same.fold_mul
#print axioms Same.add_zero
#print axioms ExprTyped.canonical
end Merlin.GraphState.Integer
