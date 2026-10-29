/- A complete observation skeleton, derived from both SSA programs. Pure nodes
are inlined under exact operand bindings; retained opaque/effect order and loop
control/carry/result slots remain explicit. This is not the optimizer's plan. -/
import Merlin.GraphStateScalarContract

namespace Merlin.GraphState.Scalar
set_option autoImplicit false

def vars : Nat → List Expr
  | 0 => []
  | count+1 => .var 0 :: (vars count).map (shift 1)

def lift (count : Nat) (symbols : List Expr) : List Expr :=
  vars count ++ symbols.map (shift count)

def lookupAll (symbols : List Expr) : List Nat → Option (List Expr)
  | [] => some []
  | index::later => do
    let expression ← symbols[index]?
    let rest ← lookupAll symbols later
    return expression::rest

def makeExpr (primitive : Prim) : List Expr → Option Expr
  | [] => if primitive.inputs.length = 0 then some (.zero primitive) else none
  | [argument] => if primitive.inputs.length = 1 then some (.one primitive argument) else none
  | [left,right] => if primitive.inputs.length = 2 then some (.two primitive left right) else none
  | _ => none

def normalize (registry : List Signature) : List Expr → Program → Option Code
  | symbols,.ret values => return .ret (← lookupAll symbols values)
  | symbols,.pure primitive arguments next => do
    let operands ← lookupAll symbols arguments
    let expression ← makeExpr primitive operands
    normalize registry (expression::symbols) next
  | symbols,.opaque identity arguments results next => do
    if identity < registry.length ∧ results = (signature registry identity).outputs then
      let operands ← lookupAll symbols arguments
      let rest ← normalize registry (lift results.length symbols) next
      return .opaque identity operands results rest
    else none
  | symbols,.loop lower upper stride initial carries body next => do
    let lb ← symbols[lower]?
    let ub ← symbols[upper]?
    let step ← symbols[stride]?
    let values ← lookupAll symbols initial
    let inside ← normalize registry (lift (carries.length+1) symbols) body
    let later ← normalize registry (lift carries.length symbols) next
    return .loop lb ub step values carries inside later

theorem lookup_append {Value : Type} (earlier environment : List Value) (index : Nat) :
    (earlier++environment)[earlier.length+index]? = environment[index]? := by
  induction earlier with
  | nil => simp
  | cons head tail ih => simpa only [List.cons_append,List.length_cons,Nat.succ_add,List.getElem?_cons_succ] using ih

@[simp] theorem eval_shift {Value : Type} (default : Value) (primitive : Prim → List Value → Value)
    (earlier environment : List Value) (expression : Expr) :
    evalExpr default primitive (earlier++environment) (shift earlier.length expression) =
      evalExpr default primitive environment expression := by
  induction expression with
  | var index => simp only [shift,evalExpr,lookup_append]
  | zero op => rfl
  | one op argument ih => simp only [shift,evalExpr,ih]
  | two op left right ihLeft ihRight => simp only [shift,evalExpr,ihLeft,ihRight]

theorem eval_vars {Value : Type} (default : Value) (primitive : Prim → List Value → Value)
    (earlier environment : List Value) :
    (vars earlier.length).map (evalExpr default primitive (earlier++environment)) = earlier := by
  induction earlier with
  | nil => rfl
  | cons head tail ih =>
    simp only [List.length_cons,vars,List.map_cons,List.map_map,Function.comp_def]
    change head :: (vars tail.length).map (fun expression =>
      evalExpr default primitive ([head]++(tail++environment)) (shift [head].length expression)) = head::tail
    simpa only [eval_shift] using congrArg (List.cons head) ih

theorem eval_lift {Value : Type} (default : Value) (primitive : Prim → List Value → Value)
    (earlier environment : List Value) (symbols : List Expr) :
    (lift earlier.length symbols).map (evalExpr default primitive (earlier++environment)) =
      earlier ++ symbols.map (evalExpr default primitive environment) := by
  simp only [lift,List.map_append,eval_vars,List.map_map,Function.comp_def,eval_shift]

theorem lookupAll_sound {Value : Type} (default : Value) (primitive : Prim → List Value → Value)
    (environment : List Value) {symbols : List Expr} {indices : List Nat} {expressions : List Expr}
    (found : lookupAll symbols indices = some expressions) :
    reads default (symbols.map (evalExpr default primitive environment)) indices =
      expressions.map (evalExpr default primitive environment) := by
  induction indices generalizing expressions with
  | nil => cases found; rfl
  | cons index rest ih =>
    simp only [lookupAll,Option.bind_eq_bind] at found
    cases item : symbols[index]? with
    | none => simp [item] at found
    | some expression =>
      cases tail : lookupAll symbols rest with
      | none => simp [item,tail] at found
      | some later =>
        simp only [item,tail,Option.bind_some,Option.pure_def,Option.some.injEq] at found
        subst expressions
        have head : (symbols.map (evalExpr default primitive environment))[index]?.getD default =
            evalExpr default primitive environment expression := by
          simp [List.getElem?_map,item]
        simpa only [reads,List.map_cons,head] using congrArg (List.cons (evalExpr default primitive environment expression)) (ih tail)

theorem makeExpr_sound {Value : Type} (default : Value) (primitive : Prim → List Value → Value)
    (environment : List Value) {op : Prim} {arguments : List Expr} {expression : Expr}
    (made : makeExpr op arguments = some expression) :
    evalExpr default primitive environment expression =
      primitive op (arguments.map (evalExpr default primitive environment)) := by
  cases arguments with
  | nil => simp only [makeExpr] at made; split at made <;> cases made <;> rfl
  | cons first rest =>
    cases rest with
    | nil => simp only [makeExpr] at made; split at made <;> cases made <;> rfl
    | cons second more =>
      cases more with
      | nil => simp only [makeExpr] at made; split at made <;> cases made <;> rfl
      | cons _ _ => cases made

end Merlin.GraphState.Scalar
