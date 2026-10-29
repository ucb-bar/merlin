/- Checked lexical scope/capture correspondence using the existing scalar
normalization root. Ghost private cells may differ; only explicit logical
bindings and the complete returned/error/world observation are compared. -/
import Merlin.GraphStateScalarSound

namespace Merlin.GraphState.Pipeline.Scope
set_option autoImplicit false
open Scalar

def check (registry : List Signature) (leftSymbols rightSymbols : List Expr)
    (left right : Program) : Bool :=
  match normalize registry leftSymbols left, normalize registry rightSymbols right with
  | some first,some second => first == second
  | _,_ => false

/-- Acceptance checks every ordered use against its own lexical environment;
it never identifies independently numbered SSA cells. The symbol maps are part
of the certificate. Their connection to actual outer frames is checked at use. -/
theorem check_sound {registry : List Signature} {leftSymbols rightSymbols : List Expr}
    {left right : Program} (accepted : check registry leftSymbols rightSymbols left right = true)
    {Value World Error : Type} (model : Model Value World Error registry) (logical : List Value) (world : World) :
    evalProgram model left (leftSymbols.map (evalExpr model.default model.primitive logical)) world =
      evalProgram model right (rightSymbols.map (evalExpr model.default model.primitive logical)) world := by
  cases first : normalize registry leftSymbols left with
  | none => simp [check,first] at accepted
  | some code =>
    cases second : normalize registry rightSymbols right with
    | none => simp [check,first,second] at accepted
    | some other =>
      have same : code = other := by simpa only [check,first,second,beq_iff_eq] using accepted
      subst other
      exact (normalize_sound model left leftSymbols code first logical world).trans
        (normalize_sound model right rightSymbols code second logical world).symm

/-- A successful body result can be rebound to a differently sized private
frame using the checked arity law; internal private slots are not observations. -/
theorem rebind {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) (earlier logical : List Value) (symbols : List Expr) :
    (lift earlier.length symbols).map (evalExpr model.default model.primitive (earlier++logical)) =
      earlier ++ symbols.map (evalExpr model.default model.primitive logical) :=
  eval_lift model.default model.primitive earlier logical symbols

end Merlin.GraphState.Pipeline.Scope
