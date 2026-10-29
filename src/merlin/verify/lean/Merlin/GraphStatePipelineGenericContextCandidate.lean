/- Additive source candidate. Arbitrary common prefixes retain complete
fault/world behavior without unfolding their partial operations in an instance. -/
import Merlin.GraphStatePipelineContext
import Merlin.GraphStatePipelineScope

namespace Merlin.GraphState.Pipeline.Context
set_option autoImplicit false
open Scalar

theorem plug_congr {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) (segment : Program) (count : Nat)
    {left right : Program}
    (same : EquivalentAt model (count + frameGrowth registry segment) left right) :
    EquivalentAt model count (plug segment left) (plug segment right) := by
  induction segment generalizing count with
  | ret values =>
    simpa only [plug,frameGrowth,Nat.add_zero] using same
  | pure op arguments later ih =>
    apply pure_congr model count op arguments
    apply ih (count + 1)
    simpa only [frameGrowth,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using same
  | «opaque» identity arguments outputs later ih =>
    apply opaque_congr model count identity arguments outputs
    apply ih ((signature registry identity).outputs.length + count)
    simpa only [frameGrowth,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using same
  | loop lower upper stride initial carries body later ihBody ihNext =>
    apply loop_congr model count lower upper stride initial carries
    · exact refl model _ body
    · apply ihNext (carries.length + count)
      simpa only [frameGrowth,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using same

/-- Identity symbols recover the whole actual lexical frame. The finite scope
checker can select a tiny closure without unpacking hundreds of private cells. -/
theorem vars_identity {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) (environment : List Value) (count : Nat)
    (size : environment.length = count) :
    (vars count).map (evalExpr model.default model.primitive environment) = environment := by
  rw [← size]
  simpa only [List.append_nil] using eval_vars model.default model.primitive environment []

theorem scope_closure {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) {left right : Program}
    (count : Nat) (slots : List Nat)
    (checked : Scope.check registry (vars count) (slots.map Expr.var) left right = true)
    (environment : List Value) (world : World) (size : environment.length = count) :
    evalProgram model left environment world =
      evalProgram model right (reads model.default environment slots) world := by
  have actual := Scope.check_sound checked model environment world
  rw [vars_identity model environment count size] at actual
  simpa only [List.map_map,Function.comp_def,evalExpr,reads] using actual

end Merlin.GraphState.Pipeline.Context
