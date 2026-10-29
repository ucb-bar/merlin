/- Explicit link from SCF complete carry/effect traces to the existing Merlin
StateEffects execution interface. This does not identify SCF with counted Kernel
or old MerlinVerify numerical semantics. Instruction/body admission remains explicit. -/
import Merlin.GraphLoopsSegmentModel

set_option autoImplicit false

namespace Merlin.GraphLoopsSegment

structure CarriedState (Value : Type) (arity : Nat) where
  carried : Fin arity → Value
  memory : StateEffects.State Value

/-- With an explicitly supplied deterministic TOTAL state action for each IV,
the relational execution is the existing StateEffects sequential execution.
No totality or locality fact about actual instructions is inferred here. -/
theorem Execution.state_actions {Value : Type}
    (run : Nat → StateEffects.State Value → StateEffects.State Value) {upper step lower : Nat}
    {initial final : StateEffects.State Value} {trace : List Nat}
    (execution : Execution (fun index before after => after = run index before) upper step lower initial trace final) :
    StateEffects.execute (trace.map run) initial = final := by
  induction execution with
  | done => rfl
  | @more lower initial middle final trace inside transition remaining induction =>
    change StateEffects.execute (trace.map run) (run lower initial) = final
    rw [← transition]
    exact induction

/-- Every actual carried slot and every existing modeled memory/state component
is preserved at the complete checked boundary, including both yield levels. -/
theorem Model.checked_carry_state {projection : Model.Projection} (accepted : Model.check projection = true)
    {Value : Type}
    (interpret : String → Nat → CarriedState Value projection.carryTypes.length →
      CarriedState Value projection.carryTypes.length → Prop)
    (initial final : CarriedState Value projection.carryTypes.length) (trace : List Nat) :
    Model.original projection interpret initial trace final ↔ Model.transformed projection interpret initial trace final := by
  exact Model.checked_complete_trace accepted interpret initial final trace

end Merlin.GraphLoopsSegment
