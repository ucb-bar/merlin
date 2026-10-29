/- Actual logical host dispatch fusion under the existing GraphRegions model.
The host projects both emitted programs independently. A target call contributes
its actual ordered body and returned values, not a planner's intended body.
Translation, total pure primitive meanings and ordinary function-call fidelity
remain explicit boundaries. No device/native implementation law is assumed. -/
import Merlin.GraphRegionsFusion

namespace Merlin.HostGraphFusion
open Merlin.GraphRegions

def execute (meaning : ∀ node, TypedAction node Value typing) (graph : Graph)
    (start count : Nat) (body : FusedBody) (initial : StateEffects.State Value) : List Value :=
  graph.observations.map (executeNodes meaning (graph.nodes.drop (start + count))
    (body.execute meaning (executeNodes meaning (graph.nodes.take start) initial)))

def check (before after : Graph) (start count : Nat) (body : FusedBody) : Bool :=
  let region := (after.nodes.drop start).take count
  decide before.Valid && decide (before = after) &&
    decide (2 ≤ count ∧ count ≤ 128 ∧ start + count ≤ after.nodes.length) &&
    decide (body.inputs.Perm (liveIns region).eraseDups) &&
    checkFusion region (after.nodes.drop (start + count)) after.observations
      { body with inputs := (liveIns region).eraseDups }

/-- The complete public result list, including producer fanout, is preserved
for every well-typed state and every admitted primitive interpretation. The
actual body binds argument positions independently of the caller's ordering
of its live-in ports. The body and published results are checked facts. -/
theorem check_sound {before after : Graph} {start count : Nat} {body : FusedBody}
    (accepted : check before after start count body = true)
    (meaning : ∀ node, TypedAction node Value typing)
    (initial : StateEffects.State Value) (domain : WellTyped typing initial) :
    before.Valid ∧ execute meaning after start count body initial =
      before.observations.map (executeNodes meaning before.nodes initial) := by
  simp only [check, Bool.and_eq_true, decide_eq_true_eq] at accepted
  obtain ⟨⟨⟨⟨valid, same⟩, _⟩, _⟩, fusion⟩ := accepted
  subst after
  refine ⟨valid, ?_⟩
  have result := checkFusion_preserves_observers fusion meaning
    (before.nodes.take start) initial domain
  simpa only [execute, FusedBody.execute, split_nodes] using result

end Merlin.HostGraphFusion
