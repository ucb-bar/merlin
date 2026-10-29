/- Checked logical fusion, using the existing admitted StateEffects.Action model.
The logical body is the actual source operation sequence with exact bindings.
No arithmetic reordering, numerical law or native implementation equality is assumed
by the logical theorem. A separate local native composition contract is explicit. -/
import Merlin.GraphRegions

namespace Merlin.GraphRegions

def publish (meaning : ∀ node, TypedAction node Value typing)
    (body : List Node) (outputs : List Nat) (initial : StateEffects.State Value) : StateEffects.State Value :=
  fun slot => if slot ∈ outputs then executeNodes meaning body initial slot else initial slot

/-- An atomic logical region computes the same ordered body and publishes every
live result. Hidden internal SSA bindings need not escape this region. -/
def logicalFusion (meaning : ∀ node, TypedAction node Value typing)
    (region suffix : List Node) (observations : List Nat) :
    StateEffects.Action ⟨liveIns region, liveOuts region suffix observations⟩ Value where
  step := publish meaning region (liveOuts region suffix observations)
  frame := by
    intro initial slot outside
    simp [publish, outside]
  locality := by
    intro first second inputs slot output
    simp only [publish, if_pos output]
    exact execute_locality meaning region first second inputs slot (List.mem_filter.mp output).1

theorem logicalFusion_law (meaning : ∀ node, TypedAction node Value typing)
    (region suffix : List Node) (observations : List Nat) :
    CompositionLaw meaning region suffix observations (logicalFusion meaning region suffix observations) := by
  intro initial _ slot output
  simp [logicalFusion, publish, output]

structure FusedBody where
  body : List Node
  inputs : List Nat
  outputs : List Nat
  deriving DecidableEq, Repr

def FusedBody.execute (meaning : ∀ node, TypedAction node Value typing)
    (fused : FusedBody) : StateEffects.State Value → StateEffects.State Value :=
  publish meaning fused.body fused.outputs

def checkFusion (region suffix : List Node) (observations : List Nat) (fused : FusedBody) : Bool :=
  decide (fused.body = region) && decide (fused.inputs = (liveIns region).eraseDups) &&
    decide (fused.outputs = liveOuts region suffix observations)

/-- Complete-context value preservation for a checked fused logical body. A
dropped producer observer or second result fails the actual output-list check. -/
theorem checkFusion_preserves_observers {region suffix : List Node} {observations : List Nat} {fused : FusedBody}
    (accepted : checkFusion region suffix observations fused = true)
    (meaning : ∀ node, TypedAction node Value typing)
    (earlier : List Node) (initial : StateEffects.State Value) (domain : WellTyped typing initial) :
    observations.map (executeNodes meaning suffix (fused.execute meaning (executeNodes meaning earlier initial))) =
      observations.map (executeNodes meaning (earlier ++ region ++ suffix) initial) := by
  simp only [checkFusion, Bool.and_eq_true, decide_eq_true_eq] at accepted
  simp only [FusedBody.execute, accepted.1.1, accepted.2]
  exact replace_region_preserves_observers meaning earlier region suffix observations
    (logicalFusion meaning region suffix observations) (logicalFusion_law meaning region suffix observations) initial domain

/-- Native implementation premises are LOCAL: exact complete effects, totality,
purity/read locality and equality of the checked live outputs. Serialized flags or
passing samples do not supply this contract. -/
theorem checked_native_contract {region suffix : List Node} {observations : List Nat} {fused : FusedBody}
    (accepted : checkFusion region suffix observations fused = true)
    (meaning : ∀ node, TypedAction node Value typing)
    (native : StateEffects.Action ⟨liveIns region, liveOuts region suffix observations⟩ Value)
    (implements : ∀ initial, WellTyped typing initial →
      AgreeOn (liveOuts region suffix observations) (native.step initial) (fused.execute meaning initial))
    (earlier : List Node) (initial : StateEffects.State Value) (domain : WellTyped typing initial) :
    observations.map (executeNodes meaning suffix (native.step (executeNodes meaning earlier initial))) =
      observations.map (executeNodes meaning (earlier ++ region ++ suffix) initial) := by
  simp only [checkFusion, Bool.and_eq_true, decide_eq_true_eq] at accepted
  apply replace_region_preserves_observers meaning earlier region suffix observations native _ initial domain
  intro entry typed slot output
  exact (implements entry typed slot output).trans (by simp [FusedBody.execute, accepted.1.1, accepted.2, publish, output])

/-- The actual retained source, proposed plan and realized logical body must all
describe the same selected boundary. This is not a native execution checker. -/
def checkRealization (before after : Graph) (start count : Nat)
    (selection : List Nat) (plan : Plan) (fused : FusedBody) : Bool :=
  checkBoundary before after start count selection plan &&
    checkFusion ((before.nodes.drop start).take count)
      (before.nodes.drop (start + count)) before.observations fused

/-- Integrated acceptance-to-complete-program observer preservation, extending
the existing StateEffects execution. The entire retained context and source SSA
are checked; logical fusion hides only results not read by the remaining program.
Concrete native implementation and source-to-model bridges are separate. -/
theorem checkRealization_sound {before after : Graph} {start count : Nat}
    {selection : List Nat} {plan : Plan} {fused : FusedBody}
    (accepted : checkRealization before after start count selection plan fused = true)
    (meaning : ∀ node, TypedAction node Value typing)
    (initial : StateEffects.State Value) (domain : WellTyped typing initial) :
    before.Valid ∧ before = after ∧
      before.observations.map (executeNodes meaning (before.nodes.drop (start + count))
        (fused.execute meaning (executeNodes meaning (before.nodes.take start) initial))) =
      before.observations.map (executeNodes meaning before.nodes initial) := by
  simp only [checkRealization, Bool.and_eq_true] at accepted
  obtain ⟨valid, retained, _⟩ := checkBoundary_sound accepted.1 meaning initial
  refine ⟨valid, retained, ?_⟩
  have result := checkFusion_preserves_observers accepted.2 meaning (before.nodes.take start) initial domain
  simpa only [split_nodes] using result

/-- The identical checked actual boundary also composes with a native action
WHEN its local initialized-output/effect contract is independently supplied. -/
theorem checkRealization_native {before after : Graph} {start count : Nat}
    {selection : List Nat} {plan : Plan} {fused : FusedBody}
    (accepted : checkRealization before after start count selection plan fused = true)
    (meaning : ∀ node, TypedAction node Value typing)
    (native : StateEffects.Action
      ⟨liveIns ((before.nodes.drop start).take count),
        liveOuts ((before.nodes.drop start).take count)
          (before.nodes.drop (start + count)) before.observations⟩ Value)
    (implements : ∀ entry, WellTyped typing entry →
      AgreeOn (liveOuts ((before.nodes.drop start).take count)
        (before.nodes.drop (start + count)) before.observations)
        (native.step entry) (fused.execute meaning entry))
    (initial : StateEffects.State Value) (domain : WellTyped typing initial) :
    before.observations.map (executeNodes meaning (before.nodes.drop (start + count))
      (native.step (executeNodes meaning (before.nodes.take start) initial))) =
      before.observations.map (executeNodes meaning before.nodes initial) := by
  simp only [checkRealization, Bool.and_eq_true] at accepted
  have result := checked_native_contract accepted.2 meaning native implements
    (before.nodes.take start) initial domain
  simpa only [split_nodes] using result

/-- Nonvacuity of the abstract typed-action domain. Type tags may be the actual
projected tensor types. This is not a concrete arithmetic/native interpretation. -/
def taggedInitial (types : Nat → String) : StateEffects.State (String × Unit) :=
  fun slot => (types slot, ())

def taggedMeaning (types : Nat → String) (node : Node) :
    TypedAction node (String × Unit) (fun slot value => value.1 = types slot) where
  action := {
    step := fun initial slot => if slot ∈ node.results then taggedInitial types slot else initial slot
    frame := by
      intro initial slot outside
      change slot ∉ node.results at outside
      simp [outside]
    locality := by
      intro first second _ slot output
      change slot ∈ node.results at output
      simp [output] }
  preservesTypes := by
    intro initial typed slot
    by_cases output : slot ∈ node.results
    · simp [output, taggedInitial]
    · simpa [output] using typed slot

theorem inhabited_domain (types : Nat → String) :
    ∃ (_meaning : ∀ node, TypedAction node (String × Unit) (fun slot value => value.1 = types slot))
      (initial : StateEffects.State (String × Unit)),
      WellTyped (fun slot value => value.1 = types slot) initial :=
  ⟨taggedMeaning types, taggedInitial types, fun _ => rfl⟩

end Merlin.GraphRegions
