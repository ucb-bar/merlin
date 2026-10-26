/- Checked structural/control boundary for actual supported SCF segmentation.
Canonical body/context syntax and SSA identities are projected from BOTH actual
modules. Their Python decoder remains trusted; no raw-MLIR parser theorem is
implied. Arbitrary interpretations of the unchanged body/context may be partial
and effectful. No numerical law or success premise is manufactured here. -/
import Merlin.GraphLoopsSegmentBounds

set_option autoImplicit false

namespace Merlin.GraphLoopsSegment.Model

structure Projection where
  bounds : Bounds
  targetLower : Nat
  targetUpper : Nat
  targetStride : Nat
  targetInnerStep : Nat
  sourceBody : String
  targetBody : String
  sourceContext : String
  targetContext : String
  carryTypes : List String
  sourceInitial : List Nat
  outerInitial : List Nat
  sourceResults : List String
  outerResults : List String
  innerResults : List String
  outerArgumentTypes : List String
  outerIV : Nat
  outerCarryArguments : List Nat
  strideValue : Nat
  upperValue : Nat
  originalStepValue : Nat
  addOperands : List Nat
  addResult : Nat
  minOperands : List Nat
  minResult : Nat
  innerOperands : List Nat
  innerResultValues : List Nat
  outerYield : List Nat
  sourceYieldArity : Nat
  deriving DecidableEq, Repr

def Numeric (projection : Projection) : Prop :=
  projection.bounds.Valid ∧ projection.targetLower = projection.bounds.lower ∧
  projection.targetUpper = projection.bounds.upper ∧ projection.targetInnerStep = projection.bounds.step ∧
  projection.targetStride = projection.bounds.step * projection.bounds.tile

def Structural (projection : Projection) : Prop :=
  projection.sourceBody = projection.targetBody ∧ projection.sourceContext = projection.targetContext ∧
  projection.sourceInitial.length = projection.carryTypes.length ∧ projection.outerInitial = projection.sourceInitial ∧
  projection.sourceResults = projection.carryTypes ∧ projection.outerResults = projection.carryTypes ∧
  projection.innerResults = projection.carryTypes ∧
  projection.outerArgumentTypes = "index" :: projection.carryTypes ∧
  projection.outerCarryArguments.length = projection.carryTypes.length ∧
  projection.addOperands = [projection.outerIV, projection.strideValue] ∧
  projection.minOperands = [projection.addResult, projection.upperValue] ∧
  projection.innerOperands = projection.outerIV :: projection.minResult ::
    projection.originalStepValue :: projection.outerCarryArguments ∧
  projection.innerResultValues.length = projection.carryTypes.length ∧
  projection.outerYield = projection.innerResultValues ∧ projection.sourceYieldArity = projection.carryTypes.length

instance (projection : Projection) : Decidable (Numeric projection) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _))

instance (projection : Projection) : Decidable (Structural projection) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))

def check (projection : Projection) : Bool := decide (Numeric projection ∧ Structural projection)

theorem check_sound {projection : Projection} (accepted : check projection = true) :
    Numeric projection ∧ Structural projection := of_decide_eq_true accepted

/-- Arbitrary actual outer stride/inner step, specialized only after the checked
control equalities are established. SSA/control decoding is separately recorded. -/
inductive Nested {σ : Type} (body : Nat → σ → σ → Prop) (upper innerStep outerStep : Nat) :
    Nat → σ → List Nat → σ → Prop where
  | done {lower : Nat} (state : σ) (stopped : upper ≤ lower) : Nested body upper innerStep outerStep lower state [] state
  | more {lower : Nat} {initial middle final : σ} {first rest : List Nat}
      (inside : lower < upper)
      (inner : Execution body (min (lower + outerStep) upper) innerStep lower initial first middle)
      (remaining : Nested body upper innerStep outerStep (lower + outerStep) middle rest final) :
      Nested body upper innerStep outerStep lower initial (first ++ rest) final

theorem nested_iff {σ : Type} (body : Nat → σ → σ → Prop) (upper step tile lower : Nat)
    (initial final : σ) (trace : List Nat) :
    Nested body upper step (step * tile) lower initial trace final ↔
      Segmented body upper step tile lower initial trace final := by
  constructor
  · intro run
    induction run with
    | done state stopped => exact .done state stopped
    | more inside inner _ induction => exact .more inside inner induction
  · intro run
    induction run with
    | done state stopped => exact .done state stopped
    | more inside inner _ induction => exact .more inside inner induction

def original {σ : Type} (projection : Projection) (interpret : String → Nat → σ → σ → Prop)
    (initial : σ) (trace : List Nat) (final : σ) : Prop :=
  Execution (interpret projection.sourceBody) projection.bounds.upper projection.bounds.step
    projection.bounds.lower initial trace final

def transformed {σ : Type} (projection : Projection) (interpret : String → Nat → σ → σ → Prop)
    (initial : σ) (trace : List Nat) (final : σ) : Prop :=
  Nested (interpret projection.targetBody) projection.targetUpper projection.targetInnerStep
    projection.targetStride projection.targetLower initial trace final

/-- The complete actual boundary, relative to the recorded canonical syntax
interpretation. Every carried component and modeled state observation is in σ. -/
theorem checked_complete_trace {projection : Projection} (accepted : check projection = true) {σ : Type}
    (interpret : String → Nat → σ → σ → Prop) (initial final : σ) (trace : List Nat) :
    original projection interpret initial trace final ↔ transformed projection interpret initial trace final := by
  obtain ⟨⟨valid, lower, upper, step, stride⟩, bindingFacts⟩ := check_sound accepted
  have body := bindingFacts.1
  simp only [original, transformed, lower, upper, step, stride, ← body]
  rw [nested_iff]
  exact complete_trace_iff _ _ _ _ _ valid.2.2.1 (by have := valid.2.2.2.1; omega) _ _ _

/-- Equal loop behavior may be plugged into the identical surrounding context.
This quantifies over its full semantics, including observers and state effects;
the IR-to-canonical-context correspondence is a trusted projection obligation. -/
theorem checked_context {projection : Projection} (accepted : check projection = true) {σ Outcome : Type}
    (interpret : String → Nat → σ → σ → Prop)
    (context : String → (σ → List Nat → σ → Prop) → Outcome) :
    context projection.sourceContext (original projection interpret) =
      context projection.targetContext (transformed projection interpret) := by
  have sameContext := (check_sound accepted).2.2.1
  have sameBehavior : original projection interpret = transformed projection interpret := by
    funext initial trace final
    exact propext (checked_complete_trace accepted interpret initial final trace)
  rw [sameContext, sameBehavior]

end Merlin.GraphLoopsSegment.Model
