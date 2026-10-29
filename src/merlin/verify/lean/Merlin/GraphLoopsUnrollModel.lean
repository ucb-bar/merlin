/- Actual complete SSA/control projection boundary for ordered unrolling.
Python's full decoder is trusted. Canonical body keys have ONE shared partial,
possibly effectful interpretation; no raw-MLIR operational bridge is assumed
proved. Every clone is checked, not just the first or an untrusted witness. -/
import Merlin.GraphLoopsUnrollBounds

set_option autoImplicit false

namespace Merlin.GraphLoopsUnroll.Model

open GraphLoopsSegment

structure Clone where
  index : Nat
  source : String
  target : String
  entry : List Nat
  exit : List Nat
  deriving DecidableEq, Repr

def Chain (source : String) (step : Nat) : Nat → List Nat → List Clone → List Nat → Prop
  | _, initial, [], final => initial = final
  | index, initial, clone :: rest, final => clone.index = index ∧ clone.source = source ∧
      clone.target = source ∧ clone.entry = initial ∧ Chain source step (index + step) clone.exit rest final

instance chainDecidable (source : String) (step index : Nat) (initial : List Nat) (clones : List Clone) (final : List Nat) :
    Decidable (Chain source step index initial clones final) := by
  cases clones with
  | nil => unfold Chain; infer_instance
  | cons clone rest =>
    unfold Chain
    have := chainDecidable source step (index + step) clone.exit rest final
    infer_instance

structure Projection where
  bounds : Bounds
  sourceBody : String
  clones : List Clone
  sourceContext : String
  targetContext : String
  sourceInitial : List Nat
  targetInitial : List Nat
  chainInitial : List Nat
  expectedExit : List Nat
  actualExit : List Nat
  sourceResults : List String
  targetResults : List String
  tailBody : String
  bulkBody : String
  targetStride : Nat
  targetBound : Nat
  deriving DecidableEq, Repr

def Structural (p : Projection) : Prop :=
  p.sourceContext = p.targetContext ∧ p.sourceInitial = p.targetInitial ∧
  p.sourceInitial.length = p.sourceResults.length ∧ p.targetResults = p.sourceResults ∧
  p.chainInitial.length = p.sourceResults.length ∧ p.expectedExit.length = p.sourceResults.length ∧
  p.expectedExit = p.actualExit ∧
  (if p.bounds.count ≤ p.bounds.factor then
    p.clones.length = p.bounds.count ∧ p.chainInitial = p.targetInitial ∧
    Chain p.sourceBody p.bounds.step p.bounds.lower p.chainInitial p.clones p.expectedExit ∧ p.tailBody = "" ∧ p.bulkBody = ""
  else p.clones.length = p.bounds.factor ∧
    Chain p.sourceBody p.bounds.step 0 p.chainInitial p.clones p.expectedExit ∧ p.bulkBody ≠ "" ∧
    (if p.bounds.remainder = 0 then p.tailBody = "" else p.tailBody = p.sourceBody))

instance (p : Projection) : Decidable (Structural p) := by
  unfold Structural
  infer_instance

def Numeric (p : Projection) : Prop := p.bounds.Valid ∧ p.targetStride = p.bounds.step * p.bounds.factor ∧
  p.targetBound = p.bounds.lower + p.bounds.step * (p.bounds.groups * p.bounds.factor)

instance (p : Projection) : Decidable (Numeric p) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def check (p : Projection) : Bool := decide (Numeric p ∧ Structural p)

theorem check_sound {p : Projection} (accepted : check p = true) : Numeric p ∧ Structural p :=
  of_decide_eq_true accepted

def original {σ : Type} (p : Projection) (interpret : String → Nat → σ → σ → Prop) :=
  Execution (interpret p.sourceBody) p.bounds.upper p.bounds.step p.bounds.lower

def transformed {σ : Type} (p : Projection) (interpret : String → Nat → σ → σ → Prop) :=
  Emitted (interpret p.sourceBody) p.bounds.lower p.bounds.upper p.bounds.step p.bounds.count
    p.bounds.factor p.bounds.groups p.bounds.remainder

/-- The interpreted emitted model is bound to a checked complete projection.
The decoder-to-raw-SCF semantics bridge remains an explicit adjacent obligation. -/
theorem checked_complete_trace {p : Projection} (accepted : check p = true) {σ : Type}
    (interpret : String → Nat → σ → σ → Prop) (initial final : σ) (trace : List Nat) :
    original p interpret initial trace final ↔ transformed p interpret initial trace final := by
  have valid := (check_sound accepted).1.1
  have factor := valid.2.2.2.1
  exact (emitted_iff_source (interpret p.sourceBody) _ _ _ _ _ _ _ valid.2.2.1
    (by omega : 0 < p.bounds.factor) valid.2.2.2.2.1 valid.2.2.2.2.2.1 initial final trace).symm

theorem checked_context {p : Projection} (accepted : check p = true) {σ Outcome : Type}
    (interpret : String → Nat → σ → σ → Prop)
    (context : String → (σ → List Nat → σ → Prop) → Outcome) :
    context p.sourceContext (original p interpret) = context p.targetContext (transformed p interpret) := by
  have keys := (check_sound accepted).2.1
  have relations : original p interpret = transformed p interpret := by
    funext initial trace final
    exact propext (checked_complete_trace accepted interpret initial final trace)
  rw [keys, relations]

theorem checked_carry_state {p : Projection} (accepted : check p = true) {Value : Type}
    (interpret : String → Nat → CarriedState Value p.sourceResults.length →
      CarriedState Value p.sourceResults.length → Prop)
    (initial final : CarriedState Value p.sourceResults.length) (trace : List Nat) :
    original p interpret initial trace final ↔ transformed p interpret initial trace final :=
  checked_complete_trace accepted interpret initial final trace

end Merlin.GraphLoopsUnroll.Model
