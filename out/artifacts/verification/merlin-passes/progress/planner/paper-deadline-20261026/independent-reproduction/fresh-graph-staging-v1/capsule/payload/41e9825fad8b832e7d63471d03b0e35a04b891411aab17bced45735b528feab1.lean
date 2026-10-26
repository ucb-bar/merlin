/- Candidate actual-pair validator. UNCOMPILED under the runtime hold.
Both modules are separate serialized terms. The decisive equation compares
full normalized code to the independently decoded AFTER module under a list
of bijective name swaps; no opaque whole-program equality is a premise. -/
import Merlin.GraphLoopsShapeAfterRename

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.After

def permuteModule : List (Nat × Nat) → Module → Module
  | [], module => module
  | (left, right) :: rest, module => swapModule left right (permuteModule rest module)

theorem complete_permuted {Value World : Type} (swaps : List (Nat × Nat))
    (meaning : Interpretation Value World) (module : Module) (name : String)
    (inputs : List (Runtime Value)) (before : World) (outputs : List (Runtime Value)) (after : World) :
    complete meaning module false name inputs before outputs after ↔
      complete meaning (permuteModule swaps module) false name inputs before outputs after := by
  induction swaps with
  | nil => rfl
  | cons pair rest induction =>
    exact induction.trans (complete_swapped pair.1 pair.2 meaning (permuteModule rest module) name inputs before outputs after)

structure Projection where
  bits : Nat
  before : Module
  after : Module
  swaps : List (Nat × Nat)
  beforeFrame : List String
  afterFrame : List String
  deriving DecidableEq, Repr

def Valid (projection : Projection) : Prop :=
  projection.before.Valid projection.bits ∧ projection.after.Valid projection.bits ∧
  projection.beforeFrame = projection.afterFrame ∧
  permuteModule projection.swaps projection.after = Epoch.normalizedModule projection.before

instance (projection : Projection) : Decidable (Valid projection) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

def check (projection : Projection) : Bool := decide (Valid projection)

theorem check_sound {projection : Projection} (accepted : check projection = true) : Valid projection :=
  of_decide_eq_true accepted

/-- Returned values and the entire world agree for possible complete outcomes.
The actual-after program is evaluated in ordinary mode, with its own raw names.
Parser/descriptor/primitive fidelity and unchanged header interpretation remain
explicit; no categorized error/divergence or native execution claim follows. -/
theorem checked_complete {Value World : Type} {projection : Projection} (accepted : check projection = true)
    (meaning : Interpretation Value World) (name : String) (inputs : List (Runtime Value))
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    complete meaning projection.before false name inputs before outputs after ↔
      complete meaning projection.after false name inputs before outputs after := by
  have valid := check_sound accepted
  have normalized := (projection.before.checked_complete projection.bits valid.1 meaning name inputs before outputs after).trans
    (Epoch.normalized_complete meaning projection.before name inputs before after outputs)
  have renamed := complete_permuted projection.swaps meaning projection.after name inputs before outputs after
  rw [valid.2.2.2] at renamed
  exact normalized.trans renamed.symm

theorem checked_context {Value World Observation : Type} {projection : Projection}
    (accepted : check projection = true) (meaning : Interpretation Value World)
    (context : (String → RegionMeaning Value World) → Observation) :
    context (complete meaning projection.before false) = context (complete meaning projection.after false) := by
  congr 1
  funext name inputs before outputs after
  exact propext (checked_complete accepted meaning name inputs before outputs after)

end Merlin.GraphLoopsShape.After
