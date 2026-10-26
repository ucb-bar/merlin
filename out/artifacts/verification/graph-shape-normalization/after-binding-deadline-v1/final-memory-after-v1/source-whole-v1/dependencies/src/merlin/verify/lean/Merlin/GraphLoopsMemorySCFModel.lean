/- Recomputed finite facts for complete actual reuse→nested segmentation.
Python supplies full parsed trees and scoped correspondence. Acceptance checks
all registry/control links; operational correctness is the existing new root. -/
import Merlin.GraphLoopsMemorySCFModes
import Merlin.GraphLoopsSegmentModel

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.Model

structure Boundary where
  config : Configuration
  segment : GraphLoopsSegment.Model.Projection
  sourceInitializer : String
  middleInitializer : String
  segmentationSourceInitializer : String
  deriving DecidableEq, Repr

structure Projection where
  reuse : GraphLoopsReuse.Model.Projection
  boundaries : List Boundary
  reuseMiddle : String
  segmentationMiddle : String
  deriving DecidableEq, Repr

def configs (p : Projection) : List Configuration := p.boundaries.map Boundary.config

def Boundary.Valid (reuse : GraphLoopsReuse.Model.Projection) (boundary : Boundary) : Prop :=
  boundary.config.Valid reuse.objects ∧
  boundary.sourceInitializer = boundary.config.key ∧
  boundary.middleInitializer = boundary.segmentationSourceInitializer ∧
  GraphLoopsSegment.Model.Numeric boundary.segment ∧ GraphLoopsSegment.Model.Structural boundary.segment ∧
  boundary.segment.bounds.bits = reuse.bits ∧ boundary.segment.bounds.lower = 0 ∧ boundary.segment.bounds.step = 1 ∧
  boundary.segment.bounds.upper = boundary.config.program.rows ∧ boundary.segment.bounds.tile = boundary.config.tile ∧
  ∃ init ∈ reuse.initializers, init.object = boundary.config.object ∧ init.sourceBody = boundary.config.key ∧
    init.ivs = boundary.config.program.ivs ∧ init.indices = boundary.config.program.leaf.indices

instance (reuse : GraphLoopsReuse.Model.Projection) (boundary : Boundary) : Decidable (boundary.Valid reuse) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))

def Bound (p : Projection) : GraphLoopsReuse.Instruction → Prop
  | .initialize object key => ∃ config ∈ configs p, config.object = object ∧ config.key = key
  | _ => True

instance (p : Projection) (instruction : GraphLoopsReuse.Instruction) : Decidable (Bound p instruction) := by
  cases instruction <;> unfold Bound <;> infer_instance

def Valid (p : Projection) : Prop :=
  p.boundaries ≠ [] ∧ p.reuseMiddle = p.segmentationMiddle ∧
  p.boundaries.length = p.reuse.initializers.length ∧
  (∀ boundary ∈ p.boundaries, boundary.Valid p.reuse) ∧
  (∀ instruction ∈ p.reuse.instructions, Bound p instruction) ∧
  ∀ first ∈ configs p, ∀ second ∈ configs p, first.key = second.key → first = second

instance (p : Projection) : Decidable (Valid p) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))

def check (p : Projection) : Bool := decide (Valid p) && GraphLoopsReuse.Model.check p.reuse

theorem check_sound {p : Projection} (accepted : check p = true) :
    Valid p ∧ GraphLoopsReuse.Model.Valid p.reuse ∧ GraphLoopsReuse.Safe p.reuse.objects GraphLoopsReuse.emptyStatus p.reuse.instructions := by
  obtain ⟨valid, checked⟩ := Bool.and_eq_true_iff.mp accepted
  exact ⟨of_decide_eq_true valid, GraphLoopsReuse.Model.check_sound checked⟩

theorem checked_configs {p : Projection} (accepted : check p = true) :
    ∀ config ∈ configs p, config.Valid p.reuse.objects := by
  intro config member
  obtain ⟨boundary, boundaryMember, rfl⟩ := List.mem_map.mp member
  exact ((check_sound accepted).1.2.2.2.1 boundary boundaryMember).1

/-- Source, exact middle and final all use the SAME operational interpretation.
Multiple changed boundaries compose by instruction-list congruence. -/
theorem checked_chain {p : Projection} (accepted : check p = true) {Value Public : Type}
    (meaning : Interpretation Value Public) (source middle : GraphLoopsReuse.Memory Value Public)
    (same : source.observed = middle.observed) (outcome : Public) :
    (outcomes p.reuse.objects (configs p) meaning false false p.reuse.instructions source outcome ↔
      outcomes p.reuse.objects (configs p) meaning true false p.reuse.instructions middle outcome) ∧
    (outcomes p.reuse.objects (configs p) meaning true false p.reuse.instructions middle outcome ↔
      outcomes p.reuse.objects (configs p) meaning true true p.reuse.instructions middle outcome) ∧
    (outcomes p.reuse.objects (configs p) meaning false false p.reuse.instructions source outcome ↔
      outcomes p.reuse.objects (configs p) meaning true true p.reuse.instructions middle outcome) := by
  have safe := (Bool.and_eq_true_iff.mp (Bool.and_eq_true_iff.mp accepted).2).2
  exact checked_reuse_then_segment p.reuse.objects (configs p) p.reuse.instructions (checked_configs accepted) safe meaning source middle same outcome

theorem checked_complete {p : Projection} (accepted : check p = true) {Value Public : Type}
    (meaning : Interpretation Value Public) (source middle : GraphLoopsReuse.Memory Value Public)
    (same : source.observed = middle.observed) (outcome : Public) :
    outcomes p.reuse.objects (configs p) meaning false false p.reuse.instructions source outcome ↔
      outcomes p.reuse.objects (configs p) meaning true true p.reuse.instructions middle outcome :=
  (checked_chain accepted meaning source middle same outcome).2.2

/-- Arbitrary common callers may observe the complete public outcome set.
The mapping from actual unchanged functions/callers is trusted full-tree Python. -/
theorem checked_context {p : Projection} (accepted : check p = true) {Value Public Outcome : Type}
    (meaning : Interpretation Value Public) (source middle : GraphLoopsReuse.Memory Value Public)
    (same : source.observed = middle.observed) (context : (Public → Prop) → Outcome) :
    context (outcomes p.reuse.objects (configs p) meaning false false p.reuse.instructions source) =
      context (outcomes p.reuse.objects (configs p) meaning true true p.reuse.instructions middle) := by
  have equal : outcomes p.reuse.objects (configs p) meaning false false p.reuse.instructions source =
      outcomes p.reuse.objects (configs p) meaning true true p.reuse.instructions middle := by
    funext outcome
    exact propext (checked_complete accepted meaning source middle same outcome)
  rw [equal]

end Merlin.GraphLoopsMemorySCF.Model
