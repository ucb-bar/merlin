/- Explicit same-SCF composition for an actual no-tail bulk unroll followed
by segmentation of THAT bulk body. The middle interpretation is installed as
the original body's logged factor grouping; no opaque interpretations are
silently equated. Raw-SCF projection fidelity remains outside this theorem. -/
import Merlin.GraphLoopsUnrollModel

set_option autoImplicit false

namespace Merlin.GraphLoopsUnroll.Composition

open GraphLoopsSegment

def Compatible (u : Model.Projection) (s : GraphLoopsSegment.Model.Projection) : Prop :=
  u.bounds.factor < u.bounds.count ∧ u.bounds.remainder = 0 ∧ s.sourceBody = u.bulkBody ∧
  s.bounds.lower = u.bounds.lower ∧ s.bounds.upper = u.targetBound ∧ s.bounds.step = u.targetStride ∧
  s.carryTypes = u.sourceResults ∧ s.sourceInitial = u.targetInitial

instance (u : Model.Projection) (s : GraphLoopsSegment.Model.Projection) : Decidable (Compatible u s) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))

def groupInterpretation {σ : Type} (u : Model.Projection) (interpret : String → Nat → σ → σ → Prop)
    (key : String) : Nat → Logged σ → Logged σ → Prop :=
  if key = u.bulkBody then GroupBody (interpret u.sourceBody) u.bounds.step u.bounds.factor else fun _ _ _ => False

theorem no_tail_bulk_iff {σ : Type} (body : Nat → σ → σ → Prop) {b : Bounds}
    (valid : b.Valid) (noTail : b.remainder = 0) (initial final : σ) (trace : List Nat) :
    Execution body b.upper b.step b.lower initial trace final ↔
      ∃ bases, Execution (GroupBody body b.step b.factor) (b.lower + (b.step * b.factor) * b.groups)
        (b.step * b.factor) b.lower (initial, []) bases (final, trace) := by
  have partition := valid.2.2.2.2.2.1
  have count : b.groups * b.factor = b.count := by omega
  have positiveFactor : 0 < b.factor := by have := valid.2.2.2.1; omega
  rw [bulk_iff_repeat body _ _ _ _ valid.2.2.1 positiveFactor]
  simp only [List.nil_append, count]
  constructor
  · intro run
    exact ⟨trace, (repeat_iff_execution body _ _ _ _ _ _ _ valid.2.2.2.2.1).mpr run, rfl⟩
  · rintro ⟨originalTrace, run, equal⟩
    subst originalTrace
    exact (repeat_iff_execution body _ _ _ _ _ _ _ valid.2.2.2.2.1).mp run

/-- Actual model-to-model composition, carrying the complete original-body
occurrence log and ALL original state, not just the bulk-loop IV trace. -/
theorem checked_composed {u : Model.Projection} {s : GraphLoopsSegment.Model.Projection}
    (unrollAccepted : Model.check u = true) (segmentAccepted : GraphLoopsSegment.Model.check s = true)
    (compatible : Compatible u s) {σ : Type} (interpret : String → Nat → σ → σ → Prop)
    (initial final : σ) (trace : List Nat) :
    Model.original u interpret initial trace final ↔
      ∃ bases, GraphLoopsSegment.Model.transformed s (groupInterpretation u interpret)
        (initial, []) bases (final, trace) := by
  obtain ⟨_, noTail, key, lower, upper, step, _, _⟩ := compatible
  obtain ⟨⟨valid, stride, bound⟩, _⟩ := Model.check_sound unrollAccepted
  have bulk := no_tail_bulk_iff (interpret u.sourceBody) valid noTail initial final trace
  change Execution (interpret u.sourceBody) _ _ _ _ _ _ ↔ _
  rw [bulk]
  apply exists_congr
  intro bases
  have segment := GraphLoopsSegment.Model.checked_complete_trace segmentAccepted
    (groupInterpretation u interpret) (initial, []) (final, trace) bases
  simpa only [GraphLoopsSegment.Model.original, groupInterpretation, key, ↓reduceIte,
    lower, upper, step, stride, bound, Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm] using segment

end Merlin.GraphLoopsUnroll.Composition
