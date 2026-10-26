/- A generated outer SCF executes a bounded dynamic inner loop per iteration.
UNCOMPILED candidate. This bridges two explicit operational relations, preserving
the complete state and all ordered body occurrences, including partial bodies.
Actual typed body/SSA binding is supplied separately, never assumed here. -/
import Merlin.GraphLoopsMemorySCFAfterMeaning

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsSegment

def Chunk {State : Type} (body : Nat → State → State → Prop) (upper step tile outer : Nat)
    (before after : State) : Prop :=
  ∃ trace, GraphLoopsSegment.Execution body (min (outer + step * tile) upper) step outer before trace after

theorem outer_to_segmented {State : Type} (body : Nat → State → State → Prop) (upper step tile : Nat)
    {lower : Nat} {before after : State} {trace : List Nat}
    (run : GraphLoopsSegment.Execution (Chunk body upper step tile) upper (step * tile) lower before trace after) :
    ∃ completeTrace, GraphLoopsSegment.Segmented body upper step tile lower before completeTrace after := by
  induction run with
  | done state stopped => exact ⟨[], .done state stopped⟩
  | more live chunk remaining induction =>
    obtain ⟨first, inner⟩ := chunk
    obtain ⟨rest, tail⟩ := induction
    exact ⟨first ++ rest, .more live inner tail⟩

theorem segmented_to_outer {State : Type} (body : Nat → State → State → Prop) (upper step tile : Nat)
    {lower : Nat} {before after : State} {trace : List Nat}
    (run : GraphLoopsSegment.Segmented body upper step tile lower before trace after) :
    ∃ outerTrace, GraphLoopsSegment.Execution (Chunk body upper step tile) upper (step * tile) lower before outerTrace after := by
  induction run with
  | done state stopped => exact ⟨[], .done state stopped⟩
  | @more lower initial middle final first rest live inner remaining induction =>
    obtain ⟨tail, next⟩ := induction
    exact ⟨lower :: tail, .more live ⟨first, inner⟩ next⟩

/-- The old trace theorem yields equality of possible complete final states.
No body purity, totality, commutation or numerical equation is used. -/
theorem outer_source_iff {State : Type} (body : Nat → State → State → Prop) (upper step tile lower : Nat)
    (positiveStep : 0 < step) (positiveTile : 0 < tile) (before after : State) :
    (∃ trace, GraphLoopsSegment.Execution (Chunk body upper step tile) upper (step * tile) lower before trace after) ↔
      ∃ trace, GraphLoopsSegment.Execution body upper step lower before trace after := by
  constructor
  · rintro ⟨trace, run⟩
    obtain ⟨full, segmented⟩ := outer_to_segmented body upper step tile run
    exact ⟨full, segmented_refines positiveStep segmented⟩
  · rintro ⟨trace, run⟩
    exact segmented_to_outer body upper step tile
      (source_refines body upper step tile positiveStep positiveTile lower before after trace run)

end Merlin.GraphLoopsMemorySCF.After
