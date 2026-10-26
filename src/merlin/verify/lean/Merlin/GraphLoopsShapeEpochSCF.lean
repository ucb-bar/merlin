/- Derive ordered whole-heap SCF correspondence from the per-iteration body.
This bridge keeps all body effects and possible outcomes; it does not postulate
whole-loop equality or enumerate the number of iterations. -/
import Merlin.GraphLoopsShapeEpochViews

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

abbrev Heap (Value : Type) := Merlin.GraphLoopsMemorySCF.Heap Value

def paired {Value Public : Type} (observed : Public) (heap : Heap Value) :
    List (Runtime Value) × MemoryWorld Value Public := ([], ⟨observed, heap⟩)

/-- An exact local body relation: the body restores its lexical carry boundary,
frames the public observation, and performs the stated private heap step. -/
def HeapBodyCorresponds {Value Public : Type} (body : RegionMeaning Value (MemoryWorld Value Public))
    (step : Nat → Heap Value → Heap Value → Prop) (observed : Public) : Prop :=
  ∀ index before outputs after,
    body [.index index] ⟨observed, before⟩ outputs after ↔
      outputs = [] ∧ after.observed = observed ∧ step index before after.heap

theorem heap_execution_forward {Value Public : Type}
    (body : RegionMeaning Value (MemoryWorld Value Public)) (step : Nat → Heap Value → Heap Value → Prop)
    (observed : Public) (bodyLaw : HeapBodyCorresponds body step observed)
    {upper stride lower : Nat} {before : Heap Value} {after : List (Runtime Value) × MemoryWorld Value Public}
    {trace : List Nat}
    (run : GraphLoopsSegment.Execution (loopBody body) upper stride lower (paired observed before) trace after) :
    ∃ final, GraphLoopsSegment.Execution step upper stride lower before trace final ∧ after = paired observed final := by
  apply GraphLoopsMemorySCF.SCF_simulate (fun state heap => state = paired observed heap) ?_ rfl run
  intro index first last heap same transition
  subst first
  have actual : body [.index index] ⟨observed, heap⟩ last.1 last.2 := transition
  obtain ⟨empty, observedEqual, changed⟩ := (bodyLaw index heap last.1 last.2).mp actual
  refine ⟨last.2.heap, changed, ?_⟩
  rcases last with ⟨values, ⟨observer, cells⟩⟩
  simp only at empty observedEqual
  simp only [paired, Prod.mk.injEq]
  exact ⟨empty, by cases observedEqual; rfl⟩

theorem heap_execution_backward {Value Public : Type}
    (body : RegionMeaning Value (MemoryWorld Value Public)) (step : Nat → Heap Value → Heap Value → Prop)
    (observed : Public) (bodyLaw : HeapBodyCorresponds body step observed)
    {upper stride lower : Nat} {before after : Heap Value} {trace : List Nat}
    (run : GraphLoopsSegment.Execution step upper stride lower before trace after) :
    GraphLoopsSegment.Execution (loopBody body) upper stride lower (paired observed before) trace (paired observed after) := by
  have simulate : ∀ index first last other, other = paired observed first → step index first last →
      ∃ next, loopBody body index other next ∧ next = paired observed last := by
    intro index first last other same transition
    subst other
    refine ⟨paired observed last, ?_, rfl⟩
    exact (bodyLaw index first [] ⟨observed, last⟩).mpr ⟨rfl, rfl, transition⟩
  obtain ⟨final, translated, same⟩ := GraphLoopsMemorySCF.SCF_simulate
    (fun heap state => state = paired observed heap) simulate rfl run
  exact same ▸ translated

theorem scf_controls {Value World : Type} (body : RegionMeaning Value World) (lower upper stride : Nat)
    (before after : World) (outputs : List (Runtime Value)) :
    SCF body [.index lower, .index upper, .index stride] before outputs after ↔
      0 < stride ∧ ∃ trace, GraphLoopsSegment.Execution (loopBody body) upper stride lower ([], before) trace (outputs, after) := by
  constructor
  · rintro ⟨actualLower, actualUpper, actualStride, carried, controls, positive, trace, run⟩
    cases controls
    exact ⟨positive, trace, run⟩
  · rintro ⟨positive, trace, run⟩
    exact ⟨lower, upper, stride, [], rfl, positive, trace, run⟩

/-- This applies to arbitrary trip counts and the exact same ordered trace. -/
theorem scf_heap_iff {Value Public : Type}
    (body : RegionMeaning Value (MemoryWorld Value Public)) (step : Nat → Heap Value → Heap Value → Prop)
    (observed : Public) (bodyLaw : HeapBodyCorresponds body step observed)
    (lower upper stride : Nat) (before : Heap Value) (outputs : List (Runtime Value)) (after : MemoryWorld Value Public) :
    SCF body [.index lower, .index upper, .index stride] ⟨observed, before⟩ outputs after ↔
      0 < stride ∧ outputs = [] ∧ after.observed = observed ∧
        ∃ trace, GraphLoopsSegment.Execution step upper stride lower before trace after.heap := by
  rw [scf_controls]
  constructor
  · rintro ⟨positive, trace, run⟩
    obtain ⟨final, translated, same⟩ := heap_execution_forward body step observed bodyLaw run
    have parts := Prod.mk.inj same
    refine ⟨positive, parts.1, ?_, trace, ?_⟩
    · exact congrArg GraphLoopsReuse.Memory.observed parts.2
    · simpa only [congrArg GraphLoopsReuse.Memory.heap parts.2] using translated
  · rintro ⟨positive, rfl, observedEqual, trace, run⟩
    refine ⟨positive, trace, ?_⟩
    have result := heap_execution_backward body step observed bodyLaw run
    rcases after with ⟨observer, heap⟩
    change observer = observed at observedEqual
    cases observedEqual
    exact result

end Merlin.GraphLoopsShape.Epoch
