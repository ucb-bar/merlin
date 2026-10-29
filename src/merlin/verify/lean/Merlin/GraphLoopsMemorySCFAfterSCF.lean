/- UNCOMPILED. Generated signed index instructions are justified only at LIVE
iterations. This bridge does not require their definedness at every natural
number. It preserves the entire heap, caller world and ordered trace.
-/
import Merlin.GraphLoopsMemorySCFAfterInitializer

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

/-- Per-step simulation restricted to indices actually executed by this loop.
No complete-loop equivalence is accepted as a premise.
-/
theorem execution_simulate_live {First Second : Type}
    (firstBody : Nat → First → First → Prop) (secondBody : Nat → Second → Second → Prop)
    (related : First → Second → Prop) (upper stride : Nat)
    (localStep : ∀ index, index < upper → ∀ before after other,
      related before other → firstBody index before after →
      ∃ next, secondBody index other next ∧ related after next)
    {lower : Nat} {first final : First} {other : Second} {trace : List Nat}
    (initial : related first other)
    (run : GraphLoopsSegment.Execution firstBody upper stride lower first trace final) :
    ∃ next, GraphLoopsSegment.Execution secondBody upper stride lower other trace next ∧ related final next := by
  induction run generalizing other with
  | done state stopped => exact ⟨other, .done other stopped, initial⟩
  | more live step remaining induction =>
    obtain ⟨middle, transition, nextRelated⟩ := localStep _ live _ _ other initial step
    obtain ⟨last, tail, finalRelated⟩ := induction nextRelated
    exact ⟨last, .more live transition tail, finalRelated⟩

theorem execution_live_iff {State : Type}
    (first second : Nat → State → State → Prop) (upper stride : Nat)
    (localStep : ∀ index, index < upper → ∀ before after,
      first index before after ↔ second index before after)
    (lower : Nat) (before after : State) (trace : List Nat) :
    GraphLoopsSegment.Execution first upper stride lower before trace after ↔
      GraphLoopsSegment.Execution second upper stride lower before trace after := by
  constructor
  · intro run
    induction run with
    | done state stopped => exact .done state stopped
    | more live step remaining induction =>
      exact .more live ((localStep _ live _ _).mp step) induction
  · intro run
    induction run with
    | done state stopped => exact .done state stopped
    | more live step remaining induction =>
      exact .more live ((localStep _ live _ _).mpr step) induction

def LiveHeapBodyCorresponds {Value Public : Type}
    (body : RegionMeaning Value (MemoryWorld Value Public))
    (step : Nat → Epoch.Heap Value → Epoch.Heap Value → Prop) (observed : Public) (upper : Nat) : Prop :=
  ∀ index, index < upper → ∀ before outputs after,
    body [.index index] ⟨observed, before⟩ outputs after ↔
      outputs = [] ∧ after.observed = observed ∧ step index before after.heap

/-- The actual SCF loop forwards empty carries and the COMPLETE caller/heap
state. Index-action bounds can be proved from `index < upper`, not postulated
outside the executed domain. Empty loops require no body definedness.
-/
theorem scf_live_heap_iff {Value Public : Type}
    (body : RegionMeaning Value (MemoryWorld Value Public))
    (step : Nat → Epoch.Heap Value → Epoch.Heap Value → Prop) (observed : Public)
    (lower upper stride : Nat)
    (localStep : LiveHeapBodyCorresponds body step observed upper)
    (before : Epoch.Heap Value) (outputs : List (Runtime Value)) (after : MemoryWorld Value Public) :
    SCF body [.index lower, .index upper, .index stride] ⟨observed, before⟩ outputs after ↔
      0 < stride ∧ outputs = [] ∧ after.observed = observed ∧
        ∃ trace, GraphLoopsSegment.Execution step upper stride lower before trace after.heap := by
  rw [Epoch.scf_controls]
  constructor
  · rintro ⟨positive, trace, run⟩
    have simulate : ∀ index, index < upper → ∀ first last heap,
        first = Epoch.paired observed heap → loopBody body index first last →
        ∃ next, step index heap next ∧ last = Epoch.paired observed next := by
      intro index live first last heap same transition
      subst first
      have actual : body [.index index] ⟨observed, heap⟩ last.1 last.2 := transition
      obtain ⟨empty, samePublic, changed⟩ := (localStep index live heap last.1 last.2).mp actual
      refine ⟨last.2.heap, changed, ?_⟩
      rcases last with ⟨values, ⟨observer, cells⟩⟩
      simp only at empty samePublic
      simp only [Epoch.paired, Prod.mk.injEq]
      exact ⟨empty, by cases samePublic; rfl⟩
    obtain ⟨final, translated, same⟩ := execution_simulate_live (loopBody body) step
      (fun state heap => state = Epoch.paired observed heap) upper stride simulate rfl run
    have parts := Prod.mk.inj same
    refine ⟨positive, parts.1, ?_, trace, ?_⟩
    · exact congrArg GraphLoopsReuse.Memory.observed parts.2
    · simpa only [congrArg GraphLoopsReuse.Memory.heap parts.2] using translated
  · rintro ⟨positive, rfl, samePublic, trace, run⟩
    have simulate : ∀ index, index < upper → ∀ first last other,
        other = Epoch.paired observed first → step index first last →
        ∃ next, loopBody body index other next ∧ next = Epoch.paired observed last := by
      intro index live first last other same transition
      subst other
      exact ⟨Epoch.paired observed last,
        (localStep index live first [] ⟨observed, last⟩).mpr ⟨rfl, rfl, transition⟩, rfl⟩
    obtain ⟨final, translated, same⟩ := execution_simulate_live step (loopBody body)
      (fun heap state => state = Epoch.paired observed heap) upper stride simulate rfl run
    refine ⟨positive, trace, ?_⟩
    rcases after with ⟨observer, heap⟩
    change observer = observed at samePublic
    cases samePublic
    simpa only [same, Epoch.paired] using translated

end Merlin.GraphLoopsMemorySCF.After
