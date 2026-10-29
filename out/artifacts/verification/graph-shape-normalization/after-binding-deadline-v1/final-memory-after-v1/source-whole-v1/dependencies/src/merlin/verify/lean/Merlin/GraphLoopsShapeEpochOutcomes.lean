/- Caller observations of complete typed executions and the derived epoch
interpretation. Private allocation names, SSA ghosts and private cells are not
caller observations. The entire caller partition, returns and user state are. -/
import Merlin.GraphLoopsShapeEpochFunction

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

def visibleHeap {Value : Type} (bound : Nat) (heap : Heap Value) : Heap Value :=
  fun object index => if object < bound then none else heap object index

structure Observation (Value User : Type) where
  returned : List (Runtime Value)
  heap : Heap Value
  user : User

def observeWorld {Value User : Type} (bound : Nat) (returned : List (Runtime Value))
    (world : MemoryWorld Value (AllocationPublic User)) : Observation Value User :=
  ⟨returned, visibleHeap bound world.heap, world.observed.observed⟩

def observePublic {Value User : Type} (bound : Nat) (state : PublicState Value User) : Observation Value User :=
  ⟨state.returned, visibleHeap bound state.callerHeap, state.observed⟩

theorem visible_merge {Value : Type} (bound : Nat) (localHeap caller : Heap Value) :
    visibleHeap bound (mergeHeap bound localHeap caller) = visibleHeap bound caller := by
  funext object index
  by_cases inside : object < bound <;> simp only [visibleHeap, mergeHeap, inside, ↓reduceIte]

theorem observe_clock {Value User : Type} (bound clock : Nat)
    (memory : GraphLoopsReuse.Memory Value (PublicState Value User)) :
    observeWorld bound memory.observed.returned (clockWorld bound clock memory) = observePublic bound memory.observed := by
  simp only [observeWorld, observePublic, clockWorld, visible_merge]

def graphOutcomes {Value User : Type} (bound : Nat)
    (meaning : Interpretation Value (MemoryWorld Value (AllocationPublic User)))
    (module : Module) (name : String) (inputs : List (Runtime Value)) (heap : Heap Value) (user : User)
    (observed : Observation Value User) : Prop :=
  ∃ returned world, complete meaning module false name inputs ⟨⟨0, user⟩, heap⟩ returned world ∧
    observeWorld bound returned world = observed

def epochOutcomes {Value User : Type} (plan : PlanContext) (kernel : Kernel) (scalar : BinaryMeaning Value)
    (candidate segmented : Bool) (before : GraphLoopsReuse.Memory Value (PublicState Value User))
    (observed : Observation Value User) : Prop :=
  ∃ state, GraphLoopsMemorySCF.outcomes plan.objects plan.configs (plan.oldMeaning scalar) candidate segmented
    kernel.instructions before state ∧ observePublic plan.objects.length state = observed

/-- This is an equality of complete observation sets, not an identification of
allocation failures, native pointer identities, private heaps or error classes. -/
theorem Kernel.observations {Value User : Type} (plan : PlanContext) (planValid : plan.Valid)
    (kernel : Kernel) (facts : kernel.Facts plan) (scalar : BinaryMeaning Value)
    (inputs : List (Runtime Value)) (typed : ValuesHold kernel.function.parameters inputs)
    (callers : CallersValid plan.objects.length plan.callers (assign emptyEnv kernel.function.parameters inputs))
    (heap : Heap Value) (user : User) (observed : Observation Value User) :
    graphOutcomes plan.objects.length (graphMeaning plan.opcodes scalar) plan.module kernel.function.name
      inputs heap user observed ↔
      epochOutcomes plan kernel scalar false false (entryMemory plan kernel.function inputs heap user) observed := by
  constructor
  · rintro ⟨returned, world, run, observation⟩
    obtain ⟨after, oldRun, returnedEqual, worldEqual⟩ :=
      (kernel.complete_equivalent plan planValid facts scalar inputs typed callers heap user returned world).mp run
    refine ⟨after.observed, ⟨after, oldRun, rfl⟩, ?_⟩
    rw [returnedEqual, worldEqual, observe_clock] at observation
    exact observation
  · rintro ⟨state, ⟨after, run, stateEqual⟩, observation⟩
    refine ⟨after.observed.returned, clockWorld plan.objects.length (finalClock 0 kernel.commands) after, ?_, ?_⟩
    · exact (kernel.complete_equivalent plan planValid facts scalar inputs typed callers heap user _ _).mpr
        ⟨after, run, rfl, rfl⟩
    · rw [observe_clock, stateEqual]
      exact observation

end Merlin.GraphLoopsShape.Epoch
