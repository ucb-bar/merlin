/- Partial relational semantics for private allocation-changing translation.
This is a NEW explicit memory model, NOT the frozen MerlinVerify evaluator or an
asserted raw SCF bridge. Initializer semantics is explicitly admitted by key:
only its successful complete rectangular result supplies a value at every cell.
No existence, totality or numeric law is imposed on retained/initializer actions. -/
import Merlin.GraphLoopsReuseProtocol

set_option autoImplicit false

namespace Merlin.GraphLoopsReuse

structure Primitives (Value Public : Type) where
  initialization : String → Public → (Nat → Value) → Prop
  address : String → Public → Nat → Prop
  load : String → Value → Public → Public → Prop
  store : String → Public → Nat → Value → Prop
  dimension : String → List Nat → Public → Public → Prop
  retained : String → Public → Public → Prop

def addressOf (objects : Objects) (candidate : Bool) (object : Nat) : Nat :=
  if candidate then slotOf objects object else object

def clearMemory {Value Public : Type} (objects : Objects) (candidate : Bool) (object : Nat) (retained : Bool)
    (memory : Memory Value Public) : Memory Value Public :=
  if candidate && !retained then memory else
    ⟨memory.observed, replaceObject memory.heap (addressOf objects candidate object) (fun _ => none)⟩

def initializeMemory {Value Public : Type} (objects : Objects) (candidate : Bool) (object : Nat)
    (values : Nat → Value) (memory : Memory Value Public) : Memory Value Public :=
  ⟨memory.observed, replaceObject memory.heap (addressOf objects candidate object)
    (initializedCells (extentOf objects object) values)⟩

def storeMemory {Value Public : Type} (objects : Objects) (candidate : Bool) (object index : Nat)
    (value : Value) (memory : Memory Value Public) : Memory Value Public :=
  ⟨memory.observed, replaceCell memory.heap (addressOf objects candidate object) index value⟩

/-- Allocations/frees are total, private and unobservable HERE. Actual allocator
conformance (failure, exhaustion, identity, async/ABI) is separately required. -/
inductive Step {Value Public : Type} (objects : Objects) (primitives : Primitives Value Public) (candidate : Bool) :
    Instruction → Memory Value Public → Memory Value Public → Prop where
  | allocate (object : Nat) (retained : Bool) (initial : Memory Value Public) :
      Step objects primitives candidate (.allocate object retained) initial (clearMemory objects candidate object retained initial)
  | initialized (object : Nat) (key : String) (initial : Memory Value Public) (values : Nat → Value)
      (computed : primitives.initialization key initial.observed values) :
      Step objects primitives candidate (.initialize object key) initial (initializeMemory objects candidate object values initial)
  | load (object index : Nat) (key : String) (initial : Memory Value Public) (value : Value) (after : Public)
      (address : primitives.address key initial.observed index) (inside : index < extentOf objects object)
      (read : initial.heap (addressOf objects candidate object) index = some value)
      (observed : primitives.load key value initial.observed after) :
      Step objects primitives candidate (.load object key) initial ⟨after, initial.heap⟩
  | store (object index : Nat) (key : String) (initial : Memory Value Public) (value : Value)
      (computed : primitives.store key initial.observed index value) (inside : index < extentOf objects object) :
      Step objects primitives candidate (.store object key) initial (storeMemory objects candidate object index value initial)
  | dimension (object : Nat) (key : String) (initial : Memory Value Public) (after : Public)
      (observed : primitives.dimension key (objectAt objects object).shape initial.observed after) :
      Step objects primitives candidate (.dimension object key) initial ⟨after, initial.heap⟩
  | release (object : Nat) (retained : Bool) (initial : Memory Value Public) :
      Step objects primitives candidate (.release object retained) initial (clearMemory objects candidate object retained initial)
  | retained (key : String) (initial : Memory Value Public) (after : Public)
      (observed : primitives.retained key initial.observed after) :
      Step objects primitives candidate (.retained key) initial ⟨after, initial.heap⟩

/-- One actual logical instruction is simulated forward, including all currently
live values. Partial primitive success is transferred, never inferred. -/
theorem step_forward {Value Public : Type} {objects : Objects} {status : Status} {instruction : Instruction}
    {primitives : Primitives Value Public} {source target sourceAfter : Memory Value Public}
    (admitted : Admitted objects status instruction)
    (related : Related (slotOf objects) (extentOf objects) status.ready source target)
    (run : Step objects primitives false instruction source sourceAfter) :
    ∃ targetAfter, Step objects primitives true instruction target targetAfter ∧
      Related (slotOf objects) (extentOf objects) (next objects status instruction).ready sourceAfter targetAfter := by
  cases run with
  | allocate object retained initial =>
    refine ⟨_, .allocate object retained target, ?_⟩
    have result := related.allocate object admitted.2.1.2.2.1 admitted.allocate_separate retained
    cases retained <;> simpa [clearMemory, addressOf, next] using result
  | initialized object key initial values computed =>
    have targetComputed : primitives.initialization key target.observed values := related.1 ▸ computed
    refine ⟨_, .initialized object key target values targetComputed, ?_⟩
    exact related.initialize object values admitted.initialize_separate
  | load object index key initial value after address inside read observed =>
    have targetRead : target.heap (slotOf objects object) index = some value :=
      (related.read admitted.2.1 inside).symm.trans read
    refine ⟨_, .load object index key target value after (related.1 ▸ address) inside targetRead (related.1 ▸ observed), ?_⟩
    exact related.observed_step after
  | store object index key initial value computed inside =>
    refine ⟨_, .store object index key target value (related.1 ▸ computed) inside, ?_⟩
    exact related.store admitted.1.ready_unique admitted.2.1 index value
  | dimension object key initial after observed =>
    exact ⟨_, .dimension object key target after (related.1 ▸ observed), related.observed_step after⟩
  | release object retained initial =>
    refine ⟨_, .release object retained target, ?_⟩
    have result := related.free admitted.1.ready_unique object admitted.2.1 retained
    cases retained <;> simpa [clearMemory, addressOf, next] using result
  | retained key initial after observed =>
    exact ⟨_, .retained key target after (related.1 ▸ observed), related.observed_step after⟩

/-- Reverse simulation rules out stale-content outcomes in the candidate.
Every load requires an epoch proved completely initialized by the protocol. -/
theorem step_backward {Value Public : Type} {objects : Objects} {status : Status} {instruction : Instruction}
    {primitives : Primitives Value Public} {source target targetAfter : Memory Value Public}
    (admitted : Admitted objects status instruction)
    (related : Related (slotOf objects) (extentOf objects) status.ready source target)
    (run : Step objects primitives true instruction target targetAfter) :
    ∃ sourceAfter, Step objects primitives false instruction source sourceAfter ∧
      Related (slotOf objects) (extentOf objects) (next objects status instruction).ready sourceAfter targetAfter := by
  cases run with
  | allocate object retained initial =>
    refine ⟨_, .allocate object retained source, ?_⟩
    have result := related.allocate object admitted.2.1.2.2.1 admitted.allocate_separate retained
    cases retained <;> simpa [clearMemory, addressOf, next] using result
  | initialized object key initial values computed =>
    refine ⟨_, .initialized object key source values (related.1.symm ▸ computed), ?_⟩
    exact related.initialize object values admitted.initialize_separate
  | load object index key initial value after address inside read observed =>
    have sourceRead : source.heap object index = some value := (related.read admitted.2.1 inside).trans read
    refine ⟨_, .load object index key source value after (related.1.symm ▸ address) inside sourceRead
      (related.1.symm ▸ observed), ?_⟩
    exact related.observed_step after
  | store object index key initial value computed inside =>
    refine ⟨_, .store object index key source value (related.1.symm ▸ computed) inside, ?_⟩
    exact related.store admitted.1.ready_unique admitted.2.1 index value
  | dimension object key initial after observed =>
    exact ⟨_, .dimension object key source after (related.1.symm ▸ observed), related.observed_step after⟩
  | release object retained initial =>
    refine ⟨_, .release object retained source, ?_⟩
    have result := related.free admitted.1.ready_unique object admitted.2.1 retained
    cases retained <;> simpa [clearMemory, addressOf, next] using result
  | retained key initial after observed =>
    exact ⟨_, .retained key source after (related.1.symm ▸ observed), related.observed_step after⟩

inductive Execution {Value Public : Type} (objects : Objects) (primitives : Primitives Value Public) (candidate : Bool) :
    List Instruction → Memory Value Public → Memory Value Public → Prop where
  | done (initial : Memory Value Public) : Execution objects primitives candidate [] initial initial
  | more {instruction : Instruction} {rest : List Instruction} {initial middle final : Memory Value Public}
      (step : Step objects primitives candidate instruction initial middle)
      (remaining : Execution objects primitives candidate rest middle final) :
      Execution objects primitives candidate (instruction :: rest) initial final

/-- Complete live-memory and caller observation preservation, forward direction. -/
theorem execution_forward {Value Public : Type} {objects : Objects} {primitives : Primitives Value Public}
    {instructions : List Instruction} {status : Status} (safe : Safe objects status instructions) :
    ∀ source target sourceAfter, Related (slotOf objects) (extentOf objects) status.ready source target →
      Execution objects primitives false instructions source sourceAfter →
      ∃ targetAfter, Execution objects primitives true instructions target targetAfter ∧ sourceAfter.observed = targetAfter.observed := by
  induction safe with
  | done =>
    intro source target sourceAfter related run
    cases run
    exact ⟨target, .done target, related.1⟩
  | step admitted remaining induction =>
    intro source target sourceAfter related run
    cases run with
    | more first suffix =>
      obtain ⟨middle, firstTarget, nextRelated⟩ := step_forward admitted related first
      obtain ⟨targetAfter, suffixTarget, observed⟩ := induction _ middle _ nextRelated suffix
      exact ⟨targetAfter, .more firstTarget suffixTarget, observed⟩

theorem execution_backward {Value Public : Type} {objects : Objects} {primitives : Primitives Value Public}
    {instructions : List Instruction} {status : Status} (safe : Safe objects status instructions) :
    ∀ source target targetAfter, Related (slotOf objects) (extentOf objects) status.ready source target →
      Execution objects primitives true instructions target targetAfter →
      ∃ sourceAfter, Execution objects primitives false instructions source sourceAfter ∧ sourceAfter.observed = targetAfter.observed := by
  induction safe with
  | done =>
    intro source target targetAfter related run
    cases run
    exact ⟨source, .done source, related.1⟩
  | step admitted remaining induction =>
    intro source target targetAfter related run
    cases run with
    | more first suffix =>
      obtain ⟨middle, firstSource, nextRelated⟩ := step_backward admitted related first
      obtain ⟨sourceAfter, suffixSource, observed⟩ := induction middle _ _ nextRelated suffix
      exact ⟨sourceAfter, .more firstSource suffixSource, observed⟩

/-- Equality of ALL possible public outcomes in the declared allocator model.
Public may contain returned scalars, caller heap and the complete retained effect
trace. Source success/primitive totality is never an extra premise. -/
theorem checked_observations {Value Public : Type} (objects : Objects) (instructions : List Instruction)
    (accepted : checkProgram objects emptyStatus instructions = true) (primitives : Primitives Value Public)
    (source target : Memory Value Public) (samePublic : source.observed = target.observed) (outcome : Public) :
    (∃ final, Execution objects primitives false instructions source final ∧ final.observed = outcome) ↔
    (∃ final, Execution objects primitives true instructions target final ∧ final.observed = outcome) := by
  have safe := checkProgram_sound objects instructions emptyStatus accepted
  have related : Related (slotOf objects) (extentOf objects) emptyStatus.ready source target := ⟨samePublic, by simp [emptyStatus]⟩
  constructor
  · rintro ⟨final, run, observed⟩
    obtain ⟨targetFinal, targetRun, equal⟩ := execution_forward safe source target final related run
    exact ⟨targetFinal, targetRun, equal.symm.trans observed⟩
  · rintro ⟨final, run, observed⟩
    obtain ⟨sourceFinal, sourceRun, equal⟩ := execution_backward safe source target final related run
    exact ⟨sourceFinal, sourceRun, equal.trans observed⟩

end Merlin.GraphLoopsReuse
