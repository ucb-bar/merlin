/- UNCOMPILED. Ordered composition of LOCAL expansion simulations. This file
does not assert that ExpansionBinding discharges LocalSimulation: that separate
checked-command derivation must be connected before any whole-pass claim.
No whole-run equality is a premise. Syntax length, not tripcount, drives induction.
-/
import Merlin.GraphLoopsMemorySCFAfterInvariant
import Merlin.GraphLoopsMemorySCFAfterScopedAccess
import Merlin.GraphLoopsMemorySCFAfterScopedAllocation
import Merlin.GraphLoopsMemorySCFAfterTransport
import Merlin.GraphLoopsMemorySCFAfterInitializerLink

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

def physicalFinalClock : Nat → List Expansion → Nat
  | clock, [] => clock
  | clock, expansion :: rest => physicalFinalClock (expansion.nextClock clock) rest

structure LocalSimulation {Value User : Type} (plan : Epoch.PlanContext)
    (scalar : Epoch.BinaryMeaning Value)
    (meaning : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (context : Context) (clock : Nat) (expansion : Expansion) : Prop where
  forward : ∀ (before after : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (actual : GraphLoopsShape.Env Value),
    Epoch.RegistersValid context plan.views before.observed.registers →
    Epoch.CallersValid plan.objects.length plan.callers before.observed.registers →
    RegistersRelated plan.objects context before.observed.registers actual →
    GraphLoopsMemorySCF.Step plan.objects plan.configs (plan.oldMeaning scalar) true true
      expansion.source.instruction before after →
    ∃ next, FragmentsRun meaning expansion.fragments actual (Epoch.clockWorld plan.objects.length clock before)
      next (Epoch.clockWorld plan.objects.length (expansion.nextClock clock) after) ∧
      RegistersRelated plan.objects (extendContext context expansion.source.results) after.observed.registers next
  backward : ∀ (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (actual next : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User)),
    Epoch.RegistersValid context plan.views before.observed.registers →
    Epoch.CallersValid plan.objects.length plan.callers before.observed.registers →
    RegistersRelated plan.objects context before.observed.registers actual →
    FragmentsRun meaning expansion.fragments actual (Epoch.clockWorld plan.objects.length clock before) next world →
    ∃ after, GraphLoopsMemorySCF.Step plan.objects plan.configs (plan.oldMeaning scalar) true true
      expansion.source.instruction before after ∧
      world = Epoch.clockWorld plan.objects.length (expansion.nextClock clock) after ∧
      RegistersRelated plan.objects (extendContext context expansion.source.results) after.observed.registers next

def LocalSimulations {Value User : Type} (plan : Epoch.PlanContext) (scalar : Epoch.BinaryMeaning Value)
    (meaning : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User))) :
    Context → Nat → List Expansion → Prop
  | _, _, [] => True
  | context, clock, expansion :: rest => LocalSimulation plan scalar meaning context clock expansion ∧
      LocalSimulations plan scalar meaning (extendContext context expansion.source.results)
        (expansion.nextClock clock) rest

theorem fragments_append {Value World : Type} (meaning : GraphLoopsShape.Interpretation Value World)
    (first rest : List Fragment) (actual next : GraphLoopsShape.Env Value) (before after : World) :
    FragmentsRun meaning (first ++ rest) actual before next after ↔
      ∃ middleEnv middle, FragmentsRun meaning first actual before middleEnv middle ∧
        FragmentsRun meaning rest middleEnv middle next after := by
  induction first generalizing actual before with
  | nil =>
    constructor
    · intro run; exact ⟨actual, before, .done _ _, run⟩
    · rintro ⟨middleEnv, middle, first, remaining⟩; cases first; exact remaining
  | cons fragment first induction =>
    constructor
    · intro run
      cases run with
      | more head suffix =>
        obtain ⟨middleEnv, middle, initialRun, remaining⟩ := (induction _ _).mp suffix
        exact ⟨middleEnv, middle, .more head initialRun, remaining⟩
    · rintro ⟨middleEnv, middle, initialRun, remaining⟩
      cases initialRun with
      | more head suffix => exact .more head ((induction _ _).mpr ⟨middleEnv, middle, suffix, remaining⟩)

theorem expansions_forward {Value User : Type} (plan : Epoch.PlanContext)
    (scalar : Epoch.BinaryMeaning Value)
    (meaning : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (expansions : List Expansion) (context : Context) (sourceClock clock : Nat)
    (checked : Epoch.CommandsValid plan context sourceClock (expansions.map Expansion.source))
    (localLaw : LocalSimulations plan scalar meaning context clock expansions)
    (before after : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (actual : GraphLoopsShape.Env Value)
    (registers : Epoch.RegistersValid context plan.views before.observed.registers)
    (callers : Epoch.CallersValid plan.objects.length plan.callers before.observed.registers)
    (related : RegistersRelated plan.objects context before.observed.registers actual)
    (run : GraphLoopsMemorySCF.Execution plan.objects plan.configs (plan.oldMeaning scalar) true true
      ((expansions.map Expansion.source).map Epoch.Command.instruction) before after) :
    ∃ next, FragmentsRun meaning (expansions.flatMap Expansion.fragments) actual
      (Epoch.clockWorld plan.objects.length clock before) next
      (Epoch.clockWorld plan.objects.length (physicalFinalClock clock expansions) after) ∧
      RegistersRelated plan.objects (Epoch.finalContext context (expansions.map Expansion.source))
        after.observed.registers next ∧
      Epoch.RegistersValid (Epoch.finalContext context (expansions.map Expansion.source)) plan.views after.observed.registers ∧
      Epoch.CallersValid plan.objects.length plan.callers after.observed.registers := by
  induction expansions generalizing context sourceClock clock before actual with
  | nil =>
    cases run
    exact ⟨actual, .done _ _, related, registers, callers⟩
  | cons expansion rest induction =>
    cases run with
    | more first suffix =>
      obtain ⟨middleEnv, firstRun, middleRelated⟩ := localLaw.1.forward before _ actual registers callers related first
      have invariant := candidate_preserves plan context sourceClock expansion.source checked.1 scalar
        before _ registers callers first
      obtain ⟨next, remaining, finalRelated, finalRegisters, finalCallers⟩ :=
        induction (extendContext context expansion.source.results) (expansion.source.nextClock sourceClock)
          (expansion.nextClock clock) checked.2 localLaw.2 _ middleEnv invariant.1 invariant.2 middleRelated suffix
      exact ⟨next, (fragments_append meaning expansion.fragments (rest.flatMap Expansion.fragments)
        actual next _ _).mpr ⟨middleEnv, _, firstRun, remaining⟩, finalRelated, finalRegisters, finalCallers⟩

theorem expansions_backward {Value User : Type} (plan : Epoch.PlanContext)
    (scalar : Epoch.BinaryMeaning Value)
    (meaning : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (expansions : List Expansion) (context : Context) (sourceClock clock : Nat)
    (checked : Epoch.CommandsValid plan context sourceClock (expansions.map Expansion.source))
    (localLaw : LocalSimulations plan scalar meaning context clock expansions)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (actual next : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User))
    (registers : Epoch.RegistersValid context plan.views before.observed.registers)
    (callers : Epoch.CallersValid plan.objects.length plan.callers before.observed.registers)
    (related : RegistersRelated plan.objects context before.observed.registers actual)
    (run : FragmentsRun meaning (expansions.flatMap Expansion.fragments) actual
      (Epoch.clockWorld plan.objects.length clock before) next world) :
    ∃ after, GraphLoopsMemorySCF.Execution plan.objects plan.configs (plan.oldMeaning scalar) true true
      ((expansions.map Expansion.source).map Epoch.Command.instruction) before after ∧
      world = Epoch.clockWorld plan.objects.length (physicalFinalClock clock expansions) after ∧
      RegistersRelated plan.objects (Epoch.finalContext context (expansions.map Expansion.source))
        after.observed.registers next ∧
      Epoch.RegistersValid (Epoch.finalContext context (expansions.map Expansion.source)) plan.views after.observed.registers ∧
      Epoch.CallersValid plan.objects.length plan.callers after.observed.registers := by
  induction expansions generalizing context sourceClock clock before actual with
  | nil =>
    cases run
    exact ⟨before, .done _, rfl, related, registers, callers⟩
  | cons expansion rest induction =>
    obtain ⟨middleEnv, middleWorld, firstRun, suffix⟩ := (fragments_append meaning expansion.fragments
      (rest.flatMap Expansion.fragments) actual next _ world).mp run
    obtain ⟨middle, oldFirst, worldEqual, middleRelated⟩ := localLaw.1.backward before actual middleEnv middleWorld
      registers callers related firstRun
    have invariant := candidate_preserves plan context sourceClock expansion.source checked.1 scalar
      before middle registers callers oldFirst
    rw [worldEqual] at suffix
    obtain ⟨after, oldRemaining, finalWorld, finalRelated, finalRegisters, finalCallers⟩ :=
      induction (extendContext context expansion.source.results) (expansion.source.nextClock sourceClock)
        (expansion.nextClock clock) checked.2 localLaw.2 middle middleEnv invariant.1 invariant.2 middleRelated suffix
    exact ⟨after, .more oldFirst oldRemaining, finalWorld, finalRelated, finalRegisters, finalCallers⟩

end Merlin.GraphLoopsMemorySCF.After
