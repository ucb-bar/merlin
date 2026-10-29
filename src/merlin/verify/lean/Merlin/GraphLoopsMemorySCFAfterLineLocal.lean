/- Actual untiled rank-one initializer local simulation.
Source and target operate on their checked logical/private-slot views; public
registers, caller heap, allocation clock and observer world remain related by
the same complete-entry invariant. This derives the local law from actual
one-loop/store semantics, not a supplied simulation or whole-run premise.
The initializeLine/untiledLine dispatch lifts this law through the complete
entry chain; native allocator fidelity remains a separate premise. -/
import Merlin.GraphLoopsMemorySCFAfterLine
import Merlin.GraphLoopsMemorySCFAfterExecution
set_option autoImplicit false
namespace Merlin.GraphLoopsMemorySCF.After
open GraphLoopsShape

theorem line_initializer_local {Value User : Type} (plan : Epoch.PlanContext) (planValid : plan.Valid)
    (context : Context) (sourceClock clock : Nat) (link : Epoch.LineInitializerLink)
    (checked : (Epoch.Command.initializeLine link).Valid plan context sourceClock)
    (positive : 0 < link.config.tile)
    (slotInside : GraphLoopsReuse.slotOf plan.objects link.config.object < plan.objects.length)
    (scalar : Epoch.BinaryMeaning Value)
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (classify : String → MemoryOpcode)
    (store : classify link.canonical.storeKey = .store) (loop : classify link.canonical.loopKey = .loop) :
    LocalSimulation plan scalar (withMemory base classify) context clock (.untiledLine link) := by
  rcases checked.1 with ⟨facts,inputBound,configMember,mapping,inside,_,_⟩
  have bufferBound := checked.2.1 link.canonical.buffer (by simp [Epoch.Command.inputs])
  have unique : ∀ config ∈ plan.configs, config.key = link.config.key → config = link.config :=
    fun config member same => planValid.1 config member link.config configMember same
  have localLaw (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
      (actual next : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User))
      (registers : Epoch.RegistersValid context plan.views before.observed.registers)
      (related : RegistersRelated plan.objects context before.observed.registers actual) :
      FragmentsRun (withMemory base classify) (Expansion.untiledLine link).fragments actual
        (Epoch.clockWorld plan.objects.length clock before) next world ↔
        ∃ after, GraphLoopsMemorySCF.Step plan.objects plan.configs (plan.oldMeaning scalar) true true
          (.initialize link.config.object link.config.key) before after ∧
          next = actual ∧ world = Epoch.clockWorld plan.objects.length clock after ∧
          RegistersRelated plan.objects context after.observed.registers next := by
    obtain ⟨value,sourceSeed,inputs⟩ := line_inputs_from_related plan.objects link facts context plan.views
      inputBound bufferBound inside before.observed.registers actual registers related
    simpa only [mapping,Epoch.PlanContext.oldMeaning,Expansion.fragments] using
      scoped_line_iff plan.objects plan.configs plan.specifications scalar link facts positive configMember unique
        clock slotInside base classify store loop context before actual related value sourceSeed inputs next world
  constructor
  · intro before after actual registers callers related run
    have publicSame := candidate_initializer_public plan.objects plan.configs (plan.oldMeaning scalar)
      link.config.object link.config.key before after run
    have finalRelated : RegistersRelated plan.objects context after.observed.registers actual := by
      rw [publicSame]; exact related
    exact ⟨actual,(localLaw before actual actual _ registers related).mpr ⟨after,run,rfl,rfl,finalRelated⟩,finalRelated⟩
  · intro before actual next world registers callers related run
    obtain ⟨after,oldRun,_,finalWorld,finalRelated⟩ := (localLaw before actual next world registers related).mp run
    exact ⟨after,oldRun,finalWorld,finalRelated⟩
end Merlin.GraphLoopsMemorySCF.After
