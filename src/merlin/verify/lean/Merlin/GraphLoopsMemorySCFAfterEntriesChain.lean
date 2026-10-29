/- UNCOMPILED successor. All listed defined entries, with exact finite coverage
of both boundary rosters. Unchanged entries require a call-closed identical
submodule; wrappers reaching the changed kernel do not pass that condition.
-/
import Merlin.GraphLoopsMemorySCFAfterEntriesRoster

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

def KernelLiteralSupport (source actualFinal : Module) (kernel : Epoch.Kernel)
    (reachable : List String) (literals : LiteralOpcodes) : Prop :=
  kernel.function.name ∈ reachable ∧ CallClosed source reachable ∧ CallClosed actualFinal reachable ∧
    ClosedLiteralAbsent source reachable literals ∧ ClosedLiteralAbsent actualFinal reachable literals

instance (source actualFinal : Module) (kernel : Epoch.Kernel) (reachable : List String) (literals : LiteralOpcodes) :
    Decidable (KernelLiteralSupport source actualFinal kernel reachable literals) := by
  unfold KernelLiteralSupport
  infer_instance

theorem unchanged_entry_outcomes {Value User : Type} (source : Module) (bits : Nat) (sourceValid : source.Valid bits)
    (plan : Epoch.PlanContext) (normalized : plan.module = Epoch.normalizedModule source)
    (kernel : Epoch.Kernel) (expansions : List Expansion) (reachable : List String)
    (agreement : ClosedAgreement plan.module (expandedModule plan kernel expansions) reachable)
    (actualFinal : Module) (swaps : List (Nat × Nat)) (sourceFrame finalFrame : List String)
    (materialized : GraphLoopsShape.After.check
      ⟨bits, expandedModule plan kernel expansions, actualFinal, swaps, sourceFrame, finalFrame⟩ = true)
    (meaning : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (name : String) (member : name ∈ reachable) (inputs : List (Runtime Value)) (heap : Heap Value)
    (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes plan.objects.length meaning source name inputs heap user observed ↔
      Epoch.graphOutcomes plan.objects.length meaning actualFinal name inputs heap user observed := by
  have first := Epoch.shape_outcomes source bits sourceValid plan.module normalized plan.objects.length meaning
    name inputs heap user observed
  have rest : Epoch.graphOutcomes plan.objects.length meaning plan.module name inputs heap user observed ↔
      Epoch.graphOutcomes plan.objects.length meaning actualFinal name inputs heap user observed := by
    unfold Epoch.graphOutcomes
    apply exists_congr; intro returned
    apply exists_congr; intro world
    apply and_congr_left; intro _
    exact (closed_complete_agreement meaning plan.module (expandedModule plan kernel expansions) reachable agreement
      name member inputs _ returned world).trans
      (checked_materialization bits plan kernel expansions actualFinal swaps sourceFrame finalFrame materialized
        meaning name inputs _ returned world)
  exact first.trans rest

/-- No per-entry semantic equality is supplied. The changed kernel uses the
derived checked chain; each other entry uses finite call-closed body facts and
the same literal/scalar interpretation. Full failure preservation is not asserted.
-/
theorem checked_entry_roster {Value User : Type} (source : Module) (bits : Nat) (sourceValid : source.Valid bits)
    (indices : IndexOpcodes) (sourceAbsent : SourceIndexAbsent indices source)
    (plan : Epoch.PlanContext) (planValid : plan.Valid) (normalized : plan.module = Epoch.normalizedModule source)
    (kernel : Epoch.Kernel) (facts : kernel.Facts plan)
    (memory : GraphLoopsMemorySCF.Model.Projection) (memoryBinding : Epoch.MemoryBinding plan kernel memory)
    (memoryAccepted : GraphLoopsMemorySCF.Model.check memory = true)
    (expansions : List Expansion) (binding : ExpansionBinding bits plan kernel indices expansions)
    (boundary : BoundaryFacts plan kernel expansions)
    (actualFinal : Module) (swaps : List (Nat × Nat)) (sourceFrame finalFrame : List String)
    (materialized : GraphLoopsShape.After.check
      ⟨bits, expandedModule plan kernel expansions, actualFinal, swaps, sourceFrame, finalFrame⟩ = true)
    (sourceHeaders finalHeaders : List EntryHeader) (rows : List EntryRow)
    (roster : EntryRosterFacts source actualFinal sourceHeaders finalHeaders plan kernel expansions rows)
    (literals : LiteralOpcodes) (kernelReachable : List String)
    (literalSource : LiteralModuleFacts literals source) (literalFinal : LiteralModuleFacts literals actualFinal)
    (support : KernelLiteralSupport source actualFinal kernel kernelReachable literals)
    (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value)
    (row : EntryRow) (member : row ∈ rows) (inputs : List (Runtime Value))
    (domain : EntryDomain source plan kernel row inputs)
    (heap : Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes plan.objects.length (entryMeaning bits indices plan.opcodes scalar literals literal)
      source row.name inputs heap user observed ↔
    Epoch.graphOutcomes plan.objects.length (entryMeaning bits indices plan.opcodes scalar literals literal)
      actualFinal row.name inputs heap user observed := by
  have entryFacts := roster.2.2.2.2.2 row member
  cases route : row.route with
  | kernel =>
    have name : row.name = kernel.function.name := by simpa only [EntryRow.Facts, route] using entryFacts
    have inputsValid : ValuesHold kernel.function.parameters inputs ∧
        Epoch.CallersValid plan.objects.length plan.callers (GraphLoopsShape.assign emptyEnv kernel.function.parameters inputs) := by
      simpa only [EntryDomain, route] using domain.2
    rw [name]
    have beforeExtension := literal_outcomes_extension plan.objects.length (finalMeaning bits indices plan.opcodes scalar)
      literals literal source kernelReachable support.2.1 support.2.2.2.1 kernel.function.name support.1 inputs heap user observed
    have afterExtension := literal_outcomes_extension plan.objects.length (finalMeaning bits indices plan.opcodes scalar)
      literals literal actualFinal kernelReachable support.2.2.1 support.2.2.2.2 kernel.function.name support.1 inputs heap user observed
    have changed := checked_actual_chain source bits sourceValid indices sourceAbsent plan planValid normalized kernel facts
      memory memoryBinding memoryAccepted expansions binding boundary actualFinal swaps sourceFrame finalFrame materialized
      scalar inputs inputsValid.1 inputsValid.2 heap user observed
    exact beforeExtension.symm.trans (changed.trans afterExtension)
  | unchanged reachable =>
    have localAgreement : row.name ∈ reachable ∧ ClosedAgreement plan.module (expandedModule plan kernel expansions) reachable := by
      simpa only [EntryRow.Facts, route] using entryFacts
    exact unchanged_entry_outcomes source bits sourceValid plan normalized kernel expansions reachable localAgreement.2 actualFinal
      swaps sourceFrame finalFrame materialized (entryMeaning bits indices plan.opcodes scalar literals literal)
      row.name localAgreement.1 inputs heap user observed

end Merlin.GraphLoopsMemorySCF.After
