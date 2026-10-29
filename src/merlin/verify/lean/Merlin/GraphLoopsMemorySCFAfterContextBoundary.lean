/- UNCOMPILED successor. Exact boundary data for the caller-context chain.
Finite syntactic/binding checks are premises; no whole-callee/run equivalence is.
Intermediate transport uses the actual materialization theorem in reverse;
no false expanded→itself normalization check or whole-world equality is assumed.
-/
import Merlin.GraphLoopsMemorySCFAfterContextCall

set_option autoImplicit false
namespace Merlin.GraphLoopsMemorySCF.After
open GraphLoopsShape

structure ContextBoundary where
  source : Module
  bits : Nat
  indices : IndexOpcodes
  plan : Epoch.PlanContext
  kernel : Epoch.Kernel
  memory : GraphLoopsMemorySCF.Model.Projection
  expansions : List Expansion
  actualFinal : Module
  swaps : List (Nat × Nat)
  sourceFrame : List String
  finalFrame : List String
  literals : LiteralOpcodes
  reachable : List String

def ContextBoundary.expanded (data : ContextBoundary) : Module := expandedModule data.plan data.kernel data.expansions

def ContextBoundary.Facts (data : ContextBoundary) : Prop :=
  data.source.Valid data.bits ∧ SourceIndexAbsent data.indices data.source ∧
  data.plan.Valid ∧ data.plan.module = Epoch.normalizedModule data.source ∧
  data.kernel.Facts data.plan ∧ Epoch.MemoryBinding data.plan data.kernel data.memory ∧
  GraphLoopsMemorySCF.Model.check data.memory = true ∧
  ExpansionBinding data.bits data.plan data.kernel data.indices data.expansions ∧
  BoundaryFacts data.plan data.kernel data.expansions ∧
  GraphLoopsShape.After.check ⟨data.bits, data.expanded, data.actualFinal, data.swaps, data.sourceFrame, data.finalFrame⟩ = true ∧
  KernelLiteralSupport data.source data.actualFinal data.kernel data.reachable data.literals ∧
  LiteralModuleFacts data.literals data.source ∧ LiteralModuleFacts data.literals data.actualFinal

instance (data : ContextBoundary) : Decidable data.Facts := by
  unfold ContextBoundary.Facts
  infer_instance

def ContextBoundary.meaning {Value User : Type} (data : ContextBoundary)
    (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value) : Interpretation Value (CallerWorld Value User) :=
  entryMeaning data.bits data.indices data.plan.opcodes scalar data.literals literal

theorem ContextBoundary.kernel_boundary {Value User : Type} (data : ContextBoundary) (checked : data.Facts)
    (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value)
    (inputs : List (Runtime Value)) (typed : ValuesHold data.kernel.function.parameters inputs)
    (callers : Epoch.CallersValid data.plan.objects.length data.plan.callers
      (assign emptyEnv data.kernel.function.parameters inputs))
    (heap : Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes data.plan.objects.length (data.meaning scalar literal) data.source
      data.kernel.function.name inputs heap user observed ↔
    Epoch.graphOutcomes data.plan.objects.length (data.meaning scalar literal) data.actualFinal
      data.kernel.function.name inputs heap user observed := by
  rcases checked with ⟨sourceValid, sourceAbsent, planValid, normalized, kernelFacts,
    memoryBinding, memoryAccepted, expansion, boundary, materialized, support, _, _⟩
  have selected := checked_actual_chain data.source data.bits sourceValid data.indices sourceAbsent
    data.plan planValid normalized data.kernel kernelFacts data.memory memoryBinding memoryAccepted data.expansions
    expansion boundary data.actualFinal data.swaps data.sourceFrame data.finalFrame materialized
    scalar inputs typed callers heap user observed
  have first := literal_outcomes_extension data.plan.objects.length
    (finalMeaning data.bits data.indices data.plan.opcodes scalar) data.literals literal data.source data.reachable
    support.2.1 support.2.2.2.1 data.kernel.function.name support.1 inputs heap user observed
  have last := literal_outcomes_extension data.plan.objects.length
    (finalMeaning data.bits data.indices data.plan.opcodes scalar) data.literals literal data.actualFinal data.reachable
    support.2.2.1 support.2.2.2.2 data.kernel.function.name support.1 inputs heap user observed
  exact first.symm.trans (selected.trans last)

/-- Derive the intermediate kernel relation from the actual checked endpoints,
normalization and actual materialization. Expanded code may still contain cast
aliases: no identity-normalization/self-check is assumed.
-/
theorem ContextBoundary.kernel_intermediate {Value User : Type} (data : ContextBoundary) (checked : data.Facts)
    (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value)
    (inputs : List (Runtime Value)) (typed : ValuesHold data.kernel.function.parameters inputs)
    (callers : Epoch.CallersValid data.plan.objects.length data.plan.callers
      (assign emptyEnv data.kernel.function.parameters inputs))
    (heap : Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes data.plan.objects.length (data.meaning scalar literal) data.plan.module
      data.kernel.function.name inputs heap user observed ↔
    Epoch.graphOutcomes data.plan.objects.length (data.meaning scalar literal) data.expanded
      data.kernel.function.name inputs heap user observed := by
  have raw := data.kernel_boundary checked scalar literal inputs typed callers heap user observed
  rcases checked with ⟨sourceValid, _, _, normalized, _, _, _, _, _, materialized, _⟩
  have first := Epoch.shape_outcomes data.source data.bits sourceValid data.plan.module normalized data.plan.objects.length
    (data.meaning scalar literal) data.kernel.function.name inputs heap user observed
  have last : Epoch.graphOutcomes data.plan.objects.length (data.meaning scalar literal) data.expanded
      data.kernel.function.name inputs heap user observed ↔
      Epoch.graphOutcomes data.plan.objects.length (data.meaning scalar literal) data.actualFinal
      data.kernel.function.name inputs heap user observed := by
    unfold Epoch.graphOutcomes
    apply exists_congr; intro returned
    apply exists_congr; intro world
    apply and_congr_left; intro _
    exact checked_materialization data.bits data.plan data.kernel data.expansions data.actualFinal data.swaps
      data.sourceFrame data.finalFrame materialized (data.meaning scalar literal) data.kernel.function.name inputs _ returned world
  exact first.symm.trans (raw.trans last.symm)

end Merlin.GraphLoopsMemorySCF.After
