/- UNCOMPILED successor. Complete supported one-call context composition. Kernel
observation preservation is DERIVED from the checked boundary, then lifted using
public-frame suffix simulation. No equality of callee worlds/runs is assumed.
-/
import Merlin.GraphLoopsMemorySCFAfterContextBoundary

set_option autoImplicit false
namespace Merlin.GraphLoopsMemorySCF.After
open GraphLoopsShape

def ContextBoundary.side (data : ContextBoundary) (candidate : Bool) : Module :=
  if candidate then data.expanded else data.plan.module

def CallerContext.Checked (wrapper : CallerContext) (data : ContextBoundary) : Prop :=
  wrapper.Arguments data.kernel ∧ wrapper.Facts data.plan.module ∧ wrapper.Facts data.expanded ∧
    PublicSuffix data.indices data.plan.opcodes data.literals wrapper.suffix

instance (wrapper : CallerContext) (data : ContextBoundary) : Decidable (wrapper.Checked data) := by
  unfold CallerContext.Checked
  infer_instance

theorem CallerContext.side_facts (wrapper : CallerContext) (data : ContextBoundary) (checked : wrapper.Checked data)
    (candidate : Bool) : wrapper.Facts (data.side candidate) := by
  cases candidate
  · exact checked.2.1
  · exact checked.2.2.1

theorem ContextBoundary.kernel_side {Value User : Type} (data : ContextBoundary) (checked : data.Facts)
    (candidate : Bool) (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value)
    (inputs : List (Runtime Value)) (typed : ValuesHold data.kernel.function.parameters inputs)
    (callers : Epoch.CallersValid data.plan.objects.length data.plan.callers
      (assign emptyEnv data.kernel.function.parameters inputs))
    (heap : Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes data.plan.objects.length (data.meaning scalar literal) (data.side candidate)
      data.kernel.function.name inputs heap user observed ↔
    Epoch.graphOutcomes data.plan.objects.length (data.meaning scalar literal) (data.side (!candidate))
      data.kernel.function.name inputs heap user observed := by
  cases candidate
  · exact data.kernel_intermediate checked scalar literal inputs typed callers heap user observed
  · exact (data.kernel_intermediate checked scalar literal inputs typed callers heap user observed).symm

/-- One direction for either choice of source/candidate side. The existential
callee result is matched by the derived theorem; only its public projection
feeds the suffix simulation. Callee private clock/cells remain unconstrained.
-/
theorem CallerContext.transfer {Value User : Type} (wrapper : CallerContext) (data : ContextBoundary)
    (boundary : data.Facts) (checked : wrapper.Checked data) (candidate : Bool)
    (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value)
    (inputs : List (Runtime Value)) (domain : wrapper.Domain data.plan data.kernel inputs)
    (heap : Heap Value) (user : User) (observed : Epoch.Observation Value User)
    (run : Epoch.graphOutcomes data.plan.objects.length (data.meaning scalar literal) (data.side candidate)
      wrapper.function.name inputs heap user observed) :
    Epoch.graphOutcomes data.plan.objects.length (data.meaning scalar literal) (data.side (!candidate))
      wrapper.function.name inputs heap user observed := by
  obtain ⟨outputs, after, whole, observation⟩ := run
  obtain ⟨typed, returned, middle, call, returnTyped, suffix⟩ :=
    (wrapper.complete_iff (data.side candidate) (wrapper.side_facts data checked candidate)
      (data.meaning scalar literal) inputs _ outputs after).mp whole
  have calleeTyped := wrapper.typed_arguments data.kernel checked.1 inputs domain.1
  have kernelCall : complete (data.meaning scalar literal) (data.side candidate) false data.kernel.function.name
      (readArguments false [] (assign emptyEnv wrapper.function.parameters inputs) wrapper.arguments)
      ⟨⟨0, user⟩, heap⟩ returned middle := by simpa only [checked.1.1] using call
  have matched := (data.kernel_side boundary candidate scalar literal _ calleeTyped domain.2.2 heap user
    (Epoch.observeWorld data.plan.objects.length returned middle)).mp ⟨returned, middle, kernelCall, rfl⟩
  obtain ⟨otherReturned, otherMiddle, otherCall, sameObservation⟩ := matched
  obtain ⟨returnsEqual, middleRelated⟩ :=
    (observation_related data.plan.objects.length otherMiddle middle otherReturned returned).mp sameObservation
  subst otherReturned
  have entrySafe : ExternalEnv data.plan.objects.length (assign emptyEnv wrapper.function.parameters inputs) :=
    external_assign _ emptyEnv (external_empty _) wrapper.function.parameters inputs domain.2.1
  have suffixSafe := external_assign _ _ entrySafe wrapper.results returned
    (nonview_values_external _ wrapper.results returned returnTyped checked.1.2.2.2.2.2)
  obtain ⟨otherAfter, otherSuffix, finalRelated⟩ := suffix_transfer data.bits data.plan.objects.length data.indices
    data.plan.opcodes data.literals scalar literal wrapper.suffix checked.2.2.2 _ _ suffixSafe
    middle otherMiddle after middleRelated.symm outputs suffix
  refine ⟨outputs, otherAfter, ?_, ?_⟩
  · apply (wrapper.complete_iff (data.side (!candidate)) (wrapper.side_facts data checked (!candidate))
      (data.meaning scalar literal) inputs _ outputs otherAfter).mpr
    refine ⟨typed, returned, otherMiddle, ?_, returnTyped, otherSuffix⟩
    simpa only [checked.1.1] using otherCall
  · have equal := (observation_related data.plan.objects.length after otherAfter outputs outputs).mpr ⟨rfl, finalRelated⟩
    exact equal.symm.trans observation

theorem CallerContext.intermediate {Value User : Type} (wrapper : CallerContext) (data : ContextBoundary)
    (boundary : data.Facts) (checked : wrapper.Checked data)
    (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value)
    (inputs : List (Runtime Value)) (domain : wrapper.Domain data.plan data.kernel inputs)
    (heap : Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes data.plan.objects.length (data.meaning scalar literal) data.plan.module
      wrapper.function.name inputs heap user observed ↔
    Epoch.graphOutcomes data.plan.objects.length (data.meaning scalar literal) data.expanded
      wrapper.function.name inputs heap user observed :=
  ⟨wrapper.transfer data boundary checked false scalar literal inputs domain heap user observed,
    wrapper.transfer data boundary checked true scalar literal inputs domain heap user observed⟩

/-- Original raw source → normalized whole module → context composition →
actual independently decoded final. Actual final syntax is checked completely.
-/
theorem CallerContext.checked_boundary {Value User : Type} (wrapper : CallerContext) (data : ContextBoundary)
    (boundary : data.Facts) (checked : wrapper.Checked data)
    (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value)
    (inputs : List (Runtime Value)) (domain : wrapper.Domain data.plan data.kernel inputs)
    (heap : Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes data.plan.objects.length (data.meaning scalar literal) data.source
      wrapper.function.name inputs heap user observed ↔
    Epoch.graphOutcomes data.plan.objects.length (data.meaning scalar literal) data.actualFinal
      wrapper.function.name inputs heap user observed := by
  have middle := wrapper.intermediate data boundary checked scalar literal inputs domain heap user observed
  rcases boundary with ⟨sourceValid, sourceAbsent, planValid, normalized, kernelFacts,
    memoryBinding, memoryAccepted, expansion, facts, materialized, support, literalSource, literalFinal⟩
  have first := Epoch.shape_outcomes data.source data.bits sourceValid data.plan.module normalized data.plan.objects.length
    (data.meaning scalar literal) wrapper.function.name inputs heap user observed
  have last : Epoch.graphOutcomes data.plan.objects.length (data.meaning scalar literal) data.expanded
      wrapper.function.name inputs heap user observed ↔
      Epoch.graphOutcomes data.plan.objects.length (data.meaning scalar literal) data.actualFinal
      wrapper.function.name inputs heap user observed := by
    unfold Epoch.graphOutcomes
    apply exists_congr; intro returned
    apply exists_congr; intro world
    apply and_congr_left; intro _
    exact checked_materialization data.bits data.plan data.kernel data.expansions data.actualFinal data.swaps
      data.sourceFrame data.finalFrame materialized (data.meaning scalar literal) wrapper.function.name inputs _ returned world
  exact first.trans (middle.trans last)

end Merlin.GraphLoopsMemorySCF.After
