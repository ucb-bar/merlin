/- Shape normalization followed by checked reuse and ordered segmentation under
ONE derived operation interpretation. Actual full-tree matching binds each IR
boundary; this theorem supplies the semantic join, not an equality of hashes. -/
import Merlin.GraphLoopsShapeEpochOutcomes
import Merlin.GraphLoopsMemorySCFModel

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

def MemoryBinding (plan : PlanContext) (kernel : Kernel) (memory : GraphLoopsMemorySCF.Model.Projection) : Prop :=
  plan.objects = memory.reuse.objects ∧ plan.configs = GraphLoopsMemorySCF.Model.configs memory ∧
  kernel.instructions = memory.reuse.instructions

instance memoryBindingDecidable (plan : PlanContext) (kernel : Kernel)
    (memory : GraphLoopsMemorySCF.Model.Projection) : Decidable (MemoryBinding plan kernel memory) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

theorem epoch_chain {Value User : Type} (plan : PlanContext) (kernel : Kernel)
    (memory : GraphLoopsMemorySCF.Model.Projection) (binding : MemoryBinding plan kernel memory)
    (accepted : GraphLoopsMemorySCF.Model.check memory = true) (scalar : BinaryMeaning Value)
    (source target : GraphLoopsReuse.Memory Value (PublicState Value User))
    (same : source.observed = target.observed) (observed : Observation Value User) :
    epochOutcomes plan kernel scalar false false source observed ↔
      epochOutcomes plan kernel scalar true true target observed := by
  unfold epochOutcomes
  apply exists_congr
  intro state
  apply and_congr_left
  intro _
  have checked := GraphLoopsMemorySCF.Model.checked_complete accepted (plan.oldMeaning scalar) source target same state
  simpa only [binding.1, binding.2.1, binding.2.2] using checked

/-- The already proved whole-module normalizer supplies the first edge, with
the same concrete operation interpreter later used by the epoch adapter. -/
theorem shape_outcomes {Value User : Type} (source : Module) (bits : Nat) (valid : source.Valid bits)
    (normalized : Module) (same : normalized = normalizedModule source)
    (bound : Nat) (meaning : Interpretation Value (MemoryWorld Value (AllocationPublic User)))
    (name : String) (inputs : List (Runtime Value)) (heap : Heap Value) (user : User)
    (observed : Observation Value User) :
    graphOutcomes bound meaning source name inputs heap user observed ↔
      graphOutcomes bound meaning normalized name inputs heap user observed := by
  unfold graphOutcomes
  apply exists_congr
  intro returned
  apply exists_congr
  intro world
  apply and_congr_left
  intro _
  rw [same]
  exact (source.checked_complete bits valid meaning name inputs _ returned world).trans
    (normalized_complete meaning source name inputs _ world returned)

/-- Complete caller-observation preservation for the actual admitted entry:
typed source graph → normalized typed graph → logical epochs → reused slots
with segmented initializers. Initial private heap contents may be arbitrary.
Allocator/model/primitive fidelity and trusted IR projection remain separate. -/
theorem Kernel.checked_chain {Value User : Type} (source : Module) (bits : Nat) (sourceValid : source.Valid bits)
    (plan : PlanContext) (planValid : plan.Valid) (normalized : plan.module = normalizedModule source)
    (kernel : Kernel) (facts : kernel.Facts plan)
    (memory : GraphLoopsMemorySCF.Model.Projection) (binding : MemoryBinding plan kernel memory)
    (accepted : GraphLoopsMemorySCF.Model.check memory = true) (scalar : BinaryMeaning Value)
    (inputs : List (Runtime Value)) (typed : ValuesHold kernel.function.parameters inputs)
    (callers : CallersValid plan.objects.length plan.callers (assign emptyEnv kernel.function.parameters inputs))
    (heap candidateHeap : Heap Value) (user : User) (observed : Observation Value User) :
    graphOutcomes plan.objects.length (graphMeaning plan.opcodes scalar) source kernel.function.name inputs heap user observed ↔
      epochOutcomes plan kernel scalar true true
        ⟨(entryMemory plan kernel.function inputs heap user).observed, candidateHeap⟩ observed := by
  exact (shape_outcomes source bits sourceValid plan.module normalized plan.objects.length _ _ inputs heap user observed).trans
    ((kernel.observations plan planValid facts scalar inputs typed callers heap user observed).trans
      (epoch_chain plan kernel memory binding accepted scalar
        (entryMemory plan kernel.function inputs heap user)
        ⟨(entryMemory plan kernel.function inputs heap user).observed, candidateHeap⟩ rfl observed))

end Merlin.GraphLoopsShape.Epoch
