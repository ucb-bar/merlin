/- Separate strict byte-binding relation for graph-chain development receipts.
This checks data equality, not filesystem I/O or cryptographic injectivity.
The trusted host must read/parse/hash these SAME retained byte buffers and load
the selected real proof objects. The preservation result is DIRECT composite
validation; independent stage interpretations are not composed by this module. -/
import Merlin.GraphStateChain
import Merlin.GraphStateReceiptBytes

namespace Merlin.GraphState.Receipt
open StateEffects
open Bytes
set_option autoImplicit false

structure Binding where
  version : Nat
  originalStages : List Encoded
  snapshotStages : List Encoded
  originalContract : Encoded
  snapshotContract : Encoded
  originalSemantics : Encoded
  snapshotSemantics : Encoded
  originalTool : Encoded
  snapshotTool : Encoded
  originalProofSources : List Encoded
  snapshotProofSources : List Encoded
  originalProjection : Encoded
  snapshotProjection : Encoded
  approved : Bool
  nativeQualified : Bool
  numericalGatesDischarged : Bool
  deriving DecidableEq, Repr

def valid (binding : Binding) : Prop :=
  binding.version = 1 ∧
  binding.originalStages.length = 4 ∧
  (∀ bytes ∈ binding.originalStages, bytes.size > 0) ∧
  binding.originalStages = binding.snapshotStages ∧
  binding.originalStages[1]? = binding.originalStages[2]? ∧
  binding.originalContract.size > 0 ∧ binding.originalContract = binding.snapshotContract ∧
  binding.originalSemantics.size > 0 ∧ binding.originalSemantics = binding.snapshotSemantics ∧
  binding.originalTool.size > 0 ∧ binding.originalTool = binding.snapshotTool ∧
  binding.originalProofSources ≠ [] ∧
  (∀ bytes ∈ binding.originalProofSources, bytes.size > 0) ∧
  binding.originalProofSources = binding.snapshotProofSources ∧
  binding.originalProjection.size > 0 ∧ binding.originalProjection = binding.snapshotProjection ∧
  binding.approved = false ∧ binding.nativeQualified = false ∧ binding.numericalGatesDischarged = false

instance (binding : Binding) : Decidable (valid binding) := by unfold valid; infer_instance

/- Boolean byte equality avoids constructing a separate linear equality proof
for every byte while evaluating the certificate. Soundness is proved once below;
the generated claim still uses ordinary kernel reduction, never reflection. -/
def checkBinding (binding : Binding) : Bool :=
  binding.version == 1 && binding.originalStages.length == 4 &&
  binding.originalStages.all (fun bytes => decide (bytes.size > 0)) &&
  binding.originalStages == binding.snapshotStages &&
  binding.originalStages[1]? == binding.originalStages[2]? &&
  decide (binding.originalContract.size > 0) && binding.originalContract == binding.snapshotContract &&
  decide (binding.originalSemantics.size > 0) && binding.originalSemantics == binding.snapshotSemantics &&
  decide (binding.originalTool.size > 0) && binding.originalTool == binding.snapshotTool &&
  binding.originalProofSources != [] && binding.originalProofSources.all (fun bytes => decide (bytes.size > 0)) &&
  binding.originalProofSources == binding.snapshotProofSources &&
  decide (binding.originalProjection.size > 0) && binding.originalProjection == binding.snapshotProjection &&
  !binding.approved && !binding.nativeQualified && !binding.numericalGatesDischarged

theorem checkBinding_sound {binding : Binding} (accepted : checkBinding binding = true) : valid binding := by
  simpa only [checkBinding,valid,Bool.and_eq_true,beq_iff_eq,List.all_eq_true,
    bne_iff_ne,decide_eq_true_eq,Bool.not_eq_true',and_assoc] using accepted

def check (binding : Binding) (footprints : List Footprint) (target : List Nat)
    (exchanges : List GraphState.Exchange) : Bool :=
  checkBinding binding && GraphState.check footprints target exchanges

/-- Accepted actual buffers are exactly bound, with an exact middle join, and
the COMPLETE original/final schedule has the existing conditional all-state
preservation relation. A saved receipt is not kernel/live-custody authority. -/
theorem check_sound {binding : Binding} {footprints : List Footprint} {target : List Nat}
    {exchanges : List GraphState.Exchange} (accepted : check binding footprints target exchanges = true) :
    valid binding ∧ Preserves footprints target := by
  have checks : checkBinding binding = true ∧ GraphState.check footprints target exchanges = true := by
    simpa only [check,Bool.and_eq_true] using accepted
  exact ⟨checkBinding_sound checks.1,GraphState.check_sound checks.2⟩

/-- Packed identity yields actual decoded byte equality; length participates,
so leading zero bytes cannot disappear. Host serialization is still explicit. -/
theorem exact_stage_bytes {binding : Binding} (accepted : checkBinding binding = true) :
    binding.originalStages.map decode = binding.snapshotStages.map decode :=
  congrArg (List.map decode) (checkBinding_sound accepted).2.2.2.1

#print axioms check_sound
#print axioms exact_stage_bytes
end Merlin.GraphState.Receipt
