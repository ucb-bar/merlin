/- Kernel-checked mathematical examples and refusals for complete raw decoding
and actual declaration matching. These independently written AST/Kernel values
are not authoritative compiler captures or native execution tests. -/
import Merlin.GraphRegionsDeclaredCaptureKernel
import Merlin.GraphRegionsDeclaredCaptureParentExamples

namespace Merlin.GraphRegionsDeclared.Capture.Examples
open MerlinVerify Merlin.GraphRegions Merlin.GraphRegionsDeclared

def inputWindow : Window := ⟨0, 0, 2⟩
def outputWindow : Window := ⟨1, 0, 2⟩
def exampleRawCall : RawCall :=
  ⟨"captured_add", [("n", 2), ("a", 0), ("ao", 0), ("dst", 1), ("out", 0), ("width", 1)],
    [], [], none, [], [], none⟩
def exampleSchema : CallSchema :=
  ⟨"captured_add", "captured-add", ["n", "a", "ao", "dst", "out", "width"],
    [32, 32, 32, 32, 32, 32], [(0, ⟨0, 0, 16⟩), (1, ⟨1, 0, 16⟩)],
    [("a", "ao")], "dst", "out", "n", "width", ["little", "little"], [2]⟩
def exampleBound : BoundCall :=
  ⟨"captured_add", "captured-add", exampleRawCall.arguments, [inputWindow], outputWindow,
    ["little", "little"], [2], 8, "synchronous", [], none, []⟩
def exampleKernel : RawKernel :=
  ⟨"captured_add", [⟨"input", [2], "i8", "read"⟩, ⟨"output", [2], "i8", "write"⟩],
    [], [exampleRawCall]⟩
def exampleItems : List CallStage := [⟨literalAddStage, exampleBound⟩]
def exampleAddresses : Nat → Window :=
  fun slot => if slot = 0 then inputWindow else if slot = 3 then outputWindow else ⟨2, 0, 0⟩
def exampleDeclarations : CallStage → DeclarationPair :=
  fun _ => ⟨declaredClosed, declaredClosed, "forward", "forward", none, none⟩

theorem complete_kernel_decoded :
    checkKernel exampleKernel "captured_add" exampleKernel.arguments [exampleSchema]
      (exampleItems.map CallStage.call) = true := by decide

theorem complete_sequence_checked : checkSequence 2 exampleAddresses [0] exampleItems = true := by decide

theorem complete_declarations_checked : checkDeclarations exampleItems exampleDeclarations = true := by
  apply List.all_eq_true.mpr
  intro item member
  have equal : item = (⟨literalAddStage, exampleBound⟩ : CallStage) := by
    simpa only [exampleItems, List.mem_singleton] using member
  subst item
  change checkScalar declaredClosed declaredClosed "forward" "forward" none none literalAdd = true
  apply checkScalar_iff.mpr
  exact ⟨rfl, rfl, by decide⟩

theorem complete_call_sites_checked : checkCallSites completeParent exampleItems = true := by decide

theorem full_source_call_slice :
    exampleItems.map (fun item => item.stage.node) = (completeParent.fullGraph.nodes.drop 2).take 1 := rfl

def exampleMemory : Window → Option (Tensor 2 Nat) :=
  fun window => if window = inputWindow then some (fun index => index.val + 3) else none

theorem initial_memory_refines :
    Refines (fun memory window => memory window) exampleAddresses [0] exampleInitial exampleMemory := by
  intro slot member
  have equal : slot = 0 := by simpa only [List.mem_singleton] using member
  subst slot
  rfl

/-- The two initialized nonzero outputs follow for ANY implementation satisfying
the explicit actual-declaration contract. No implementation is defined equal to
the source and this does not claim native arithmetic fidelity. -/
theorem actual_kernel_initialized_outputs
    (invoke : BoundCall → (Window → Option (Tensor 2 Nat)) → Option (Window → Option (Tensor 2 Nat)))
    (contract : DeclaredInstructionContract (fun memory window => memory window) invoke exampleMeaning
      exampleBound declaredClosed) :
    ∃ final, runActualKernel [exampleSchema] exampleKernel invoke exampleMemory = some final ∧
      final outputWindow = some (tensorEvaluate literalAdd exampleMeaning exampleArguments) ∧
      tensorEvaluate literalAdd exampleMeaning exampleArguments 0 = 9 ∧
      tensorEvaluate literalAdd exampleMeaning exampleArguments 1 = 10 := by
  have contracts : ∀ item ∈ exampleItems,
      DeclaredInstructionContract (fun memory window => memory window) invoke exampleMeaning item.call
        (exampleDeclarations item).declaration := by
    intro item member
    have equal : item = (⟨literalAddStage, exampleBound⟩ : CallStage) := by
      simpa only [exampleItems, List.mem_singleton] using member
    subst item
    exact contract
  obtain ⟨final, ran, refined⟩ := checked_actual_kernel_refines complete_kernel_decoded
    complete_declarations_checked exampleMeaning exampleAddresses [0] complete_sequence_checked contracts
    initial_memory_refines
  have computed := refined 3 (by decide)
  exact ⟨final, ran, computed, initialized_nonzero_elements⟩

def wrongRawOperand : RawCall :=
  { exampleRawCall with arguments := [("n", 2), ("a", 1), ("ao", 0), ("dst", 1), ("out", 0), ("width", 1)] }

theorem wrong_same_width_binding_refused :
    checkKernel { exampleKernel with calls := [wrongRawOperand] } "captured_add" exampleKernel.arguments
      [exampleSchema] [exampleBound] = false := by decide

theorem missing_call_refused :
    checkKernel { exampleKernel with calls := [] } "captured_add" exampleKernel.arguments
      [exampleSchema] [exampleBound] = false := by decide

theorem extra_call_refused :
    checkKernel { exampleKernel with calls := [exampleRawCall, exampleRawCall] } "captured_add"
      exampleKernel.arguments [exampleSchema] [exampleBound] = false := by decide

theorem changed_role_refused :
    checkKernel exampleKernel "captured_add" exampleKernel.arguments
      [{ exampleSchema with role := "different" }] [exampleBound] = false := by decide

theorem asynchronous_call_refused :
    decodeCall exampleSchema { exampleRawCall with awaits := ["completion"] } = none := by decide

theorem changed_abi_refused :
    checkKernel { exampleKernel with arguments := exampleKernel.arguments.reverse } "captured_add"
      exampleKernel.arguments [exampleSchema] [exampleBound] = false := by decide

theorem changed_attributes_refused :
    checkKernel { exampleKernel with attrs := [("unknown", "effect")] } "captured_add"
      exampleKernel.arguments [exampleSchema] [exampleBound] = false := by decide

theorem instruction_failure_retained
    (invoke : BoundCall → Nat → Option Nat) (fault : invoke exampleBound 7 = none) :
    runActualKernel [exampleSchema] exampleKernel invoke 7 = none := by
  rw [checked_kernel_execution complete_kernel_decoded]
  simp only [exampleItems, runCalls, fault, Option.bind_none]

end Merlin.GraphRegionsDeclared.Capture.Examples
