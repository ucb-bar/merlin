/- A direct declared-call bridge, with independent physical memory and logical
SSA identities. Per-instruction read/compute/commit/no-fault/frame contracts are
explicit premises; qualification identifiers and hashes cannot supply them. -/
import Merlin.GraphRegionsDeclaredNodes

namespace Merlin.GraphRegionsDeclared
open Merlin.GraphRegions

structure Window where
  backing : Nat
  start : Nat
  bytes : Nat
  deriving DecidableEq, Repr

def Window.Disjoint (first second : Window) : Prop :=
  first.backing ≠ second.backing ∨ first.start + first.bytes ≤ second.start ∨
    second.start + second.bytes ≤ first.start

instance (first second : Window) : Decidable (first.Disjoint second) := by
  unfold Window.Disjoint; infer_instance

structure BoundCall where
  instruction : String
  role : String
  arguments : List (String × Nat)
  reads : List Window
  output : Window
  byteOrders : List String
  shape : List Nat
  elementBits : Nat
  completion : String
  waits : List String
  token : Option String
  extraEffects : List String
  deriving DecidableEq, Repr

def callValid (elements : Nat) (stage : ScalarStage) (addresses : Nat → Window) (call : BoundCall) : Prop :=
  call.role = stage.role ∧ call.reads = stage.inputs.map addresses ∧ call.output = addresses stage.output ∧
  stage.program.inputs.length = stage.inputs.length ∧ stage.program.output = call.elementBits ∧
  stage.program.inputs = List.replicate stage.inputs.length call.elementBits ∧
  call.shape.foldl (· * ·) 1 = elements ∧ 0 < elements ∧ 0 < call.elementBits ∧ call.elementBits % 8 = 0 ∧
  call.output.bytes = elements * (call.elementBits / 8) ∧
  (∀ window ∈ call.reads, window.bytes = call.output.bytes) ∧
  call.byteOrders = List.replicate (call.reads.length + 1) "little" ∧
  call.completion = "synchronous" ∧ call.waits = [] ∧ call.token = none ∧ call.extraEffects = []

instance (elements : Nat) (stage : ScalarStage) (addresses : Nat → Window) (call : BoundCall) :
    Decidable (callValid elements stage addresses call) := by unfold callValid; infer_instance

/-- Actual instruction/argument identity is checked against its separately
retained selected declaration, not inferred from an asserted role string. -/
def checkCall (elements : Nat) (stage : ScalarStage) (addresses : Nat → Window)
    (actual selected : BoundCall) : Bool := decide (actual = selected ∧ callValid elements stage addresses actual)

theorem checkCall_sound {elements : Nat} {stage : ScalarStage} {addresses : Nat → Window}
    {actual selected : BoundCall} (accepted : checkCall elements stage addresses actual selected = true) :
    actual = selected ∧ callValid elements stage addresses actual := by
  exact of_decide_eq_true accepted

def Refines {α M : Type} {elements : Nat} (read : M → Window → Option (Tensor elements α))
    (addresses : Nat → Window) (known : List Nat)
    (logical : StateEffects.State (Tensor elements α)) (memory : M) : Prop :=
  ∀ slot ∈ known, read memory (addresses slot) = some (logical slot)

/-- The machine is opaque and may fault. This is finer than a region agreement:
one instruction must capture its ordered initialized inputs, implement the
matched scalar declaration elementwise, commit synchronously without fault,
and preserve every disjoint read window. Aliasing/read-before-write/no-fault
fidelity are NOT established by identity checks or native samples. -/
structure InstructionContract {α M : Type} [Inhabited α] {elements : Nat}
    (read : M → Window → Option (Tensor elements α)) (invoke : BoundCall → M → Option M)
    (primitive : Primitive → List α → α) (call : BoundCall) (program : Program) : Prop where
  capture_compute_commit : ∀ memory arguments,
    call.reads.map (read memory) = arguments.map some →
    ∃ next, invoke call memory = some next ∧
      read next call.output = some (tensorEvaluate program primitive arguments) ∧
      ∀ window, window.Disjoint call.output → read next window = read memory window

/-- One actual call extends initialized SSA refinement. Every retained live input
survives by the physical frame; the new result is supplied by the scalar contract.
No desired whole-region equality is assumed. -/
theorem instruction_refines {α M : Type} [Inhabited α] {elements : Nat}
    {read : M → Window → Option (Tensor elements α)} {invoke : BoundCall → M → Option M}
    {primitive : Primitive → List α → α} {stage : ScalarStage} {call : BoundCall}
    {addresses : Nat → Window} {known : List Nat}
    (valid : callValid elements stage addresses call)
    (inputs : ∀ slot ∈ stage.inputs, slot ∈ known)
    (nonalias : ∀ slot ∈ known, slot ≠ stage.output → (addresses slot).Disjoint (addresses stage.output))
    (contract : InstructionContract read invoke primitive call stage.program)
    {logical : StateEffects.State (Tensor elements α)} {memory : M}
    (refined : Refines read addresses known logical memory) :
    ∃ next, invoke call memory = some next ∧
      Refines read addresses (known ++ [stage.output]) (stage.step primitive logical) next := by
  have reads : call.reads.map (read memory) = (stage.inputs.map logical).map some := by
    rw [valid.2.1]
    simp only [List.map_map]
    exact List.map_congr_left (fun slot member => refined slot (inputs slot member))
  obtain ⟨next, runs, computed, frame⟩ := contract.capture_compute_commit memory _ reads
  refine ⟨next, runs, ?_⟩
  intro slot member
  by_cases output : slot = stage.output
  · subst slot
    simpa only [valid.2.2.1, ScalarStage.step, bind, ↓reduceIte] using computed
  · have old : slot ∈ known := (List.mem_append.mp member).resolve_right (by simpa using output)
    rw [frame _ (by rw [valid.2.2.1]; exact nonalias slot old output), refined slot old]
    simp [ScalarStage.step, bind, output]

structure CallStage where
  stage : ScalarStage
  call : BoundCall
  deriving DecidableEq, Repr

def finalKnown : List Nat → List CallStage → List Nat
  | known, [] => known
  | known, item :: rest => finalKnown (known ++ [item.stage.output]) rest

def runCalls {M : Type} (invoke : BoundCall → M → Option M) : List CallStage → M → Option M
  | [], memory => some memory
  | item :: rest, memory => (invoke item.call memory).bind (runCalls invoke rest)

def SequenceValid (elements : Nat) (addresses : Nat → Window) : List Nat → List CallStage → Prop
  | _, [] => True
  | known, item :: rest =>
    callValid elements item.stage addresses item.call ∧
    (∀ slot ∈ item.stage.inputs, slot ∈ known) ∧ item.stage.output ∉ known ∧
    (∀ slot ∈ known, (addresses slot).Disjoint (addresses item.stage.output)) ∧
    SequenceValid elements addresses (known ++ [item.stage.output]) rest

/-- Induction over actual ordered instructions supplies initialized-success and
logical refinement for the complete call sequence, including BOTH consumers.
Failure exclusion comes only from each explicit instruction contract. -/
theorem sequence_refines {α M : Type} [Inhabited α] {elements : Nat}
    {read : M → Window → Option (Tensor elements α)} {invoke : BoundCall → M → Option M}
    (primitive : Primitive → List α → α) (addresses : Nat → Window) (items : List CallStage)
    (known : List Nat) (valid : SequenceValid elements addresses known items)
    (contracts : ∀ item ∈ items, InstructionContract read invoke primitive item.call item.stage.program)
    {logical : StateEffects.State (Tensor elements α)} {memory : M}
    (refined : Refines read addresses known logical memory) :
    ∃ final, runCalls invoke items memory = some final ∧
      Refines read addresses (finalKnown known items)
        (executeStages primitive (items.map CallStage.stage) logical) final := by
  induction items generalizing known logical memory with
  | nil => exact ⟨memory, rfl, refined⟩
  | cons item rest ih =>
    obtain ⟨next, step, nextRefined⟩ := instruction_refines valid.1 valid.2.1
      (fun slot member _ => valid.2.2.2.1 slot member) (contracts item (by simp)) refined
    obtain ⟨final, tail, result⟩ := ih (known ++ [item.stage.output]) valid.2.2.2.2
      (fun item member => contracts item (by simp [member])) nextRefined
    refine ⟨final, ?_, result⟩
    simp [runCalls, step, tail]

end Merlin.GraphRegionsDeclared
