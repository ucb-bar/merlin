/- Complete raw Kernel decoding and execution, connected to the captured-source
parent theorem. The decoder covers the entire submitted ordered call list.
Native declaration fidelity, physical effects, initialized packing and ABI
publication remain explicit premises; no machine implementation is substituted. -/
import Merlin.GraphRegionsDeclaredCaptureDeclaration
import Merlin.GraphRegionsDeclaredRaw

namespace Merlin.GraphRegionsDeclared.Capture
open MerlinVerify Merlin.GraphRegions Merlin.GraphRegionsDeclared

/-- Execution is partial: decoding refusal and an instruction fault both remain
`none`. Unsupported Kernel attributes are refused before executing any call. -/
def runDecodedCalls {M : Type} (invoke : BoundCall → M → Option M) : List BoundCall → M → Option M
  | [], memory => some memory
  | call :: rest, memory => (invoke call memory).bind (runDecodedCalls invoke rest)

def runActualKernel {M : Type} (schemas : List CallSchema) (actual : RawKernel)
    (invoke : BoundCall → M → Option M) (memory : M) : Option M :=
  if actual.attrs = [] then
    (decodeCalls schemas actual.calls).bind (fun calls => runDecodedCalls invoke calls memory)
  else none

theorem runDecodedCalls_stages {M : Type} (invoke : BoundCall → M → Option M)
    (items : List CallStage) (memory : M) :
    runDecodedCalls invoke (items.map CallStage.call) memory = runCalls invoke items memory := by
  induction items generalizing memory with
  | nil => rfl
  | cons item rest ih =>
    simp only [List.map_cons, runDecodedCalls, runCalls]
    cases invoke item.call memory with
    | none => rfl
    | some next => exact ih next

theorem checked_kernel_execution {M : Type} {actual : RawKernel} {name : String}
    {arguments : List TensorArgument} {schemas : List CallSchema} {items : List CallStage}
    (accepted : checkKernel actual name arguments schemas (items.map CallStage.call) = true)
    (invoke : BoundCall → M → Option M) (memory : M) :
    runActualKernel schemas actual invoke memory = runCalls invoke items memory := by
  have checked := checkKernel_sound accepted
  simp only [runActualKernel, checked.2.2.1, ↓reduceIte, checked.2.2.2, Option.bind_some]
  exact runDecodedCalls_stages invoke items memory

theorem unsupported_kernel_attrs_refused {M : Type} {actual : RawKernel}
    (unsupported : actual.attrs ≠ []) (schemas : List CallSchema)
    (invoke : BoundCall → M → Option M) (memory : M) :
    runActualKernel schemas actual invoke memory = none := by
  simp only [runActualKernel, if_neg unsupported]

theorem undecodable_kernel_refused {M : Type} {actual : RawKernel} {schemas : List CallSchema}
    (undecodable : decodeCalls schemas actual.calls = none)
    (invoke : BoundCall → M → Option M) (memory : M) :
    runActualKernel schemas actual invoke memory = none := by
  simp only [runActualKernel, undecodable, Option.bind_none]
  split <;> rfl

theorem decoded_fault_propagates {M : Type} (invoke : BoundCall → M → Option M)
    {call : BoundCall} (rest : List BoundCall) {memory : M}
    (fault : invoke call memory = none) :
    runDecodedCalls invoke (call :: rest) memory = none := by
  simp only [runDecodedCalls, fault, Option.bind_none]

/-- A poisoned/uninitialized live input is outside the initialized refinement
domain. Merely matching its type/window cannot supply an initialized value. -/
theorem uninitialized_input_excludes_refinement {α M : Type} {elements : Nat}
    {read : M → Window → Option (Tensor elements α)} {addresses : Nat → Window}
    {known : List Nat} {slot : Nat} {logical : StateEffects.State (Tensor elements α)} {memory : M}
    (live : slot ∈ known) (uninitialized : read memory (addresses slot) = none) :
    ¬ Refines read addresses known logical memory := by
  intro refined
  have impossible := (refined slot live).symm.trans uninitialized
  cases impossible

/-- A fault on checked initialized arguments contradicts the declared commit
premise. The observer theorem cannot be applied to erase this failing run. -/
theorem fault_excludes_declared_contract {α M : Type} [Inhabited α] {elements : Nat}
    {read : M → Window → Option (Tensor elements α)} {invoke : BoundCall → M → Option M}
    {primitive : Primitive → List α → α} {call : BoundCall} {pair : DeclarationPair} {program : Program}
    (accepted : pair.check program = true) {memory : M} {arguments : List (Tensor elements α)}
    (initialized : call.reads.map (read memory) = arguments.map some)
    (fault : invoke call memory = none) :
    ¬ DeclaredInstructionContract read invoke primitive call pair.declaration := by
  intro contract
  obtain ⟨next, ran, _⟩ := (instruction_contract_from_declaration accepted contract).capture_compute_commit
    memory arguments initialized
  have impossible := ran.symm.trans fault
  cases impossible

/-- Actual complete raw Kernel success/refinement follows from its exact
decoder, structural sequence checks, actual declaration checks and the finer
declaration execution contracts. Failure exclusion is not obtained from IDs. -/
theorem checked_actual_kernel_refines {α M : Type} [Inhabited α] {elements : Nat}
    {actual : RawKernel} {name : String} {arguments : List TensorArgument}
    {schemas : List CallSchema} {items : List CallStage}
    (kernel : checkKernel actual name arguments schemas (items.map CallStage.call) = true)
    {declarations : CallStage → DeclarationPair}
    (declared : checkDeclarations items declarations = true)
    (primitive : Primitive → List α → α) (addresses : Nat → Window) (known : List Nat)
    (sequence : checkSequence elements addresses known items = true)
    {read : M → Window → Option (Tensor elements α)} {invoke : BoundCall → M → Option M}
    (contracts : ∀ item ∈ items,
      DeclaredInstructionContract read invoke primitive item.call (declarations item).declaration)
    {logical : StateEffects.State (Tensor elements α)} {memory : M}
    (initialized : Refines read addresses known logical memory) :
    ∃ final, runActualKernel schemas actual invoke memory = some final ∧
      Refines read addresses (finalKnown known items)
        (executeStages primitive (items.map CallStage.stage) logical) final := by
  obtain ⟨final, ran, refined⟩ := sequence_refines primitive addresses items known
    (checkSequence_sound sequence) (checked_declaration_contracts declared contracts) initialized
  exact ⟨final, (checked_kernel_execution kernel invoke memory).trans ran, refined⟩

/-- Complete captured-source observations plus actual raw Kernel initialized
success. All scalar declaration/call facts are checked structurally. Remaining
instruction fidelity/completion/frame and ABI publication are explicit. This is
an additive source interpretation, not an identification with frozen Core.Sem. -/
theorem checked_full_actual_kernel_fusion {α M : Type} [Inhabited α] {elements : Nat}
    {parent : GraphParent} {actual retained : Module}
    (binding : FullSourceBinding parent actual retained)
    (primitive : Primitive → List α → α) (initialCells : Prefix.Cells α)
    (fallback : ∀ node, TypedAction node (Tensor elements α) (fun _ _ => True))
    (unique : (parent.stages.map ScalarStage.node).Nodup)
    {after : Graph} {start count : Nat} {selection : List Nat} {plan : Plan} {fused : FusedBody}
    (accepted : checkRealization parent.fullGraph after start count selection plan fused = true)
    (items : List CallStage) (sites : checkCallSites parent items = true)
    (boundary : items.map (fun item => item.stage.node) = (parent.fullGraph.nodes.drop start).take count)
    {raw : RawKernel} {name : String} {arguments : List TensorArgument} {schemas : List CallSchema}
    (kernel : checkKernel raw name arguments schemas (items.map CallStage.call) = true)
    {declarations : CallStage → DeclarationPair}
    (declared : checkDeclarations items declarations = true)
    (addresses : Nat → Window) (known : List Nat)
    (sequence : checkSequence elements addresses known items = true)
    (read : M → Window → Option (Tensor elements α)) (invoke : BoundCall → M → Option M)
    (contracts : ∀ item ∈ items,
      DeclaredInstructionContract read invoke primitive item.call (declarations item).declaration)
    (pack : StateEffects.State (Tensor elements α) → M)
    (initialized : ∀ entry, Refines read addresses known entry (pack entry))
    (outputs : ∀ slot ∈ liveOuts (items.map (fun item => item.stage.node))
      (parent.fullGraph.nodes.drop (start + count)) parent.fullGraph.observations, slot ∈ finalKnown known items)
    (native : StateEffects.Action
      ⟨liveIns ((parent.fullGraph.nodes.drop start).take count),
        liveOuts ((parent.fullGraph.nodes.drop start).take count)
          (parent.fullGraph.nodes.drop (start + count)) parent.fullGraph.observations⟩ (Tensor elements α))
    (publication : ∀ entry final, runActualKernel schemas raw invoke (pack entry) = some final →
      ∀ slot ∈ liveOuts ((parent.fullGraph.nodes.drop start).take count)
        (parent.fullGraph.nodes.drop (start + count)) parent.fullGraph.observations,
        read final (addresses slot) = some (native.step entry slot))
    (initial : StateEffects.State (Tensor elements α)) (extent : elements = parent.shape.foldl (· * ·) 1) :
    let meaning := registry primitive fallback parent.stages
    (∀ entry, ∃ final, runActualKernel schemas raw invoke (pack entry) = some final ∧
      Refines read addresses (finalKnown known items)
        (executeStages primitive (items.map CallStage.stage) entry) final) ∧
    actual = retained ∧ GraphParentExecution primitive actual initial
      (parent.fullGraph.observations.map (executeNodes meaning (parent.fullGraph.nodes.drop (start + count))
        (native.step (executeNodes meaning (parent.fullGraph.nodes.take start) initial)))) := by
  refine ⟨?_, ?_⟩
  · intro entry
    exact checked_actual_kernel_refines kernel declared primitive addresses known sequence contracts (initialized entry)
  · exact checked_full_capture_fusion binding primitive initialCells fallback unique accepted items sites boundary
      addresses known (checkSequence_sound sequence) read invoke (checked_declaration_contracts declared contracts)
      pack initialized outputs native
      (by
        intro entry final ran
        apply publication entry final
        exact (checked_kernel_execution kernel invoke (pack entry)).trans ran)
      initial extent

end Merlin.GraphRegionsDeclared.Capture
