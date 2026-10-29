/- Make prefix projection/elision explicit:
the actual old projectModule graph includes prefix nodes, while the tensor-site
graph uses independently interpreted literal cells. A structural acceptance fact
binds the actual complete AST to the full projection; a proved frame relation
would transport observations. No silent graph/semantics identification occurs.
All prefix Actions below are total and framed by the existing interface; native
allocation/failure truth remains a separate explicit obligation. -/
import Merlin.GraphRegionsDeclaredCaptureFusion

namespace Merlin.GraphRegionsDeclared.Capture
open MerlinVerify Merlin.GraphRegions Merlin.GraphRegionsDeclared

def Prefix.projectNodes : Nat → List Prefix.Instruction → List Node
  | _, [] => []
  | ordinal, instruction :: rest =>
    projectedNode [] ordinal instruction.emit :: Prefix.projectNodes (ordinal + 1) rest

theorem Prefix.projected_writes (leading : List Prefix.Instruction) (ordinal : Nat) :
    writes (Prefix.projectNodes ordinal leading) = leading.map Prefix.Instruction.result := by
  induction leading generalizing ordinal with
  | nil => rfl
  | cons first rest ih =>
    have results : (projectedNode [] ordinal first.emit).results = [first.result] := by
      cases first <;> rfl
    change (projectedNode [] ordinal first.emit).results ++ writes (Prefix.projectNodes (ordinal + 1) rest) = _
    rw [results, ih]
    rfl

def GraphParent.fullGraph (parent : GraphParent) : Graph :=
  ⟨Prefix.projectNodes 0 parent.leading ++ parent.graph.nodes,
    parent.graph.inputs, parent.graph.observations, []⟩

/-- All prefix outputs must be unread by tensor sites and absent from returns.
Captures are read through the independently executed literal-cell interpreter,
not by pretending scalar prefix values are tensor inputs in this graph state. -/
def checkPrefixPrivate (parent : GraphParent) : Bool :=
  (parent.leading.map Prefix.Instruction.result).all
    (fun slot => decide (slot ∉ reads parent.graph.nodes ++ parent.graph.observations))

/-- The projection equality is kernel-checked on the COMPLETE actual AST. It is
an executable structural obligation, not a Python flag or semantic equivalence
assumption. `projectModule` alone does not validate captured generic syntax;
parent.Accepts supplies the complete capture-aware grammar/eligibility check. -/
def FullSourceBinding (parent : GraphParent) (actual retained : Module) : Prop :=
  parent.Accepts actual retained ∧
  projectModule actual (parent.sites.map (·.generic)) = some parent.fullGraph ∧
  checkPrefixPrivate parent = true

theorem executeNodes_append {Value : Type} {typing : Nat → Value → Prop}
    (meaning : ∀ node, TypedAction node Value typing)
    (leading body : List Node) (initial : StateEffects.State Value) :
    executeNodes meaning (leading ++ body) initial =
      executeNodes meaning body (executeNodes meaning leading initial) := by
  simp only [executeNodes, List.map_append, StateEffects.execute, List.foldl_append]

/-- Explicit observation transport for pure total prefix Actions. This does NOT
erase an allocation failure, asynchronous event or observer: those behaviors
cannot be represented by these assumptions without their own justified bridge. -/
theorem prefix_unobserved {Value : Type} {typing : Nat → Value → Prop}
    (meaning : ∀ node, TypedAction node Value typing)
    (leading body : List Node) (observations : List Nat)
    (unobserved : ∀ slot ∈ writes leading, slot ∉ reads body ++ observations)
    (initial : StateEffects.State Value) :
    observations.map (executeNodes meaning (leading ++ body) initial) =
      observations.map (executeNodes meaning body initial) := by
  have same : AgreeOn (reads body ++ observations) (executeNodes meaning leading initial) initial := by
    intro slot observed
    apply execute_frame meaning leading initial slot
    intro written
    exact unobserved slot written observed
  have outputs := suffix_congruence meaning body observations (executeNodes meaning leading initial) initial same
  rw [executeNodes_append]
  exact List.map_congr_left outputs

/-- Complete actual source-parent execution corresponds to the FULL graph
projection, including its prefix nodes. The reduced/full observer relation is
derived from exact write/read/return sets, not assumed as whole-program equality.
Actual compiler-buffer association and qualification remain separate. -/
theorem checked_full_source_correspondence {α : Type} [Inhabited α] {elements : Nat}
    {parent : GraphParent} {actual retained : Module}
    (binding : FullSourceBinding parent actual retained)
    (primitive : Primitive → List α → α) (initialCells : Prefix.Cells α)
    (fallback : ∀ node, TypedAction node (Tensor elements α) (fun _ _ => True))
    (unique : (parent.stages.map ScalarStage.node).Nodup)
    (initial : StateEffects.State (Tensor elements α)) (extent : elements = parent.shape.foldl (· * ·) 1) :
    let meaning := registry primitive fallback parent.stages
    actual = retained ∧
    projectModule actual (parent.sites.map (·.generic)) = some parent.fullGraph ∧
    GraphParentExecution primitive actual initial
      (parent.fullGraph.observations.map (executeNodes meaning parent.fullGraph.nodes initial)) := by
  let meaning := registry primitive fallback parent.stages
  have unobserved : ∀ slot ∈ writes (Prefix.projectNodes 0 parent.leading),
      slot ∉ reads parent.graph.nodes ++ parent.graph.observations := by
    rw [Prefix.projected_writes]
    simpa only [checkPrefixPrivate, List.all_eq_true, decide_eq_true_eq] using binding.2.2
  have observer := prefix_unobserved meaning (Prefix.projectNodes 0 parent.leading) parent.graph.nodes
    parent.graph.observations unobserved initial
  have source := actual_parent_graph_correspondence binding.1 primitive initialCells fallback unique initial extent
  refine ⟨source.1, binding.2.1, ?_⟩
  change GraphParentExecution primitive actual initial
    (parent.graph.observations.map (executeNodes meaning
      (Prefix.projectNodes 0 parent.leading ++ parent.graph.nodes) initial))
  rw [observer]
  exact source.2

/-- Full-source/full-projection acceptance to actual local call realization.
The complete AST, graph prefix/suffix and observers are checked. Per-instruction
fidelity/commit, pack initialization, complete Action effects and ABI publication
remain explicit; actual raw-buffer association and review/qualification are
separate pending obligations. No whole-region output equality is a premise. -/
theorem checked_full_capture_fusion {α M : Type} [Inhabited α] {elements : Nat}
    {parent : GraphParent} {actual retained : Module}
    (binding : FullSourceBinding parent actual retained)
    (primitive : Primitive → List α → α) (initialCells : Prefix.Cells α)
    (fallback : ∀ node, TypedAction node (Tensor elements α) (fun _ _ => True))
    (unique : (parent.stages.map ScalarStage.node).Nodup)
    {after : Graph} {start count : Nat} {selection : List Nat} {plan : Plan} {fused : FusedBody}
    (accepted : checkRealization parent.fullGraph after start count selection plan fused = true)
    (items : List CallStage) (sites : checkCallSites parent items = true)
    (boundary : items.map (fun item => item.stage.node) = (parent.fullGraph.nodes.drop start).take count)
    (addresses : Nat → Window) (known : List Nat)
    (valid : SequenceValid elements addresses known items)
    (read : M → Window → Option (Tensor elements α)) (invoke : BoundCall → M → Option M)
    (contracts : ∀ item ∈ items, InstructionContract read invoke primitive item.call item.stage.program)
    (pack : StateEffects.State (Tensor elements α) → M)
    (initialized : ∀ entry, Refines read addresses known entry (pack entry))
    (outputs : ∀ slot ∈ liveOuts (items.map (fun item => item.stage.node))
      (parent.fullGraph.nodes.drop (start + count)) parent.fullGraph.observations, slot ∈ finalKnown known items)
    (native : StateEffects.Action
      ⟨liveIns ((parent.fullGraph.nodes.drop start).take count),
        liveOuts ((parent.fullGraph.nodes.drop start).take count)
          (parent.fullGraph.nodes.drop (start + count)) parent.fullGraph.observations⟩ (Tensor elements α))
    (publication : ∀ entry final, runCalls invoke items (pack entry) = some final →
      ∀ slot ∈ liveOuts ((parent.fullGraph.nodes.drop start).take count)
        (parent.fullGraph.nodes.drop (start + count)) parent.fullGraph.observations,
        read final (addresses slot) = some (native.step entry slot))
    (initial : StateEffects.State (Tensor elements α)) (extent : elements = parent.shape.foldl (· * ·) 1) :
    let meaning := registry primitive fallback parent.stages
    actual = retained ∧ GraphParentExecution primitive actual initial
      (parent.fullGraph.observations.map (executeNodes meaning (parent.fullGraph.nodes.drop (start + count))
        (native.step (executeNodes meaning (parent.fullGraph.nodes.take start) initial)))) := by
  let meaning := registry primitive fallback parent.stages
  let nativeItems : StateEffects.Action
      ⟨liveIns (items.map (fun item => item.stage.node)),
        liveOuts (items.map (fun item => item.stage.node))
          (parent.fullGraph.nodes.drop (start + count)) parent.fullGraph.observations⟩ (Tensor elements α) := {
    step := native.step
    frame := by simpa only [boundary] using native.frame
    locality := by simpa only [boundary] using native.locality }
  have localLaw := source_derived_local_contract parent primitive fallback unique items sites addresses known
    valid read invoke contracts pack initialized (parent.fullGraph.nodes.drop (start + count))
    parent.fullGraph.observations outputs nativeItems
    (by intro entry final ran slot member; exact publication entry final ran slot (by simpa only [boundary] using member))
  have law : CompositionLaw meaning ((parent.fullGraph.nodes.drop start).take count)
      (parent.fullGraph.nodes.drop (start + count)) parent.fullGraph.observations native := by
    intro entry typed slot member
    have result := localLaw entry typed slot (by simpa only [boundary] using member)
    simpa only [nativeItems, boundary] using result
  have fusion : checkFusion ((parent.fullGraph.nodes.drop start).take count)
      (parent.fullGraph.nodes.drop (start + count)) parent.fullGraph.observations fused = true := by
    simp only [checkRealization, Bool.and_eq_true] at accepted
    exact accepted.2
  simp only [checkFusion, Bool.and_eq_true, decide_eq_true_eq] at fusion
  have observers := checkRealization_native accepted meaning native
    (by
      intro entry typed slot member
      have result := law entry typed slot member
      simpa only [FusedBody.execute, fusion.1.1, fusion.2, publish, if_pos member] using result)
    initial (fun _ => trivial)
  have source := checked_full_source_correspondence binding primitive initialCells fallback unique initial extent
  refine ⟨source.1, ?_⟩
  rw [observers]
  exact source.2.2

end Merlin.GraphRegionsDeclared.Capture
