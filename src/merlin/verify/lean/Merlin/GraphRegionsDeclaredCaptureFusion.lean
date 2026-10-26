/- Captured-source fusion bridge under the restricted additive interpretation.
Connect the complete captured-source parent to the existing checked fusion root.
All source-site meanings are constructed from their scalar programs; only finer
per-instruction execution/packing/publication and Action effects are premises.
Actual Kernel/ABI/raw-buffer association and independent runtime truth remain
separate obligations; this draft does not inherit the historical qualification. -/
import Merlin.GraphRegionsDeclaredCaptureActualGraph
import Merlin.GraphRegionsDeclaredBridge

namespace Merlin.GraphRegionsDeclared.Capture
open MerlinVerify Merlin.GraphRegions Merlin.GraphRegionsDeclared

def GraphParent.stages (parent : GraphParent) : List ScalarStage :=
  parent.sites.map (fun site => site.generic.stage)

/-- Exact stage membership includes the full scalar program, operand order,
types, role and source ordinal. Matching node labels alone is insufficient. -/
def checkCallSites (parent : GraphParent) (items : List CallStage) : Bool :=
  items.all (fun item => decide (item.stage ∈ parent.stages))

theorem checkCallSites_sound {parent : GraphParent} {items : List CallStage}
    (accepted : checkCallSites parent items = true) :
    ∀ item ∈ items, item.stage ∈ parent.stages := by
  simpa only [checkCallSites, List.all_eq_true, decide_eq_true_eq] using accepted

/-- Derived local output law under the FULL source registry, including every
unchanged prefix/suffix site. No arbitrary meaning for those sites and no
whole-region agreement hypothesis are introduced by this bridge. -/
theorem source_derived_local_contract {α M : Type} [Inhabited α] {elements : Nat}
    (parent : GraphParent) (primitive : Primitive → List α → α)
    (fallback : ∀ node, TypedAction node (Tensor elements α) (fun _ _ => True))
    (unique : (parent.stages.map ScalarStage.node).Nodup)
    (items : List CallStage) (sites : checkCallSites parent items = true)
    (addresses : Nat → Window) (known : List Nat)
    (valid : SequenceValid elements addresses known items)
    (read : M → Window → Option (Tensor elements α)) (invoke : BoundCall → M → Option M)
    (contracts : ∀ item ∈ items, InstructionContract read invoke primitive item.call item.stage.program)
    (pack : StateEffects.State (Tensor elements α) → M)
    (initialized : ∀ entry, Refines read addresses known entry (pack entry))
    (suffix : List Node) (observations : List Nat)
    (outputs : ∀ slot ∈ liveOuts (items.map (fun item => item.stage.node)) suffix observations,
      slot ∈ finalKnown known items)
    (native : StateEffects.Action
      ⟨liveIns (items.map (fun item => item.stage.node)),
        liveOuts (items.map (fun item => item.stage.node)) suffix observations⟩ (Tensor elements α))
    (publication : ∀ entry final, runCalls invoke items (pack entry) = some final →
      ∀ slot ∈ liveOuts (items.map (fun item => item.stage.node)) suffix observations,
        read final (addresses slot) = some (native.step entry slot)) :
    CompositionLaw (registry primitive fallback parent.stages)
      (items.map (fun item => item.stage.node)) suffix observations native := by
  intro entry _ slot member
  obtain ⟨final, ran, refined⟩ := sequence_refines primitive addresses items known valid contracts (initialized entry)
  have result := Option.some.inj ((publication entry final ran slot member).symm.trans
    (refined slot (outputs slot member)))
  rw [result]
  have lookup : ∀ stage ∈ items.map CallStage.stage,
      (registry primitive fallback parent.stages stage.node).action.step = stage.step primitive := by
    intro stage member
    obtain ⟨item, present, equal⟩ := List.mem_map.mp member
    subst stage
    exact registry_steps primitive fallback parent.stages unique item.stage
      (checkCallSites_sound sites item present)
  have logical := executeStages_nodes primitive (registry primitive fallback parent.stages)
    (items.map CallStage.stage) lookup entry
  simpa only [List.map_map, Function.comp_def] using congrArg (fun state => state slot) logical

/-- Acceptance to complete captured-source observations through the existing
checkRealization_native theorem. Source-prefix literal initialization and all
source-site meanings are derived. Instruction fidelity, initialized packing,
publication and total/private/local Action effects remain explicit contracts.

Actual captured-provider receipts and fresh replay are separate qualification
obligations. This is an additive source interpretation, not frozen Core.Sem. -/
theorem checked_capture_fusion {α M : Type} [Inhabited α] {elements : Nat}
    {parent : GraphParent} {actual retained : Module}
    (sourceAccepted : parent.Accepts actual retained)
    (primitive : Primitive → List α → α) (initialCells : Prefix.Cells α)
    (fallback : ∀ node, TypedAction node (Tensor elements α) (fun _ _ => True))
    (unique : (parent.stages.map ScalarStage.node).Nodup)
    {after : Graph} {start count : Nat} {selection : List Nat} {plan : Plan} {fused : FusedBody}
    (accepted : checkRealization parent.graph after start count selection plan fused = true)
    (items : List CallStage) (sites : checkCallSites parent items = true)
    (boundary : items.map (fun item => item.stage.node) = (parent.graph.nodes.drop start).take count)
    (addresses : Nat → Window) (known : List Nat)
    (valid : SequenceValid elements addresses known items)
    (read : M → Window → Option (Tensor elements α)) (invoke : BoundCall → M → Option M)
    (contracts : ∀ item ∈ items, InstructionContract read invoke primitive item.call item.stage.program)
    (pack : StateEffects.State (Tensor elements α) → M)
    (initialized : ∀ entry, Refines read addresses known entry (pack entry))
    (outputs : ∀ slot ∈ liveOuts (items.map (fun item => item.stage.node))
      (parent.graph.nodes.drop (start + count)) parent.graph.observations, slot ∈ finalKnown known items)
    (native : StateEffects.Action
      ⟨liveIns ((parent.graph.nodes.drop start).take count),
        liveOuts ((parent.graph.nodes.drop start).take count)
          (parent.graph.nodes.drop (start + count)) parent.graph.observations⟩ (Tensor elements α))
    (publication : ∀ entry final, runCalls invoke items (pack entry) = some final →
      ∀ slot ∈ liveOuts ((parent.graph.nodes.drop start).take count)
        (parent.graph.nodes.drop (start + count)) parent.graph.observations,
        read final (addresses slot) = some (native.step entry slot))
    (initial : StateEffects.State (Tensor elements α)) (extent : elements = parent.shape.foldl (· * ·) 1) :
    let meaning := registry primitive fallback parent.stages
    actual = retained ∧ GraphParentExecution primitive actual initial
      (parent.graph.observations.map (executeNodes meaning (parent.graph.nodes.drop (start + count))
        (native.step (executeNodes meaning (parent.graph.nodes.take start) initial)))) := by
  let meaning := registry primitive fallback parent.stages
  let nativeItems : StateEffects.Action
      ⟨liveIns (items.map (fun item => item.stage.node)),
        liveOuts (items.map (fun item => item.stage.node))
          (parent.graph.nodes.drop (start + count)) parent.graph.observations⟩ (Tensor elements α) := {
    step := native.step
    frame := by simpa only [boundary] using native.frame
    locality := by simpa only [boundary] using native.locality }
  have localLaw := source_derived_local_contract parent primitive fallback unique items sites addresses known
    valid read invoke contracts pack initialized (parent.graph.nodes.drop (start + count))
    parent.graph.observations outputs nativeItems
    (by intro entry final ran slot member; exact publication entry final ran slot (by simpa only [boundary] using member))
  have law : CompositionLaw meaning ((parent.graph.nodes.drop start).take count)
      (parent.graph.nodes.drop (start + count)) parent.graph.observations native := by
    intro entry typed slot member
    have result := localLaw entry typed slot (by simpa only [boundary] using member)
    simpa only [nativeItems, boundary] using result
  have fusion : checkFusion ((parent.graph.nodes.drop start).take count)
      (parent.graph.nodes.drop (start + count)) parent.graph.observations fused = true := by
    simp only [checkRealization, Bool.and_eq_true] at accepted
    exact accepted.2
  simp only [checkFusion, Bool.and_eq_true, decide_eq_true_eq] at fusion
  have observers := checkRealization_native accepted meaning native
    (by
      intro entry typed slot member
      have result := law entry typed slot member
      simpa only [FusedBody.execute, fusion.1.1, fusion.2, publish, if_pos member] using result)
    initial (fun _ => trivial)
  have source := actual_parent_graph_correspondence sourceAccepted primitive initialCells fallback unique initial extent
  refine ⟨source.1, ?_⟩
  rw [observers]
  exact source.2

end Merlin.GraphRegionsDeclared.Capture
