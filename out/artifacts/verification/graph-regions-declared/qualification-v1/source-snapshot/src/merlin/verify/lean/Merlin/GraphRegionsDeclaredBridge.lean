/- Existing-root extension: derive the old LOCAL implementation law from actual
ordered instruction contracts and a physical/logical ABI refinement. This is not
an unconditional machine-code theorem or a bridge to the unqualified Core.Sem
profile. Surrounding observers use unchanged GraphRegionsFusion roots. -/
import Merlin.GraphRegionsDeclaredMemory

namespace Merlin.GraphRegionsDeclared
open Merlin.GraphRegions

/-- Local output agreement is a CONCLUSION. Packing only initializes live-ins;
publication only relates actual machine result windows to the supplied action.
The supplied Action's frame/locality are separate physical effect obligations. -/
theorem derived_local_contract {α M : Type} [Inhabited α] {elements : Nat}
    (primitive : Primitive → List α → α)
    (fallback : ∀ node, TypedAction node (Tensor elements α) (fun _ _ => True))
    (items : List CallStage) (addresses : Nat → Window) (known : List Nat)
    (valid : SequenceValid elements addresses known items)
    (unique : ((items.map CallStage.stage).map ScalarStage.node).Nodup)
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
    CompositionLaw (registry primitive fallback (items.map CallStage.stage))
      (items.map (fun item => item.stage.node)) suffix observations native := by
  intro entry _ slot member
  obtain ⟨final, ran, refined⟩ := sequence_refines primitive addresses items known valid contracts (initialized entry)
  have actual := publication entry final ran slot member
  have expected := refined slot (outputs slot member)
  have equal := Option.some.inj (actual.symm.trans expected)
  rw [equal]
  have logical := executeStages_nodes primitive (registry primitive fallback (items.map CallStage.stage))
    (items.map CallStage.stage) (registry_steps primitive fallback _ unique) entry
  simpa only [List.map_map, Function.comp_def] using congrArg (fun state => state slot) logical

/-- Acceptance-to-complete-observer preservation. Exact actual retained graph,
plan and fused output correspondence are recomputed by the existing checker.
The former whole-region agreement premise is replaced by the finer instruction
and ABI premises above, under one common scalar interpretation. -/
theorem checked_calls_preserve_observers {α M : Type} [Inhabited α] {elements : Nat}
    (primitive : Primitive → List α → α)
    (fallback : ∀ node, TypedAction node (Tensor elements α) (fun _ _ => True))
    {before after : Graph} {start count : Nat} {selection : List Nat} {plan : Plan} {fused : FusedBody}
    (accepted : checkRealization before after start count selection plan fused = true)
    (items : List CallStage)
    (boundary : items.map (fun item => item.stage.node) = (before.nodes.drop start).take count)
    (addresses : Nat → Window) (known : List Nat)
    (valid : SequenceValid elements addresses known items)
    (unique : ((items.map CallStage.stage).map ScalarStage.node).Nodup)
    (read : M → Window → Option (Tensor elements α)) (invoke : BoundCall → M → Option M)
    (contracts : ∀ item ∈ items, InstructionContract read invoke primitive item.call item.stage.program)
    (pack : StateEffects.State (Tensor elements α) → M)
    (initialized : ∀ entry, Refines read addresses known entry (pack entry))
    (outputs : ∀ slot ∈ liveOuts (items.map (fun item => item.stage.node))
      (before.nodes.drop (start + count)) before.observations, slot ∈ finalKnown known items)
    (native : StateEffects.Action
      ⟨liveIns ((before.nodes.drop start).take count),
        liveOuts ((before.nodes.drop start).take count)
          (before.nodes.drop (start + count)) before.observations⟩ (Tensor elements α))
    (publication : ∀ entry final, runCalls invoke items (pack entry) = some final →
      ∀ slot ∈ liveOuts ((before.nodes.drop start).take count)
        (before.nodes.drop (start + count)) before.observations,
        read final (addresses slot) = some (native.step entry slot))
    (initial : StateEffects.State (Tensor elements α)) :
    let meaning := registry primitive fallback (items.map CallStage.stage)
    before.observations.map (executeNodes meaning (before.nodes.drop (start + count))
      (native.step (executeNodes meaning (before.nodes.take start) initial))) =
      before.observations.map (executeNodes meaning before.nodes initial) := by
  let meaning := registry primitive fallback (items.map CallStage.stage)
  let nativeItems : StateEffects.Action
      ⟨liveIns (items.map (fun item => item.stage.node)),
        liveOuts (items.map (fun item => item.stage.node))
          (before.nodes.drop (start + count)) before.observations⟩ (Tensor elements α) := {
    step := native.step
    frame := by simpa only [boundary] using native.frame
    locality := by simpa only [boundary] using native.locality }
  have itemLaw := derived_local_contract primitive fallback items addresses known valid unique read invoke contracts
    pack initialized (before.nodes.drop (start + count)) before.observations outputs nativeItems
    (by intro entry final ran slot member; exact publication entry final ran slot (by simpa only [boundary] using member))
  have law : CompositionLaw meaning ((before.nodes.drop start).take count)
      (before.nodes.drop (start + count)) before.observations native := by
    intro entry typed slot member
    have result := itemLaw entry typed slot (by simpa only [boundary] using member)
    simpa only [nativeItems, boundary] using result
  have fusion : checkFusion ((before.nodes.drop start).take count)
      (before.nodes.drop (start + count)) before.observations fused = true := by
    simp only [checkRealization, Bool.and_eq_true] at accepted
    exact accepted.2
  simp only [checkFusion, Bool.and_eq_true, decide_eq_true_eq] at fusion
  apply checkRealization_native accepted meaning native _ initial (fun _ => trivial)
  intro entry typed slot member
  have result := law entry typed slot member
  simpa only [FusedBody.execute, fusion.1.1, fusion.2, publish, if_pos member] using result

end Merlin.GraphRegionsDeclared
