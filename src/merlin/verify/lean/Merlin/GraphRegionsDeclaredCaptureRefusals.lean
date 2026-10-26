/- Restricted captured-source refusals. These are structural logical
consequences, not executed tests or authoritative compiler-valid mutants.
Actual provider mutations and compiler/Lean refusals remain held separately. -/
import Merlin.GraphRegionsDeclaredCaptureProjection

namespace Merlin.GraphRegionsDeclared.Capture
open MerlinVerify Merlin.GraphRegions Merlin.GraphRegionsDeclared

/-- Wrong same-typed SSA capture names are refused by the actual reference
checker, independent of scalar evaluation and any sampled numerical values. -/
theorem wrong_capture_identity_refused {leading : List Op} {program : Program} {names : List Nat}
    {reference : Reference}
    (wrong : names[reference.slot]? ≠ some reference.sourceId) :
    checkReference leading program names reference = false := by
  cases observed : checkReference leading program names reference with
  | false => rfl
  | true =>
    obtain ⟨_, _, _, _, _, named, _⟩ := checkReference_sound observed
    exact False.elim (wrong named)

/-- A nonconstant/missing dominating operation cannot be laundered by a
canonical literal hint. Complete operation properties and SSA results matter. -/
theorem wrong_capture_definition_refused {leading : List Op} {program : Program} {names : List Nat}
    {reference : Reference}
    (wrong : ∀ width literal,
      leading[reference.prefixIndex]? ≠ some ((⟨.constant width literal, []⟩ : Instruction).operation reference.sourceId)) :
    checkReference leading program names reference = false := by
  cases observed : checkReference leading program names reference with
  | false => rfl
  | true =>
    obtain ⟨_, width, literal, instruction, _, _, primitive, operands, definition⟩ := checkReference_sound observed
    apply False.elim
    apply wrong width literal
    simpa only [Instruction.operation, primitive, operands] using definition

/-- Duplicate materialization slots cannot suppress two different source
definitions. This follows from the recomputed capture-list freshness check. -/
theorem duplicate_capture_slot_refused {leading : List Op} {program : Program} {names : List Nat}
    {references : List Reference} (duplicate : ¬ (references.map (·.slot)).Nodup) :
    checkReferences leading program names references = false := by
  simp [checkReferences, duplicate]

/-- Empty tensor elements remain uninitialized in the scalar prefix projection;
the literal-initialization proof cannot be reused to claim empty initialization. -/
theorem empty_projection_uninitialized {α : Type} (meaning : Primitive → List α → α)
    (result width : Nat) (shape : List Nat) :
    Prefix.Instruction.value meaning (.empty result shape width) = (none : Option α) := rfl

/-- All actual operations/helpers must fit the complete parent grammar. A
module with an extra function cannot pass through a local-region-only guard. -/
theorem extra_top_level_refused {parent : GraphParent} {actual retained : Module}
    (extra : actual.ops.length ≠ 1) : ¬ parent.Accepts actual retained := by
  intro accepted
  apply extra
  rw [accepted.1]
  rfl

/-- Dropping or swapping an externally observed return cannot qualify as this
captured-source parent, even if all scalar signatures remain unchanged. -/
theorem changed_complete_parent_refused {parent : GraphParent} {actual retained : Module}
    (changed : actual ≠ parent.module) : ¬ parent.Accepts actual retained := by
  intro accepted
  exact changed accepted.1

/-- Matching node shapes alone cannot admit a call whose full scalar program
is absent from the source parent. The complete stage is compared structurally. -/
theorem wrong_call_stage_refused {parent : GraphParent} {items : List CallStage} {item : CallStage}
    (present : item ∈ items) (wrong : item.stage ∉ parent.stages) :
    checkCallSites parent items = false := by
  cases observed : checkCallSites parent items with
  | false => rfl
  | true => exact False.elim (wrong (checkCallSites_sound observed item present))

/-- Tensor sites may not read an uninitialized empty destination or expose a
discarded prefix result. Its exact SSA identity is checked against ALL reads and
returns, including unchanged sites. This is not a sampled definedness claim. -/
theorem observed_prefix_refused {parent : GraphParent} {slot : Nat}
    (written : slot ∈ parent.leading.map Prefix.Instruction.result)
    (observed : slot ∈ reads parent.graph.nodes ++ parent.graph.observations) :
    checkPrefixPrivate parent = false := by
  cases accepted : checkPrefixPrivate parent with
  | false => rfl
  | true =>
    have unobserved : ∀ id ∈ parent.leading.map Prefix.Instruction.result,
        id ∉ reads parent.graph.nodes ++ parent.graph.observations := by
      simpa only [checkPrefixPrivate, List.all_eq_true, decide_eq_true_eq] using accepted
    exact False.elim (unobserved slot written observed)

/-- A stale/substituted graph projection cannot be accepted by exact source
buffer equality alone. The full graph projection is independently recomputed. -/
theorem stale_full_projection_refused {parent : GraphParent} {actual retained : Module}
    (wrong : projectModule actual (parent.sites.map (·.generic)) ≠ some parent.fullGraph) :
    ¬ FullSourceBinding parent actual retained := by
  intro accepted
  exact wrong accepted.2.1

end Merlin.GraphRegionsDeclared.Capture
