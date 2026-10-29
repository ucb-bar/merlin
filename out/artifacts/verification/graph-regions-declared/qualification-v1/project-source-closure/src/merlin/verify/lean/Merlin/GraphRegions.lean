/-
Checked connected-region correspondence at the existing StateEffects.Action model.
Node functions are arbitrary deterministic total pure actions with explicit typed
preservation, frame and read-locality premises. This module does not identify the
older MerlinVerify operational semantics or qualify a target implementation.

Soundness target: acceptance of a complete SSA-bound plan preserves execution of
the actual ordered source nodes in every surrounding sequential context, and keeps
all externally observed results. The source projection and native realization
contracts remain separate named obligations.
-/
import Merlin.StateEffects

namespace Merlin.GraphRegions
open Merlin.StateEffects

structure Node where
  ordinal : Nat
  semanticKey : String
  operands : List Nat
  results : List Nat
  operandTypes : List String
  resultTypes : List String
  deriving DecidableEq, Repr

def Node.footprint (node : Node) : Footprint := ⟨node.operands, node.results⟩

def WellTyped (typing : Nat → Value → Prop) (state : State Value) : Prop :=
  ∀ slot, typing slot (state slot)

/-- Logical SSA writes are not hidden memory effects. A native implementation must
independently establish these premises under its declared composition contract. -/
structure TypedAction (node : Node) (Value : Type) (typing : Nat → Value → Prop) where
  action : Action node.footprint Value
  preservesTypes : ∀ state, WellTyped typing state → WellTyped typing (action.step state)

def executeNodes (meaning : ∀ node, TypedAction node Value typing)
    (nodes : List Node) (initial : State Value) : State Value :=
  StateEffects.execute (nodes.map fun node => (meaning node).action.step) initial

def AgreeOn (slots : List Nat) (first second : State Value) : Prop :=
  ∀ slot, slot ∈ slots → first slot = second slot

def writes (nodes : List Node) : List Nat := nodes.flatMap (·.results)
def reads (nodes : List Node) : List Nat := nodes.flatMap (·.operands)

/-- Ordered liveness: a later read is external exactly when no earlier result
supplies it. Duplicates are harmless for agreement and are not used as SSA identity. -/
def liveIns : List Node → List Nat
  | [] => []
  | node :: rest => node.operands ++ (liveIns rest).filter (fun slot => !node.results.contains slot)

def liveOuts (region outside : List Node) (observations : List Nat) : List Nat :=
  (writes region).filter (fun slot => (reads outside ++ observations).contains slot)

/-- Every external read of a region-defined value is retained, including an
unchanged observer or a second consumer outside the selected region. -/
theorem external_read_retained {region outside : List Node} {observations : List Nat} {slot : Nat}
    (defined : slot ∈ writes region) (observed : slot ∈ reads outside ++ observations) :
    slot ∈ liveOuts region outside observations := by
  simp only [liveOuts, List.mem_filter, List.contains_iff_mem]
  exact ⟨defined, observed⟩

theorem execute_frame (meaning : ∀ node, TypedAction node Value typing)
    (nodes : List Node) (initial : State Value) (slot : Nat)
    (untouched : slot ∉ writes nodes) : executeNodes meaning nodes initial slot = initial slot := by
  induction nodes generalizing initial with
  | nil => rfl
  | cons node rest ih =>
    have fresh : slot ∉ node.results := fun h => untouched (by simp [writes, h])
    have tailFresh : slot ∉ writes rest := by
      intro h
      apply untouched
      change slot ∈ node.results ++ writes rest
      exact List.mem_append_right _ h
    change executeNodes meaning rest ((meaning node).action.step initial) slot = _
    rw [ih _ tailFresh, (meaning node).action.frame initial slot fresh]

theorem execute_typed (meaning : ∀ node, TypedAction node Value typing)
    (nodes : List Node) (initial : State Value) (domain : WellTyped typing initial) :
    WellTyped typing (executeNodes meaning nodes initial) := by
  induction nodes generalizing initial with
  | nil => exact domain
  | cons node rest ih =>
    exact ih ((meaning node).action.step initial) ((meaning node).preservesTypes initial domain)

/-- Inputs determine all region results; totality and locality come from the
existing Action interface, not a planner-supplied Boolean purity annotation. -/
theorem execute_locality (meaning : ∀ node, TypedAction node Value typing)
    (nodes : List Node) (first second : State Value)
    (inputs : AgreeOn (liveIns nodes) first second) :
    AgreeOn (writes nodes) (executeNodes meaning nodes first) (executeNodes meaning nodes second) := by
  induction nodes generalizing first second with
  | nil => intro slot h; simp [writes] at h
  | cons node rest ih =>
    have args : AgreeOn node.operands first second := by
      intro slot h; exact inputs slot (by simp [liveIns, h])
    have written : AgreeOn node.results ((meaning node).action.step first) ((meaning node).action.step second) :=
      (meaning node).action.locality first second args
    have tailInputs : AgreeOn (liveIns rest) ((meaning node).action.step first) ((meaning node).action.step second) := by
      intro slot h
      by_cases hit : slot ∈ node.results
      · exact written slot hit
      · rw [(meaning node).action.frame first slot hit, (meaning node).action.frame second slot hit]
        exact inputs slot (by simp [liveIns, h, hit])
    have tail := ih _ _ tailInputs
    intro slot h
    change executeNodes meaning rest ((meaning node).action.step first) slot =
      executeNodes meaning rest ((meaning node).action.step second) slot
    by_cases hit : slot ∈ writes rest
    · exact tail slot hit
    · rw [execute_frame meaning rest _ slot hit, execute_frame meaning rest _ slot hit]
      apply written slot
      have parts : slot ∈ node.results ∨ slot ∈ writes rest := by
        change slot ∈ node.results ++ writes rest at h
        exact List.mem_append.mp h
      exact parts.resolve_right hit

structure Stage where
  ordinal : Nat
  semanticKey : String
  inputs : List Nat
  outputs : List Nat
  operandTypes : List String
  resultTypes : List String
  deriving DecidableEq, Repr

/-- Plan indices must resolve to complete ordered source operand/result IDs. -/
def Stage.decode (values : List Nat) (stage : Stage) : Option Node := do
  let operands ← stage.inputs.mapM (values[·]?)
  let results ← stage.outputs.mapM (values[·]?)
  pure ⟨stage.ordinal, stage.semanticKey, operands, results, stage.operandTypes, stage.resultTypes⟩

structure Plan where
  values : List Nat
  stages : List Stage
  inputs : List Nat
  outputs : List Nat
  lifetimes : List (Int × Nat)
  deriving DecidableEq, Repr

def Plan.decode (plan : Plan) : Option (List Node) := plan.stages.mapM (Stage.decode plan.values)

def regionValues (region : List Node) : List Nat :=
  (region.flatMap fun node => node.operands ++ node.results).eraseDups

def lifetime (region outside : List Node) (observations : List Nat) (slot : Nat) : Int × Nat :=
  let created := region.findIdx? (fun node => node.results.contains slot)
  let first := created.map Int.ofNat |>.getD (-1)
  let last := if (liveIns region ++ liveOuts region outside observations).contains slot then region.length else
    (region.zipIdx).foldl (fun latest (node, index) =>
      if node.operands.contains slot || node.results.contains slot then max latest index else latest) 0
  (first, last)

def checkPlan (region outside : List Node) (observations : List Nat) (plan : Plan) : Bool :=
  decide (plan.decode = some region) && decide (plan.values = regionValues region) &&
  decide (plan.lifetimes = plan.values.map (lifetime region outside observations)) &&
  (match plan.inputs.mapM (plan.values[·]?), plan.outputs.mapM (plan.values[·]?) with
    | some inputs, some outputs =>
      decide (inputs = (liveIns region).eraseDups) && decide (outputs = liveOuts region outside observations)
    | _, _ => false)

theorem checkPlan_decode {region outside : List Node} {observations : List Nat} {plan : Plan}
    (accepted : checkPlan region outside observations plan = true) : plan.decode = some region := by
  simp only [checkPlan, Bool.and_eq_true, decide_eq_true_eq] at accepted
  exact accepted.1.1.1

/-- Acceptance connects the actual decoded plan to existing sequential execution,
preserving every state slot and arbitrary unchanged prefix/suffix actions. -/
theorem checkPlan_preserves_context {region outside : List Node} {observations : List Nat} {plan : Plan}
    (accepted : checkPlan region outside observations plan = true)
    (meaning : ∀ node, TypedAction node Value typing)
    (before after : List Node) (initial : State Value) :
    ∃ planned, plan.decode = some planned ∧
      executeNodes meaning (before ++ planned ++ after) initial =
        executeNodes meaning (before ++ region ++ after) initial :=
  ⟨region, checkPlan_decode accepted, rfl⟩

theorem checkPlan_interface {region outside : List Node} {observations : List Nat} {plan : Plan}
    (accepted : checkPlan region outside observations plan = true) :
    ∃ inputs outputs, plan.inputs.mapM (plan.values[·]?) = some inputs ∧
      plan.outputs.mapM (plan.values[·]?) = some outputs ∧
      inputs = (liveIns region).eraseDups ∧ outputs = liveOuts region outside observations := by
  cases hi : plan.inputs.mapM (plan.values[·]?) with
  | none => simp [checkPlan, hi] at accepted
  | some inputs =>
    cases ho : plan.outputs.mapM (plan.values[·]?) with
    | none => simp [checkPlan, hi, ho] at accepted
    | some outputs =>
      simp only [checkPlan, hi, ho, Bool.and_eq_true, decide_eq_true_eq] at accepted
      exact ⟨inputs, outputs, rfl, rfl, accepted.2.1, accepted.2.2⟩

/-- The plan publishes precisely the complete region outputs, rather than merely
retaining their values in an internal mathematical state. -/
theorem checkPlan_publishes {region outside : List Node} {observations : List Nat} {plan : Plan}
    (accepted : checkPlan region outside observations plan = true)
    (meaning : ∀ node, TypedAction node Value typing) (initial : State Value) :
    ∃ planned outputs, plan.decode = some planned ∧
      plan.outputs.mapM (plan.values[·]?) = some outputs ∧
      outputs.map (executeNodes meaning planned initial) =
        (liveOuts region outside observations).map (executeNodes meaning region initial) := by
  obtain ⟨inputs, outputs, _, mapped, _, equal⟩ := checkPlan_interface accepted
  exact ⟨region, outputs, checkPlan_decode accepted, mapped, by rw [equal]⟩

/-- An unchanged suffix may read every externally visible region value, including
multiple consumers. Its complete read set must agree, not merely its result shape. -/
theorem suffix_congruence (meaning : ∀ node, TypedAction node Value typing)
    (suffix : List Node) (observations : List Nat) (first second : State Value)
    (same : AgreeOn (reads suffix ++ observations) first second) :
    AgreeOn observations (executeNodes meaning suffix first) (executeNodes meaning suffix second) := by
  induction suffix generalizing first second with
  | nil => simpa [reads, executeNodes, StateEffects.execute] using same
  | cons node rest ih =>
    have args : AgreeOn node.operands first second := by
      intro slot h
      apply same slot
      change slot ∈ (node.operands ++ reads rest) ++ observations
      simp [h]
    have next : AgreeOn (reads rest ++ observations)
        ((meaning node).action.step first) ((meaning node).action.step second) := by
      intro slot h
      by_cases hit : slot ∈ node.results
      · exact (meaning node).action.locality first second args slot hit
      · rw [(meaning node).action.frame first slot hit, (meaning node).action.frame second slot hit]
        apply same slot
        change slot ∈ (node.operands ++ reads rest) ++ observations
        rcases List.mem_append.mp h with hr | ho
        · exact List.mem_append_left _ (List.mem_append_right _ hr)
        · exact List.mem_append_right _ ho
    exact ih _ _ next

/-- Composition contract for one realized region. It is a local output law, not
an assumption of whole-program equivalence. Its physical/native implementation
and effect declaration need independent evidence. -/
def CompositionLaw (meaning : ∀ node, TypedAction node Value typing)
    (region suffix : List Node) (observations : List Nat)
    (replacement : Action ⟨liveIns region, liveOuts region suffix observations⟩ Value) : Prop :=
  ∀ initial, WellTyped typing initial → AgreeOn (liveOuts region suffix observations)
    (replacement.step initial) (executeNodes meaning region initial)

/-- A region implementation satisfying its declared local law preserves every
observation after the actual unchanged suffix, under an arbitrary unchanged prefix.
The original source's internal unobserved bindings need not be materialized. -/
theorem replace_region_preserves_observers (meaning : ∀ node, TypedAction node Value typing)
    (earlier region suffix : List Node) (observations : List Nat)
    (replacement : Action ⟨liveIns region, liveOuts region suffix observations⟩ Value)
    (law : CompositionLaw meaning region suffix observations replacement) (initial : State Value)
    (domain : WellTyped typing initial) :
    observations.map (executeNodes meaning suffix (replacement.step (executeNodes meaning earlier initial))) =
      observations.map (executeNodes meaning (earlier ++ region ++ suffix) initial) := by
  let entry := executeNodes meaning earlier initial
  have boundary : AgreeOn (reads suffix ++ observations)
      (replacement.step entry) (executeNodes meaning region entry) := by
    intro slot observed
    by_cases changed : slot ∈ writes region
    · exact law entry (execute_typed meaning earlier initial domain) slot (external_read_retained changed observed)
    · have outside : slot ∉ liveOuts region suffix observations := by
        intro h
        have defined := (List.mem_filter.mp h).1
        exact changed defined
      rw [replacement.frame entry slot outside, execute_frame meaning region entry slot changed]
  have final := suffix_congruence meaning suffix observations _ _ boundary
  have observed := List.map_congr_left final
  simpa only [entry, executeNodes, StateEffects.execute, List.map_append, List.foldl_append] using observed

structure Graph where
  nodes : List Node
  inputs : List Nat
  observations : List Nat
  context : List String
  deriving DecidableEq, Repr

def ValidNodes (available : List Nat) : List Node → Prop
  | [] => True
  | node :: rest =>
    node.operands.length = node.operandTypes.length ∧ node.results.length = node.resultTypes.length ∧
    node.results ≠ [] ∧ node.results.Nodup ∧
    (∀ slot ∈ node.operands, slot ∈ available) ∧
    (∀ slot ∈ node.results, slot ∉ available) ∧ ValidNodes (available ++ node.results) rest

def validNodesDec (available : List Nat) : (nodes : List Node) → Decidable (ValidNodes available nodes)
  | [] => isTrue trivial
  | node :: rest =>
    letI := validNodesDec (available ++ node.results) rest
    inferInstanceAs (Decidable (
      node.operands.length = node.operandTypes.length ∧ node.results.length = node.resultTypes.length ∧
      node.results ≠ [] ∧ node.results.Nodup ∧
      (∀ slot ∈ node.operands, slot ∈ available) ∧
      (∀ slot ∈ node.results, slot ∉ available) ∧ ValidNodes (available ++ node.results) rest))

instance (available : List Nat) (nodes : List Node) : Decidable (ValidNodes available nodes) :=
  validNodesDec available nodes

def Graph.Valid (graph : Graph) : Prop :=
  graph.inputs.Nodup ∧ ValidNodes graph.inputs graph.nodes ∧
  ∀ slot ∈ graph.observations, slot ∈ graph.inputs ++ writes graph.nodes

instance (graph : Graph) : Decidable graph.Valid := by
  unfold Graph.Valid
  infer_instance

/-- The planner retains actual IR. Check that complete retained context, including
initializers and unchanged observers, before accepting its changed representation. -/
def checkBoundary (before after : Graph) (start count : Nat) (selection : List Nat) (plan : Plan) : Bool :=
  let region := (before.nodes.drop start).take count
  let outside := before.nodes.take start ++ before.nodes.drop (start + count)
  decide before.Valid && decide (before = after) &&
    decide (2 ≤ count ∧ count ≤ 64 ∧ start + count ≤ before.nodes.length) &&
    decide (region.map (·.ordinal) = selection) && checkPlan region outside before.observations plan

theorem split_nodes (nodes : List Node) (start count : Nat) :
    nodes.take start ++ (nodes.drop start).take count ++ nodes.drop (start + count) = nodes := by
  have tail := List.take_append_drop count (nodes.drop start)
  rw [List.drop_drop] at tail
  rw [List.append_assoc, tail, List.take_append_drop]

/-- Full-boundary acceptance proves unchanged actual context and a semantic plan
for the complete selected contiguous region, with no dropped external interface. -/
theorem checkBoundary_sound {before after : Graph} {start count : Nat}
    {selection : List Nat} {plan : Plan}
    (accepted : checkBoundary before after start count selection plan = true)
    (meaning : ∀ node, TypedAction node Value typing) (initial : State Value) :
    before.Valid ∧ before = after ∧
    ∃ planned outputs,
      plan.decode = some planned ∧
      plan.outputs.mapM (plan.values[·]?) = some outputs ∧
      executeNodes meaning (before.nodes.take start ++ planned ++ before.nodes.drop (start + count)) initial =
        executeNodes meaning before.nodes initial ∧
      outputs = liveOuts ((before.nodes.drop start).take count)
        (before.nodes.take start ++ before.nodes.drop (start + count)) before.observations := by
  simp only [checkBoundary, Bool.and_eq_true, decide_eq_true_eq] at accepted
  refine ⟨accepted.1.1.1.1, accepted.1.1.1.2, ?_⟩
  obtain ⟨inputs, outputs, _, mapped, _, equal⟩ := checkPlan_interface accepted.2
  refine ⟨(before.nodes.drop start).take count, outputs, checkPlan_decode accepted.2, mapped, ?_, equal⟩
  rw [split_nodes]

end Merlin.GraphRegions
