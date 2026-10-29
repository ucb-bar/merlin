/- Explicit frozen source execution -> existing GraphRegions execution -> checked
observer preservation. Source success and initialized intermediates are conclusions.
The initial input-binding construction is supplied by GraphRegionsSourceInputs. -/
import Merlin.GraphRegionsSourceProgram

namespace Merlin.GraphRegionsSource
open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

def graphNodes (items : List Item) : List GraphRegions.Node := items.flatMap Item.node

theorem runSourceState_graph {entries : List Entry} {types : Nat → TensorType}
    {logical initializers : TensorTypes} {constants : List CapturedConstant} {items : List Item}
    (valid : ItemsValid entries types logical initializers constants items) (state : Nat → Tensor) :
    runSourceState items state = GraphRegions.executeNodes (sourceMeaning entries types) (graphNodes items) state := by
  induction items generalizing logical initializers constants state with
  | nil => rfl
  | cons item rest ih =>
    rw [runSourceState, ih valid.2]
    cases item with
    | empty result type => rfl
    | constant value => rfl
    | generic entry =>
      have selected := valid.1.2.1
      have step : (sourceMeaning entries types entry.node).action.step state =
          put state entry.stage.result (compute entry state) := sourceStep_selected selected state
      change GraphRegions.executeNodes (sourceMeaning entries types) (graphNodes rest)
        (put state entry.stage.result (compute entry state)) =
        GraphRegions.executeNodes (sourceMeaning entries types) (entry.node :: graphNodes rest) state
      simp only [GraphRegions.executeNodes, List.map_cons, StateEffects.execute, List.foldl_cons, step]

def ReturnedTyped (logical : TensorTypes) (returned : List Nat) (resultTypes : List TensorType) : Prop :=
  List.Forall₂ (fun id type => logical.lookup id = some type) returned resultTypes

theorem returned_bound {logical : TensorTypes} {environment : Env} {state : Nat → Tensor}
    (related : EnvRel logical environment state) {returned : List Nat} {resultTypes : List TensorType}
    (typed : ReturnedTyped logical returned resultTypes) :
    environment.getAll returned = .ok ((returned.map state).map RVal.tensor) := by
  induction typed with
  | nil => rfl
  | @cons id type ids types head tail ih =>
    change (Outcome.ofOption "unbound SSA value" (environment.get? id)).bind
      (fun value => (environment.getAll ids).bind (fun rest => .ok (value :: rest))) = _
    rw [(related id type head).1]
    simp only [Outcome.ofOption, Outcome.bind_ok, ih, List.map_cons]

theorem returned_initialized {logical : TensorTypes} {environment : Env} {state : Nat → Tensor}
    (related : EnvRel logical environment state) {returned : List Nat} {resultTypes : List TensorType}
    (typed : ReturnedTyped logical returned resultTypes) :
    ResultsDefined ((returned.map state).map RVal.tensor) := by
  induction typed with
  | nil => simp [ResultsDefined]
  | @cons id type ids types head tail ih =>
    intro value member
    rcases List.mem_cons.mp member with rfl | member
    · exact initialized_defined (related id type head).2
    · exact ih value member

def returnOperation (returned : List Nat) : Op := .mk "func.return" [] returned [] [] [] []

/-- Complete actual source body execution, returned order and initializedness,
related to the constructed logical graph. No source-success premise. -/
theorem source_observers {entries : List Entry} {types : Nat → TensorType}
    {logical initializers : TensorTypes} {constants : List CapturedConstant} {items : List Item}
    (valid : ItemsValid entries types logical initializers constants items)
    {environment : Env} {state : Nat → Tensor}
    (related : EnvRel logical environment state) (pool : PoisonRel initializers environment)
    (captured : ConstantsRel constants environment)
    {returned : List Nat} {resultTypes : List TensorType}
    (typed : ReturnedTyped (finalLogical items logical) returned resultTypes)
    (ctx : Ctx) (call : CallSem) :
    evalOps opSem ctx call environment (items.map Item.operation ++ [returnOperation returned]) =
        .ok ((returned.map (GraphRegions.executeNodes (sourceMeaning entries types) (graphNodes items) state)).map
          RVal.tensor) ∧
      ResultsDefined ((returned.map (GraphRegions.executeNodes (sourceMeaning entries types) (graphNodes items) state)).map
          RVal.tensor) := by
  obtain ⟨executed, finalRelated, _⟩ := source_program_execution valid related pool captured ctx call
  have observed := returned_bound finalRelated typed
  have initialized := returned_initialized finalRelated typed
  rw [runSourceState_graph valid] at observed initialized
  refine ⟨?_, initialized⟩
  rw [evalOps_append]
  · rw [executed, Outcome.bind_ok]
    change (runSourceEnvironment items state environment).getAll returned = _
    exact observed
  · intro op member
    obtain ⟨item, _, rfl⟩ := List.mem_map.mp member
    exact item.nonterminator

/-- The existing accepted complete-region theorem now uses the constructed frozen
integer source interpretation. The native implementation contract is still open. -/
theorem source_checked_observers {entries : List Entry} {types : Nat → TensorType}
    {logical initializers : TensorTypes} {constants : List CapturedConstant} {items : List Item}
    (valid : ItemsValid entries types logical initializers constants items)
    {environment : Env} {state : Nat → Tensor}
    (related : EnvRel logical environment state) (pool : PoisonRel initializers environment)
    (captured : ConstantsRel constants environment)
    {returned : List Nat} {resultTypes : List TensorType}
    (typed : ReturnedTyped (finalLogical items logical) returned resultTypes)
    {before after : GraphRegions.Graph} {start count : Nat} {selection : List Nat}
    {plan : GraphRegions.Plan} {fused : GraphRegions.FusedBody}
    (nodes : before.nodes = graphNodes items) (observations : before.observations = returned)
    (accepted : GraphRegions.checkRealization before after start count selection plan fused = true)
    (entryTyped : GraphRegions.WellTyped (fun slot tensor => (types slot).initialized tensor) state)
    (ctx : Ctx) (call : CallSem) :
    evalOps opSem ctx call environment (items.map Item.operation ++ [returnOperation returned]) =
      .ok ((before.observations.map
        (GraphRegions.executeNodes (sourceMeaning entries types) (before.nodes.drop (start + count))
          (fused.execute (sourceMeaning entries types)
            (GraphRegions.executeNodes (sourceMeaning entries types) (before.nodes.take start) state)))).map RVal.tensor) := by
  have source := (source_observers valid related pool captured typed ctx call).1
  have observer := (GraphRegions.checkRealization_sound accepted (sourceMeaning entries types) state entryTyped).2.2
  rw [nodes, observations] at observer
  rw [source, nodes, observations, observer]

end Merlin.GraphRegionsSource
