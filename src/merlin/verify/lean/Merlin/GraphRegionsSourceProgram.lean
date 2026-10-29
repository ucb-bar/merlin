/- Direct complete straight-line source execution. Poison allocations and logical
initialized values have distinct invariants; successful execution is a conclusion.
All previous Core/Sem and GraphRegions definitions are imported unchanged. -/
import Merlin.GraphRegionsSourceEnvironment

namespace Merlin.GraphRegionsSource
open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

def emptyOperation (result : Nat) (type : TensorType) : Op :=
  .mk "tensor.empty" [⟨result, type.ty⟩] [] [] [] [] []

theorem empty_execution (result : Nat) (type : TensorType) (ctx : Ctx) (call : CallSem) :
    opSem ctx call (emptyOperation result type) [] [] = .ok [.tensor (poisonTensor type)] := by
  have shape : (type.shape.map some).mapM id = some type.shape := by
    induction type.shape with
    | nil => rfl
    | cons dimension shape ih => simp [ih]
  have modelled : unmodelledAttr? (emptyOperation result type).blind = none := rfl
  have dialect : dialectOf (emptyOperation result type).blind.name = "tensor" := by
    change dialectOf "tensor.empty" = "tensor"
    cbv
  rw [opSem_modelled modelled]
  simp only [dispatch, dialect]
  change Tensor.sem ctx call (emptyOperation result type).blind [] [] = _
  simp [Tensor.sem, emptyOperation, Op.blind, Op.name, Op.results, Tensor.resultTensorTy,
    TensorType.ty, Ty.staticTensor?, shape, poisonTensor]

theorem empty_step (result : Nat) (type : TensorType) (ctx : Ctx) (call : CallSem) (environment : Env) :
    stepOp opSem ctx call environment (emptyOperation result type) =
      .ok ((result, .tensor (poisonTensor type)) :: environment) := by
  simp only [stepOp, emptyOperation, Op.operands, Env.getAll, Outcome.mapM, Outcome.bind_ok,
    Op.regions, evalRegions]
  change (opSem ctx call (emptyOperation result type) [] []).bind _ = _
  rw [empty_execution]
  simp [Env.bindAll, emptyOperation, Op.results, Outcome.bind_ok]

theorem source_stage_step {entries : List Entry} {types : Nat → TensorType} {entry : Entry}
    (selected : selectEntry entries types entry.node = some entry) (valid : entry.stage.Valid)
    {logical initializers : TensorTypes} {environment : Env} {state : Nat → Tensor}
    (related : EnvRel logical environment state) (pool : PoisonRel initializers environment)
    (typed : entry.stage.InputsTyped logical)
    (destination : initializers.lookup entry.stage.destination = some entry.stage.output)
    (captures : ScalarAgree (constantTypes entry.stage.body.captures) environment
      (constantState entry.stage.body.captures))
    (ctx : Ctx) (call : CallSem) :
    stepOp opSem ctx call environment entry.stage.operation =
      .ok ((entry.stage.result, .tensor (compute entry state)) :: environment) := by
  have initialized := source_inputs_initialized related typed
  have allInputs := source_inputs_bound related typed
  have old := pool _ _ destination
  have all : environment.getAll (entry.stage.inputs ++ [entry.stage.destination]) =
      .ok (((entry.stage.inputs.map state).map RVal.tensor) ++ [.tensor (poisonTensor entry.stage.output)]) := by
    rw [getAll_append, allInputs, Outcome.bind_ok]
    simp [Env.getAll, Outcome.mapM, Outcome.ofOption, old, Outcome.bind_ok]
  have execution := stage_execution entry.stage valid _ initialized
    (poisonTensor entry.stage.output) rfl rfl ctx call environment captures
  have operands : entry.stage.operation.operands = entry.stage.inputs ++ [entry.stage.destination] := rfl
  have results : entry.stage.operation.results = [⟨entry.stage.result, entry.stage.output.ty⟩] := rfl
  simp only [stepOp, operands]
  rw [all, Outcome.bind_ok, execution]
  simp [Env.bindAll, results, Outcome.bind_ok, compute]

inductive Item where
  | empty (result : Nat) (type : TensorType)
  | constant (value : CapturedConstant)
  | generic (entry : Entry)
  deriving DecidableEq, Repr

def Item.operation : Item → Op
  | .empty result type => emptyOperation result type
  | .constant value => value.operation
  | .generic entry => entry.stage.operation

def Item.node : Item → List GraphRegions.Node
  | .empty .. => []
  | .constant _ => []
  | .generic entry => [entry.node]

def Item.nextLogical (logical : TensorTypes) : Item → TensorTypes
  | .empty .. => logical
  | .constant _ => logical
  | .generic entry => (entry.stage.result, entry.stage.output) :: logical

def Item.nextInitializers (initializers : TensorTypes) : Item → TensorTypes
  | .empty result type => (result, type) :: initializers
  | .constant _ => initializers
  | .generic _ => initializers

def Item.nextConstants (constants : List CapturedConstant) : Item → List CapturedConstant
  | .constant value => value :: constants
  | _ => constants

def Item.nextState (state : Nat → Tensor) : Item → Nat → Tensor
  | .empty .. => state
  | .constant _ => state
  | .generic entry => put state entry.stage.result (compute entry state)

def Item.nextEnvironment (state : Nat → Tensor) (environment : Env) : Item → Env
  | .empty result type => (result, .tensor (poisonTensor type)) :: environment
  | .constant value => (value.id, scalar value.width value.value) :: environment
  | .generic entry => (entry.stage.result, .tensor (compute entry state)) :: environment

def Item.Valid (entries : List Entry) (types : Nat → TensorType)
    (logical initializers : TensorTypes) (constants : List CapturedConstant) : Item → Prop
  | .empty result type => type.Valid ∧
      result ∉ (logical ++ initializers).map Prod.fst ++ constants.map CapturedConstant.id
  | .constant value => 0 < value.width ∧
      value.id ∉ (logical ++ initializers).map Prod.fst ++ constants.map CapturedConstant.id
  | .generic entry => entry.stage.Valid ∧ selectEntry entries types entry.node = some entry ∧
    entry.stage.InputsTyped logical ∧ initializers.lookup entry.stage.destination = some entry.stage.output ∧
    CapturesAvailable constants entry.stage.body.captures ∧
    entry.stage.result ∉ (logical ++ initializers).map Prod.fst ++ constants.map CapturedConstant.id

theorem constant_step (value : CapturedConstant) (positive : 0 < value.width)
    (ctx : Ctx) (call : CallSem) (environment : Env) :
    stepOp opSem ctx call environment value.operation =
      .ok ((value.id, scalar value.width value.value) :: environment) := by
  have execution := primitive_execution_nat (.constant value.width value.literal) positive [] rfl value.id [] ctx call
  simp only [Code.arguments, Code.inputs, List.zip_nil_right, List.map_nil] at execution
  simp only [stepOp, CapturedConstant.operation, Code.operation, Op.operands, Env.getAll,
    Outcome.mapM, Outcome.bind_ok, Op.regions, evalRegions]
  change (opSem ctx call ((Code.constant value.width value.literal).operation value.id []) [] []).bind _ = _
  rw [execution]
  simp [Env.bindAll, Code.operation, Code.width, Op.results, Outcome.bind_ok,
    CapturedConstant.value, Code.eval, Code.bits] <;> rfl

/-- One actual source item advances the proved invariant. Initializers are never
made initialized by typing or a source-success premise. -/
theorem item_execution {entries : List Entry} {types : Nat → TensorType}
    {logical initializers : TensorTypes} {constants : List CapturedConstant} {item : Item}
    (valid : item.Valid entries types logical initializers constants)
    {environment : Env} {state : Nat → Tensor}
    (related : EnvRel logical environment state) (pool : PoisonRel initializers environment)
    (captured : ConstantsRel constants environment)
    (ctx : Ctx) (call : CallSem) :
    stepOp opSem ctx call environment item.operation = .ok (item.nextEnvironment state environment) ∧
      EnvRel (item.nextLogical logical) (item.nextEnvironment state environment) (item.nextState state) ∧
      PoisonRel (item.nextInitializers initializers) (item.nextEnvironment state environment) ∧
      ConstantsRel (item.nextConstants constants) (item.nextEnvironment state environment) := by
  cases item with
  | empty result type =>
    have fresh : result ∉ logical.map Prod.fst := by
      intro member
      exact valid.2 (by simp [member])
    have freshConstants : result ∉ constants.map CapturedConstant.id := by
      intro member
      exact valid.2 (by simp [member])
    exact ⟨empty_step result type ctx call environment, envRel_fresh related fresh _, poisonRel_put pool result type,
      constants_fresh captured freshConstants _⟩
  | constant value =>
    have freshLogical : value.id ∉ logical.map Prod.fst := by
      intro member; exact valid.2 (by simp [member])
    have freshInitializers : value.id ∉ initializers.map Prod.fst := by
      intro member; exact valid.2 (by simp [member])
    have freshConstants : value.id ∉ constants.map CapturedConstant.id := by
      intro member; exact valid.2 (by simp [member])
    exact ⟨constant_step value valid.1 ctx call environment, envRel_fresh related freshLogical _,
      poisonRel_fresh pool freshInitializers _, constants_cons captured value freshConstants⟩
  | generic entry =>
    obtain ⟨stageValid, selected, typed, destination, captures, fresh⟩ := valid
    have freshPool : entry.stage.result ∉ initializers.map Prod.fst := by
      intro member
      exact fresh (by simp [member])
    have freshConstants : entry.stage.result ∉ constants.map CapturedConstant.id := by
      intro member; exact fresh (by simp [member])
    exact ⟨source_stage_step selected stageValid related pool typed destination
        (constants_agree captured captures) ctx call,
      envRel_put related entry.stage.result entry.stage.output (compute entry state)
        (tensor_result_initialized _ _ _), poisonRel_fresh pool freshPool _, constants_fresh captured freshConstants _⟩

def ItemsValid (entries : List Entry) (types : Nat → TensorType) :
    TensorTypes → TensorTypes → List CapturedConstant → List Item → Prop
  | _, _, _, [] => True
  | logical, initializers, constants, item :: rest => item.Valid entries types logical initializers constants ∧
    ItemsValid entries types (item.nextLogical logical) (item.nextInitializers initializers)
      (item.nextConstants constants) rest

def runSourceEnvironment : List Item → (Nat → Tensor) → Env → Env
  | [], _, environment => environment
  | item :: rest, state, environment =>
    runSourceEnvironment rest (item.nextState state) (item.nextEnvironment state environment)

def runSourceState : List Item → (Nat → Tensor) → Nat → Tensor
  | [], state => state
  | item :: rest, state => runSourceState rest (item.nextState state)

def finalLogical : List Item → TensorTypes → TensorTypes
  | [], logical => logical
  | item :: rest, logical => finalLogical rest (item.nextLogical logical)

def finalInitializers : List Item → TensorTypes → TensorTypes
  | [], initializers => initializers
  | item :: rest, initializers => finalInitializers rest (item.nextInitializers initializers)

def finalConstants : List Item → List CapturedConstant → List CapturedConstant
  | [], constants => constants
  | item :: rest, constants => finalConstants rest (item.nextConstants constants)

theorem Item.nonterminator (item : Item) : item.operation.isTerminator = false := by
  cases item with
  | empty _ _ => rfl
  | constant value => exact (Code.constant value.width value.literal).nonterminator value.id []
  | generic entry => exact entry.stage.nonterminator

/-- Complete source execution is derived inductively from the input relation and
the static checker conditions. No intermediate invariant or success is assumed. -/
theorem source_program_execution {entries : List Entry} {types : Nat → TensorType}
    {logical initializers : TensorTypes} {constants : List CapturedConstant} {items : List Item}
    (valid : ItemsValid entries types logical initializers constants items)
    {environment : Env} {state : Nat → Tensor}
    (related : EnvRel logical environment state) (pool : PoisonRel initializers environment)
    (captured : ConstantsRel constants environment)
    (ctx : Ctx) (call : CallSem) :
    execOps opSem ctx call environment (items.map Item.operation) =
        .ok (runSourceEnvironment items state environment) ∧
      EnvRel (finalLogical items logical) (runSourceEnvironment items state environment) (runSourceState items state) ∧
      PoisonRel (finalInitializers items initializers) (runSourceEnvironment items state environment) ∧
      ConstantsRel (finalConstants items constants) (runSourceEnvironment items state environment) := by
  induction items generalizing logical initializers constants environment state with
  | nil => exact ⟨rfl, related, pool, captured⟩
  | cons item rest ih =>
    obtain ⟨step, nextRelated, nextPool, nextCaptured⟩ := item_execution valid.1 related pool captured ctx call
    obtain ⟨tail, finalRelated, finalPool, finalCaptured⟩ := ih valid.2 nextRelated nextPool nextCaptured
    refine ⟨?_, finalRelated, finalPool, finalCaptured⟩
    rw [List.map_cons, execOps_cons_step opSem ctx call environment item.operation _ item.nonterminator,
      step, Outcome.bind_ok]
    exact tail

end Merlin.GraphRegionsSource
