/- Closed scalar SSA execution: each result is derived, typed and initialized.
The old output argument is deliberately absent from the initial scalar types.
This proves independence of arbitrary poison initialization and unused captures. -/
import Merlin.GraphRegionsSourceScalar
import MerlinVerify.Check.Step

namespace Merlin.GraphRegionsSource
open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

abbrev ScalarTypes := List (Nat × Nat)
abbrev ScalarState := Nat → Nat

def ScalarAgree (types : ScalarTypes) (environment : Env) (values : ScalarState) : Prop :=
  ∀ id width, types.lookup id = some width → environment.get? id = some (scalar width (values id))

def put (values : Nat → α) (result : Nat) (value : α) : Nat → α :=
  fun id => if id = result then value else values id

structure Instruction where
  code : Code
  result : Nat
  operands : List Nat
  deriving DecidableEq, Repr

def Instruction.operation (instruction : Instruction) : Op :=
  instruction.code.operation instruction.result instruction.operands

def Instruction.next (instruction : Instruction) (values : ScalarState) : ScalarState :=
  put values instruction.result (instruction.code.eval (instruction.operands.map values))

def Instruction.Valid (types : ScalarTypes) (instruction : Instruction) : Prop :=
  instruction.code.Valid ∧
    List.Forall₂ (fun id width => types.lookup id = some width) instruction.operands instruction.code.inputs ∧
    instruction.result ∉ types.map Prod.fst

theorem arguments_lookup {types : ScalarTypes} {environment : Env} {values : ScalarState}
    (agree : ScalarAgree types environment values) {ids widths : List Nat}
    (typed : List.Forall₂ (fun id width => types.lookup id = some width) ids widths) :
    environment.getAll ids =
      .ok ((widths.zip (ids.map values)).map fun (width, value) => scalar width value) := by
  induction typed with
  | nil => rfl
  | @cons id width ids widths head tail ih =>
    change (Outcome.ofOption "unbound SSA value" (environment.get? id)).bind
      (fun value => (environment.getAll ids).bind (fun rest => .ok (value :: rest))) = _
    rw [agree id width head]
    simp only [Outcome.ofOption, Outcome.bind_ok, ih, List.map_cons, List.zip_cons_cons]

theorem agree_put {types : ScalarTypes} {environment : Env} {values : ScalarState}
    (agree : ScalarAgree types environment values) (id width value : Nat) :
    ScalarAgree ((id, width) :: types) ((id, scalar width value) :: environment) (put values id value) := by
  intro key ty found
  by_cases same : key = id
  · subst key
    have ht : ty = width := by simpa [List.lookup] using found.symm
    subst ty
    simp [Env.get?, List.lookup, put]
  · have different : (key == id) = false := beq_eq_false_iff_ne.mpr same
    have old : types.lookup key = some ty := by simpa [List.lookup, different] using found
    simpa [Env.get?, List.lookup, put, same, different] using agree key ty old

theorem Code.nonterminator (code : Code) (result : Nat) (operands : List Nat) :
    (code.operation result operands).isTerminator = false := by
  cases code with
  | constant _ _ => rfl
  | binary _ kind => cases kind <;> rfl
  | cast _ _ kind => cases kind <;> rfl

/-- A concrete supported step succeeds from its checked operand bindings. No
source-success or global-intermediate-typing assumption appears here. -/
theorem instruction_execution {types : ScalarTypes} {instruction : Instruction}
    (valid : instruction.Valid types) {environment : Env} {values : ScalarState}
    (agree : ScalarAgree types environment values) (ctx : Ctx) (call : CallSem) :
    stepOp opSem ctx call environment instruction.operation =
      .ok ((instruction.result, scalar instruction.code.width
        (instruction.code.eval (instruction.operands.map values))) :: environment) := by
  obtain ⟨code, operands, _⟩ := valid
  have args := arguments_lookup agree operands
  have arity : (instruction.operands.map values).length = instruction.code.inputs.length :=
    by simpa using operands.length_eq
  have semantics := primitive_execution_nat instruction.code code (instruction.operands.map values)
    arity instruction.result instruction.operands ctx call
  simp only [stepOp, Instruction.operation, Code.operation, Op.operands, Op.regions, evalRegions]
  rw [args]
  simp only [Outcome.bind_ok]
  change (opSem ctx call (instruction.code.operation instruction.result instruction.operands)
    (instruction.code.arguments (instruction.operands.map values)) []).bind _ = _
  rw [semantics]
  simp [Env.bindAll, Code.operation, Op.results, Outcome.bind_ok]

def runScalar : List Instruction → ScalarState → ScalarState
  | [], values => values
  | instruction :: rest, values => runScalar rest (instruction.next values)

def finalTypes : List Instruction → ScalarTypes → ScalarTypes
  | [], types => types
  | instruction :: rest, types => finalTypes rest ((instruction.result, instruction.code.width) :: types)

def runEnvironment : List Instruction → ScalarState → Env → Env
  | [], _, environment => environment
  | instruction :: rest, values, environment =>
    runEnvironment rest (instruction.next values)
      ((instruction.result, scalar instruction.code.width
        (instruction.code.eval (instruction.operands.map values))) :: environment)

def ScalarValid : ScalarTypes → List Instruction → Prop
  | _, [] => True
  | types, instruction :: rest => instruction.Valid types ∧
      ScalarValid ((instruction.result, instruction.code.width) :: types) rest

theorem scalar_run_agrees {types : ScalarTypes} {instructions : List Instruction}
    (valid : ScalarValid types instructions) {environment : Env} {values : ScalarState}
    (agree : ScalarAgree types environment values) :
    ScalarAgree (finalTypes instructions types) (runEnvironment instructions values environment)
      (runScalar instructions values) := by
  induction instructions generalizing types environment values with
  | nil => exact agree
  | cons instruction rest ih =>
    apply ih valid.2
    exact agree_put agree instruction.result instruction.code.width _

theorem scalar_run_execution {types : ScalarTypes} {instructions : List Instruction}
    (valid : ScalarValid types instructions) {environment : Env} {values : ScalarState}
    (agree : ScalarAgree types environment values) (ctx : Ctx) (call : CallSem) :
    execOps opSem ctx call environment (instructions.map Instruction.operation) =
      .ok (runEnvironment instructions values environment) := by
  induction instructions generalizing types environment values with
  | nil => rfl
  | cons instruction rest ih =>
    rw [List.map_cons]
    have nonterm : instruction.operation.isTerminator = false :=
      instruction.code.nonterminator instruction.result instruction.operands
    rw [execOps_cons_step opSem ctx call environment instruction.operation _ nonterm,
      instruction_execution valid.1 agree, Outcome.bind_ok]
    exact ih valid.2 (agree_put agree instruction.result instruction.code.width _)

def yieldOp (returned : Nat) : Op := .mk "linalg.yield" [] [returned] [] [] [] []

/-- Exact closed-body execution, including the explicit yield. All intermediate
bindings and the returned initialized value are derived from the entry bindings. -/
theorem scalar_body_execution {types : ScalarTypes} {instructions : List Instruction}
    (valid : ScalarValid types instructions) {environment : Env} {values : ScalarState}
    (agree : ScalarAgree types environment values) {returned width : Nat}
    (typed : (finalTypes instructions types).lookup returned = some width)
    (ctx : Ctx) (call : CallSem) :
    evalOps opSem ctx call environment (instructions.map Instruction.operation ++ [yieldOp returned]) =
      .ok [scalar width (runScalar instructions values returned)] := by
  rw [evalOps_append]
  · rw [scalar_run_execution valid agree, Outcome.bind_ok]
    have found := scalar_run_agrees valid agree returned width typed
    change (runEnvironment instructions values environment).getAll [returned] = _
    simp [Env.getAll, Outcome.mapM, Outcome.ofOption, found]
  · intro op member
    obtain ⟨instruction, _, rfl⟩ := List.mem_map.mp member
    exact instruction.code.nonterminator instruction.result instruction.operands

instance (instruction : Instruction) (types : ScalarTypes) : Decidable (instruction.Valid types) := by
  unfold Instruction.Valid
  infer_instance

def scalarValidDec (types : ScalarTypes) : (instructions : List Instruction) → Decidable (ScalarValid types instructions)
  | [] => isTrue trivial
  | instruction :: rest =>
    letI := scalarValidDec ((instruction.result, instruction.code.width) :: types) rest
    inferInstanceAs (Decidable (instruction.Valid types ∧ ScalarValid _ rest))

instance (types : ScalarTypes) (instructions : List Instruction) : Decidable (ScalarValid types instructions) :=
  scalarValidDec types instructions

def Instruction.matches (instruction : Instruction) : Op → Bool
  | .mk name [⟨id, ty⟩] operands attrs props [] [] =>
    decide (name = instruction.code.name ∧ id = instruction.result ∧ ty = .int instruction.code.width ∧
      operands = instruction.operands ∧ attrs = instruction.code.attrs ∧ props = [])
  | _ => false

theorem Instruction.matches_sound {instruction : Instruction} {source : Op}
    (accepted : instruction.matches source = true) : source = instruction.operation := by
  unfold Instruction.matches at accepted
  split at accepted
  · simp only [decide_eq_true_eq] at accepted
    obtain ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩ := accepted
    rfl
  · contradiction

def matchesInstructions : List Instruction → List Op → Bool
  | [], [] => true
  | instruction :: rest, source :: tail => instruction.matches source && matchesInstructions rest tail
  | _, _ => false

theorem matchesInstructions_sound {instructions : List Instruction} {source : List Op}
    (accepted : matchesInstructions instructions source = true) :
    source = instructions.map Instruction.operation := by
  induction instructions generalizing source with
  | nil => cases source <;> simp_all [matchesInstructions]
  | cons instruction rest ih =>
    cases source with
    | nil => simp [matchesInstructions] at accepted
    | cons op tail =>
      simp only [matchesInstructions, Bool.and_eq_true] at accepted
      rw [instruction.matches_sound accepted.1, ih accepted.2]
      rfl

end Merlin.GraphRegionsSource
