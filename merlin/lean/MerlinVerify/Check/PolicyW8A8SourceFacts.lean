/- Exact scalar calibration facts from executed constants and rank-zero tensor splats.
Facts are invalidated on SSA writes. No quantization implementation, provenance hint or float
identity supplies a parameter value. Whole-program source semantics remain a separate review. -/
import MerlinVerify.Check.ConstantFacts
import MerlinVerify.Rules.W8A8SourceContract

namespace MerlinVerify.Check.PolicyW8A8SourceFacts

open MerlinVerify MerlinVerify.Sem.Ops
open ConstantFacts

def parameterType (ty : Ty) : Bool := ty == .float .f32 || ty == .int 32 || ty == .int 64

def SplatEligible (op : Op) : Prop :=
  op.name = "tensor.splat" ∧ unmodelledAttr? op.blind = none ∧ op.regions = [] ∧
    op.successors = [] ∧ op.operands.length = 1 ∧ op.results.length = 1 ∧
    op.results.all (fun result => match result.ty with
      | .tensor [] elt none => parameterType elt
      | _ => false) = true

instance (op : Op) : Decidable (SplatEligible op) := by unfold SplatEligible; infer_instance

theorem splat_context_independent {op : Op} (eligible : SplatEligible op)
    (ctx : Ctx) (call : CallSem) (args : List RVal) (ib : Nat) :
    opSem ctx call op args [] = opSem (reference ib) (reference ib).ext op args [] := by
  have dialect : dialectOf op.name = "tensor" := by rw [eligible.1]; cbv
  rw [opSem_modelled eligible.2.1, opSem_modelled eligible.2.1]
  simp only [dispatch, Op.blind_name]
  rw [dialect]
  simp only [Sem.Ops.Tensor.sem, Op.blind_name, eligible.1]

def find? (facts : List Fact) (id : Nat) : Option Fact := facts.find? fun fact => fact.id == id

theorem find?_sound {facts : List Fact} {id : Nat} {fact : Fact} {env : Env}
    (valid : BindingFacts.Valid model facts env) (found : find? facts id = some fact) :
    env.get? id = some fact.value := by
  have member := List.mem_of_find?_eq_some found
  have tested : (fact.id == id) = true := List.find?_some (p := fun f : Fact => f.id == id) found
  have same : fact.id = id := eq_of_beq tested
  exact same ▸ valid fact member

def splatLearn (ib : Nat) (facts : List Fact) (op : Op) : Option Fact := do
  if SplatEligible op then
    let [input] := op.operands | none
    let source ← find? facts input
    let .ok [value] := opSem (reference ib) (reference ib).ext op [source.value] [] | none
    let [result] := op.results | none
    pure ⟨result.id, value⟩
  else none

theorem splatLearn_exec {ib : Nat} {facts : List Fact} {op : Op} {fact : Fact} {env : Env}
    (valid : BindingFacts.Valid model facts env) (learned : splatLearn ib facts op = some fact)
    (ctx : Ctx) (call : CallSem) :
    ∃ result, op.results = [result] ∧
      stepOp opSem ctx call env op = .ok ((result.id, fact.value) :: env) ∧ result.id = fact.id := by
  unfold splatLearn at learned
  split at learned
  · next eligible =>
    split at learned
    · next input operands =>
      cases found : find? facts input with
      | none => simp [found] at learned
      | some source =>
        simp only [found, bind, Option.bind] at learned
        split at learned
        · next value computed =>
          split at learned
          · next result results =>
            cases learned
            refine ⟨result, results, ?_, rfl⟩
            have binding := find?_sound valid found
            unfold stepOp
            rw [operands, eligible.2.2.1]
            simp only [Env.getAll, Outcome.mapM, binding, Outcome.ofOption, Outcome.bind_ok, evalRegions]
            rw [splat_context_independent eligible ctx call _ ib, computed]
            simp [Env.bindAll, results]
          · cases learned
        · cases learned
    · cases learned
  · cases learned

def learn (ib : Nat) (facts : List Fact) (op : Op) : Option Fact :=
  if op.name == "tensor.splat" then splatLearn ib facts op
  else if op.name == "arith.constant" && op.results.all (fun result => parameterType result.ty) then
    ConstantFacts.learn ib op
  else none

/-- A semantics extension may change quantization while preserving these two producer meanings.
This is an explicit theorem parameter, not an axiom or a new numerical law. -/
def ProducerAgreement (sem : OpSem) : Prop :=
  ∀ ctx call env op, op.name = "arith.constant" ∨ op.name = "tensor.splat" →
    stepOp sem ctx call env op = stepOp opSem ctx call env op

theorem learn_sound {ib : Nat} {facts : List Fact} {op : Op} {fact : Fact}
    {ctx : Ctx} {call : CallSem} {env env' : Env} {sem : OpSem}
    (agree : ProducerAgreement sem) (valid : BindingFacts.Valid model facts env)
    (learned : learn ib facts op = some fact) (width : ctx.indexBits = ib)
    (executed : stepOp sem ctx call env op = .ok env') : model.holds fact env' := by
  unfold learn at learned
  split at learned
  · next name =>
    have same := agree ctx call env op (Or.inr (eq_of_beq name))
    rw [same] at executed
    obtain ⟨result, _, produced, id⟩ := splatLearn_exec valid learned ctx call
    rw [produced] at executed
    cases executed
    simp [model, Env.get?, List.lookup, id]
  · split at learned
    · next condition =>
      have checkedName : (op.name == "arith.constant") = true := by
        simp only [Bool.and_eq_true] at condition
        exact condition.1
      have name : op.name = "arith.constant" := eq_of_beq checkedName
      rw [agree ctx call env op (Or.inl name)] at executed
      exact ConstantFacts.learn_sound learned width executed
    · cases learned

def advance (ib : Nat) (facts : List Fact) (op : Op) : List Fact :=
  let retained := BindingFacts.forget model (op.results.map (·.id)) facts
  match learn ib facts op with
  | none => retained
  | some fact => fact :: retained

theorem advance_sound {ib : Nat} {facts : List Fact} {op : Op}
    {ctx : Ctx} {call : CallSem} {env env' : Env} {sem : OpSem}
    (agree : ProducerAgreement sem) (valid : BindingFacts.Valid model facts env)
    (width : ctx.indexBits = ib) (executed : stepOp sem ctx call env op = .ok env') :
    BindingFacts.Valid model (advance ib facts op) env' := by
  have retained := BindingFacts.forget_step model valid executed
  unfold advance
  cases learned : learn ib facts op with
  | none => exact retained
  | some fact =>
    intro known member
    rcases List.mem_cons.1 member with rfl | member
    · exact learn_sound agree valid learned width executed
    · exact retained known member

theorem frozen_producers : ProducerAgreement opSem := fun _ _ _ _ _ => rfl

def value? (facts : List Fact) (id : Nat) : Option RVal := (find? facts id).map (·.value)

theorem value?_sound {facts : List Fact} {id : Nat} {value : RVal} {env : Env}
    (valid : BindingFacts.Valid model facts env) (found : value? facts id = some value) :
    env.get? id = some value := by
  unfold value? at found
  cases stored : find? facts id with
  | none => simp [stored] at found
  | some fact =>
    have equal : fact.value = value := by simpa [stored] using found
    exact (find?_sound valid stored).trans (congrArg some equal)

/-- Calibration values refer to the actual currently bound scalar tensors. The data operand
and its numerical domain remain unconstrained by this parameter-only assertion. -/
def CalibrationCorrect (types : List (ValueId × Ty)) (facts : List Fact) (op : Op) : Prop :=
  ∀ env, BindingFacts.Valid model facts env →
    ∃ c dataId scaleId zeroId scale zero,
      Rules.W8A8SourceContract.inspect types op = some c ∧ op.operands = [dataId, scaleId, zeroId] ∧
      env.get? scaleId = some (.tensor scale) ∧ env.get? zeroId = some (.tensor zero) ∧
      scale.shape = [] ∧ zero.shape = [] ∧
      Dequant.tensorTyped scale (.float .f32) = true ∧ Dequant.tensorTyped zero (.int c.zeroWidth) = true ∧
      ∃ s z, Dequant.float32 (scale.get []) = some s ∧ Dequant.bits c.zeroWidth (zero.get []) = some z ∧
        Rules.W8A8SourceQuantize.scaleAdmissible s = true

def checkParameters (types : List (ValueId × Ty)) (facts : List Fact) (op : Op) : Bool :=
  match Rules.W8A8SourceContract.inspect types op, op.operands with
  | some c, [_, scaleId, zeroId] =>
    match value? facts scaleId, value? facts zeroId with
    | some (.tensor scale), some (.tensor zero) =>
      match Dequant.float32 (scale.get []), Dequant.bits c.zeroWidth (zero.get []) with
      | some s, some _ => decide (scale.shape = [] ∧ zero.shape = []) &&
          Dequant.tensorTyped scale (.float .f32) && Dequant.tensorTyped zero (.int c.zeroWidth) &&
          Rules.W8A8SourceQuantize.scaleAdmissible s
      | _, _ => false
    | _, _ => false
  | _, _ => false

theorem checkParameters_sound {types : List (ValueId × Ty)} {facts : List Fact} {op : Op}
    (checked : checkParameters types facts op = true) : CalibrationCorrect types facts op := by
  intro env valid
  unfold checkParameters at checked
  split at checked
  · next c dataId scaleId zeroId inspected operands =>
    split at checked
    · next scale zero scaleBound zeroBound =>
      split at checked
      · next s z scaleDecoded zeroDecoded =>
        simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
        obtain ⟨⟨⟨⟨scaleShape, zeroShape⟩, scaleTyped⟩, zeroTyped⟩, admissible⟩ := checked
        exact ⟨c, dataId, scaleId, zeroId, scale, zero, inspected, operands,
          value?_sound valid scaleBound, value?_sound valid zeroBound, scaleShape, zeroShape,
          scaleTyped, zeroTyped, s, z, scaleDecoded, zeroDecoded, admissible⟩
      · cases checked
    · cases checked
  · cases checked

/-- Actual checked calibration producers suffice for typed, non-stuck quantization for every
float model and every typed data tensor of the recognized shape. Poison results are permitted
here: initialized results still require the separately checked numerical domain. -/
theorem calibrated_quantize_supported {ib : Nat} {types : List (ValueId × Ty)} {facts : List Fact} {op : Op}
    (checked : checkParameters types facts op = true) (named : op.name = "quant_ext.quantize_per_tensor")
    {env : Env} (valid : BindingFacts.Valid model facts env) :
    ∃ c dataId scaleId zeroId scale zero,
      Rules.W8A8SourceContract.inspect types op = some c ∧ op.operands = [dataId, scaleId, zeroId] ∧
      env.get? scaleId = some (.tensor scale) ∧ env.get? zeroId = some (.tensor zero) ∧
      ∀ (M : FloatModel) (input : Tensor), input.shape = c.shape → Dequant.tensorTyped input (.float .f32) = true →
        ∃ output, Rules.W8A8SourceContract.eval types op M [.tensor input, .tensor scale, .tensor zero] =
          .ok [.tensor output] ∧ RVal.HasTy ib (.tensor output) (.tensor c.shape (.int 8)) := by
  obtain ⟨c, dataId, scaleId, zeroId, scale, zero, accepted, operands, scaleBound, zeroBound,
    scaleShape, zeroShape, scaleTyped, zeroTyped, s, z, scaleDecoded, zeroDecoded, admissible⟩ :=
    checkParameters_sound checked env valid
  obtain ⟨_, _, eligible⟩ := Rules.W8A8SourceContract.inspect_sound accepted
  have kind : c.kind = .quantize := by
    have expectedName := eligible.1
    cases actualKind : c.kind <;> simp_all [Rules.W8A8SourceContract.name]
  have widthAllowed : c.zeroWidth = 32 ∨ c.zeroWidth = 64 := eligible.2.2.2.2.2.2.1
  refine ⟨c, dataId, scaleId, zeroId, scale, zero, accepted, operands, scaleBound, zeroBound, ?_⟩
  intro M input inputShape inputTyped
  have inputs : Rules.W8A8SourceQuantize.validInputs c.zeroWidth c.shape input scale zero = true := by
    simp only [Rules.W8A8SourceQuantize.validInputs, Bool.and_eq_true, Bool.or_eq_true,
      beq_iff_eq, decide_eq_true_eq]
    exact ⟨⟨⟨⟨widthAllowed, ⟨inputShape, scaleShape, zeroShape⟩⟩, inputTyped⟩, scaleTyped⟩, zeroTyped⟩
  refine ⟨Rules.W8A8SourceQuantize.result c.zeroWidth M c.shape input s z, ?_,
    Rules.W8A8SourceQuantize.result_typed ib c.zeroWidth M c.shape input s z⟩
  rw [Rules.W8A8SourceContract.eval_recognized accepted, Rules.W8A8SourceContract.run, kind]
  simp [Rules.W8A8SourceQuantize.run, inputs, scaleDecoded, zeroDecoded, admissible]

#print axioms splatLearn_exec
#print axioms learn_sound
#print axioms advance_sound
#print axioms checkParameters_sound
#print axioms calibrated_quantize_supported

end MerlinVerify.Check.PolicyW8A8SourceFacts
