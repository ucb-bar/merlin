/- Whole-function typed support with calibration facts. Ordinary nested regions reuse the
existing progress framework under the explicit source interpretation. Source Q/DQ is supported
at the current block level; source operations inside nested regions currently refuse. -/
import MerlinVerify.Check.PolicyW8A8SourceProgress
import MerlinVerify.Check.PolicyW8A8Dequant
import MerlinVerify.Check.Rulebook

namespace MerlinVerify.Check.PolicyW8A8Support

open MerlinVerify Rules
open ConstantFacts (Fact model)

local notation "sourceSem" => W8A8SourceSemantics.opSem

def ordinaryBook (ib : Nat) : Rulebook sourceSem ib where
  rule depth op args results :=
    if W8A8SourceSemantics.isSource op.name then none else (merlinBook ib).rule depth op args results
  sound := by
    intro depth op args results sigs checked
    split at checked
    · cases checked
    · next ordinary =>
      intro ctx call values regions width loop typed safe
      rw [W8A8SourceSemantics.unchanged (by simpa using ordinary)]
      exact (merlinBook ib).sound depth op args results sigs checked ctx call values regions width loop typed safe

def checkOp (ib depth : Nat) (types : TEnv) (facts : List Fact) (op : Op)
    (args results : List VTy) : Bool :=
  if W8A8SourceSemantics.isSource op.name then
    PolicyW8A8Dequant.check op args results || PolicyW8A8SourceProgress.check facts op args results
  else match (ordinaryBook ib).rule depth op args results with
    | none => false
    | some sigs => checkRegions (ordinaryBook ib) types op.regions sigs

theorem checkOp_sound {ib depth : Nat} {types : TEnv} {facts : List Fact} {op : Op}
    {argTys resTys : List VTy} (checked : checkOp ib depth types facts op argTys resTys = true)
    {ctx : Ctx} {call : CallSem} {env : Env} {args : List RVal}
    (width : ctx.indexBits = ib) (loop : depth ≤ ctx.loopIndex.length)
    (typedEnv : EnvTyped ib types env) (valid : BindingFacts.Valid model facts env)
    (fetched : env.getAll op.operands = .ok args) (typed : HasTys ib args argTys) :
    (sourceSem ctx call op args (evalRegions sourceSem call env op.regions)).NotStuck ∧
      ∀ results, sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok results →
        HasTys ib results resTys := by
  unfold checkOp at checked
  split at checked
  · have total : ∃ results, sourceSem ctx call op args
        (evalRegions sourceSem call env op.regions) = .ok results ∧ HasTys ib results resTys := by
      simp only [Bool.or_eq_true] at checked
      rcases checked with dequant | calibrated
      · obtain ⟨results, executed, outputTyped, _⟩ := PolicyW8A8Dequant.check_sound dequant ctx call env typed
        exact ⟨results, executed, outputTyped⟩
      · exact PolicyW8A8SourceProgress.check_sound calibrated ctx call env args valid fetched typed
    obtain ⟨results, executed, outputTyped⟩ := total
    rw [executed]
    exact ⟨trivial, fun _ same => by cases same; exact outputTyped⟩
  · split at checked
    · cases checked
    · next sigs accepted =>
      exact (ordinaryBook ib).sound depth op argTys resTys sigs accepted ctx call args
        (evalRegions sourceSem call env op.regions) width loop typed
        (checkRegions_safe (ordinaryBook ib) types op.regions sigs call env checked typedEnv)

def checkOps (ib depth : Nat) : TEnv → List Fact → List Op → List VTy → Bool
  | _, _, [], _ => false
  | types, facts, op :: rest, yields =>
    match op.operands.mapM (fun id => types.lookup id) with
    | none => false
    | some argTys =>
      if op.isTerminator then rest.isEmpty && argTys == yields
      else match op.results.mapM (fun result => vtyOf ib result.ty) with
        | none => false
        | some resTys => checkOp ib depth types facts op argTys resTys &&
          checkOps ib depth ((op.results.map (·.id)).zip resTys ++ types)
            (PolicyW8A8SourceFacts.advance ib facts op) rest yields

theorem checkOps_progress (ib depth : Nat) :
    ∀ (types : TEnv) (facts : List Fact) (ops : List Op) (yields : List VTy)
      (ctx : Ctx) (call : CallSem) (env : Env),
      checkOps ib depth types facts ops yields = true → EnvTyped ib types env →
      BindingFacts.Valid model facts env → ctx.indexBits = ib → depth ≤ ctx.loopIndex.length →
      (evalOps sourceSem ctx call env ops).NotStuck ∧
        ∀ results, evalOps sourceSem ctx call env ops = .ok results → HasTys ib results yields
  | _, _, [], _, _, _, _, checked, _, _, _, _ => by simp [checkOps] at checked
  | types, facts, (.mk name outputs inputs attrs props regions successors) :: rest,
      yields, ctx, call, env, checked, typedEnv, valid, width, loop => by
    let op := Op.mk name outputs inputs attrs props regions successors
    simp only [checkOps] at checked
    split at checked
    · cases checked
    · next argTys argTypes =>
      obtain ⟨args, fetched, typed⟩ := typedEnv.getAll inputs argTys argTypes
      rw [evalOps.eq_2]
      split
      · next terminal =>
        rw [ite_eq_left terminal] at checked
        simp only [Bool.and_eq_true, List.isEmpty_iff, beq_iff_eq] at checked
        obtain ⟨rfl, rfl⟩ := checked
        simp only [List.isEmpty_nil, ↓reduceIte, fetched]
        exact ⟨trivial, fun _ same => by cases same; exact typed⟩
      · next ordinary =>
        rw [ite_eq_right ordinary] at checked
        split at checked
        · cases checked
        · next resTys resultTypes =>
          simp only [Bool.and_eq_true] at checked
          obtain ⟨headChecked, restChecked⟩ := checked
          have safe := checkOp_sound (call := call) headChecked width loop typedEnv valid fetched typed
          rw [fetched, Outcome.bind_ok]
          cases executed : sourceSem ctx call op args (evalRegions sourceSem call env regions) with
          | ub => exact ⟨trivial, fun _ impossible => by cases impossible⟩
          | stuck reason => exact absurd (executed ▸ safe.1) (by simp [Outcome.NotStuck])
          | ok values =>
            have valuesTyped := safe.2 values executed
            have length : outputs.length = values.length := by
              exact (optionMapM_length _ _ _ resultTypes).symm.trans (HasTys.length valuesTyped).symm
            have bound := bindAll_ok (env := env) (vs := outputs) length
            have step : stepOp sourceSem ctx call env op = .ok ((outputs.map (·.id)).zip values ++ env) := by
              unfold stepOp
              simp only [op, Op.operands, Op.results, Op.regions]
              rw [fetched, Outcome.bind_ok, executed, Outcome.bind_ok]
              exact bound
            simp only [Outcome.bind_ok, Op.results]
            rw [bound, Outcome.bind_ok]
            exact checkOps_progress ib depth _ _ rest yields ctx call _ restChecked
              (typedEnv.extend _ values resTys valuesTyped)
              (PolicyW8A8SourceFacts.advance_sound W8A8SourceSemantics.producers valid width step) width loop

def checkFunction (ib : Nat) (m : Module) (fn : String) (argTys resTys : List VTy) : Bool :=
  match m.func? fn with
  | none => false
  | some function => !function.isDecl &&
    match function.body with
    | .mk [.mk inputs ops] =>
      inputs.mapM (fun input => vtyOf ib input.ty) == some argTys &&
        checkOps ib 0 ((inputs.map (·.id)).zip argTys) [] ops resTys
    | _ => false

theorem checkFunction_sound {ib : Nat} {m : Module} {fn : String} {argTys resTys : List VTy}
    (checked : checkFunction ib m fn argTys resTys = true) :
    FuncNotStuck sourceSem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys) m fn := by
  intro ctx width args typed
  unfold checkFunction at checked
  cases found : m.func? fn with
  | none => simp [found] at checked
  | some function =>
    simp only [found, Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at checked
    obtain ⟨defined, checked⟩ := checked
    split at checked
    · next inputs ops body =>
      simp only [Bool.and_eq_true, beq_iff_eq] at checked
      obtain ⟨inputTypes, operations⟩ := checked
      have length : inputs.length = args.length := by
        rw [← optionMapM_length _ _ _ inputTypes]
        exact (HasTys.length typed).symm
      have empty : EnvTyped ib [] [] := by intro _ _ missing; cases missing
      simp only [denote, evalFunc, found, defined, Bool.false_eq_true, ↓reduceIte,
        body, evalRegion, evalBlocks, evalBlock]
      rw [bindAll_ok length, Outcome.bind_ok]
      have entry : EnvTyped ib ((inputs.map (·.id)).zip argTys)
          ((inputs.map (·.id)).zip args ++ []) := by
        simpa using empty.extend (inputs.map (·.id)) args argTys typed
      exact (checkOps_progress ib 0 _ [] ops resTys ctx (evalFunc sourceSem ctx m m.funcs.length) _ operations
        entry (PolicyW8A8SourcePrefix.empty_valid _) width (Nat.zero_le _)).1
    · cases checked

#print axioms checkOp_sound
#print axioms checkOps_progress
#print axioms checkFunction_sound

end MerlinVerify.Check.PolicyW8A8Support
