/- Fact-aware progress, reusable by calibrated source interpretations. Every accepted operation
must preserve types; literal/splat facts follow actual SSA execution and are invalidated on writes.
Region block arguments shadow captures in both the type and calibration environments. -/
import MerlinVerify.Check.PolicyW8A8SourceFacts
import MerlinVerify.Check.Progress

namespace MerlinVerify.Check.CalibratedProgress

open MerlinVerify
open ConstantFacts (Fact model)

structure Book (sem : OpSem) (ib : Nat) where
  rule : Nat → TEnv → List Fact → Op → List VTy → List VTy → Bool
  sound : ∀ depth types facts op argTys resTys,
    rule depth types facts op argTys resTys = true →
    ∀ ctx call env args, ctx.indexBits = ib → depth ≤ ctx.loopIndex.length →
      EnvTyped ib types env → BindingFacts.Valid model facts env →
      env.getAll op.operands = .ok args → HasTys ib args argTys →
      (sem ctx call op args (evalRegions sem call env op.regions)).NotStuck ∧
        ∀ results, sem ctx call op args (evalRegions sem call env op.regions) = .ok results →
          HasTys ib results resTys

variable {sem : OpSem} {ib : Nat}

def checkOps (R : Book sem ib) (depth : Nat) : TEnv → List Fact → List Op → List VTy → Bool
  | _, _, [], _ => false
  | types, facts, op :: rest, yields =>
    match op.operands.mapM (fun id => types.lookup id) with
    | none => false
    | some argTys =>
      if op.isTerminator then rest.isEmpty && argTys == yields
      else match op.results.mapM (fun result => vtyOf ib result.ty) with
        | none => false
        | some resTys => R.rule depth types facts op argTys resTys &&
          checkOps R depth ((op.results.map (·.id)).zip resTys ++ types)
            (PolicyW8A8SourceFacts.advance ib facts op) rest yields

theorem checkOps_sound (R : Book sem ib) (producers : PolicyW8A8SourceFacts.ProducerAgreement sem)
    (depth : Nat) :
    ∀ (types : TEnv) (facts : List Fact) (ops : List Op) (yields : List VTy)
      (ctx : Ctx) (call : CallSem) (env : Env),
      checkOps R depth types facts ops yields = true → EnvTyped ib types env →
      BindingFacts.Valid model facts env → ctx.indexBits = ib → depth ≤ ctx.loopIndex.length →
      (evalOps sem ctx call env ops).NotStuck ∧
        ∀ results, evalOps sem ctx call env ops = .ok results → HasTys ib results yields
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
          have safe := R.sound depth types facts op argTys resTys headChecked
            ctx call env args width loop typedEnv valid fetched typed
          rw [fetched, Outcome.bind_ok]
          cases executed : sem ctx call op args (evalRegions sem call env regions) with
          | ub => exact ⟨trivial, fun _ impossible => by cases impossible⟩
          | stuck reason => exact absurd (executed ▸ safe.1) (by simp [Outcome.NotStuck])
          | ok values =>
            have valuesTyped := safe.2 values executed
            have length : outputs.length = values.length := by
              exact (optionMapM_length _ _ _ resultTypes).symm.trans (HasTys.length valuesTyped).symm
            have bound := bindAll_ok (env := env) (vs := outputs) length
            have step : stepOp sem ctx call env op = .ok ((outputs.map (·.id)).zip values ++ env) := by
              unfold stepOp
              simp only [op, Op.operands, Op.results, Op.regions]
              rw [fetched, Outcome.bind_ok, executed, Outcome.bind_ok]
              exact bound
            simp only [Outcome.bind_ok, Op.results]
            rw [bound, Outcome.bind_ok]
            exact checkOps_sound R producers depth _ _ rest yields ctx call _ restChecked
              (typedEnv.extend _ values resTys valuesTyped)
              (PolicyW8A8SourceFacts.advance_sound producers valid width step) width loop

def checkRegion (R : Book sem ib) (types : TEnv) (facts : List Fact)
    (region : Region) (sig : RegionSig) : Bool :=
  match region with
  | .mk [.mk inputs ops] =>
    inputs.mapM (fun input => vtyOf ib input.ty) == some sig.args &&
      checkOps R sig.depth ((inputs.map (·.id)).zip sig.args ++ types)
        (BindingFacts.forget model (inputs.map (·.id)) facts) ops sig.yields
  | _ => false

theorem checkRegion_sound (R : Book sem ib)
    (producers : PolicyW8A8SourceFacts.ProducerAgreement sem)
    {types : TEnv} {facts : List Fact} {region : Region} {sig : RegionSig}
    (checked : checkRegion R types facts region sig = true) (call : CallSem) (env : Env)
    (typedEnv : EnvTyped ib types env) (valid : BindingFacts.Valid model facts env) :
    RegionSafe ib (fun ctx => evalRegion sem ctx call env region) sig := by
  cases region with
  | mk blocks =>
    cases blocks with
    | nil => cases checked
    | cons block rest =>
      cases rest with
      | cons _ _ => cases block; cases checked
      | nil =>
        cases block with
        | mk inputs ops =>
          simp only [checkRegion, Bool.and_eq_true, beq_iff_eq] at checked
          obtain ⟨inputTypes, operations⟩ := checked
          intro ctx width loop args typed
          have length : inputs.length = args.length := by
            rw [← optionMapM_length _ _ _ inputTypes]
            exact (HasTys.length typed).symm
          simp only [evalRegion, evalBlocks, evalBlock]
          rw [bindAll_ok length, Outcome.bind_ok]
          exact checkOps_sound R producers sig.depth _ _ ops sig.yields ctx call _ operations
            (typedEnv.extend _ args sig.args typed)
            (BindingFacts.forget_sound model valid _ args) width loop

def checkRegions (R : Book sem ib) (types : TEnv) (facts : List Fact) :
    List Region → List RegionSig → Bool
  | [], [] => true
  | region :: rest, sig :: sigs =>
    checkRegion R types facts region sig && checkRegions R types facts rest sigs
  | _, _ => false

theorem checkRegions_sound (R : Book sem ib)
    (producers : PolicyW8A8SourceFacts.ProducerAgreement sem) :
    ∀ types facts regions sigs call env,
      checkRegions R types facts regions sigs = true → EnvTyped ib types env →
      BindingFacts.Valid model facts env →
      List.Forall₂ (RegionSafe ib) (evalRegions sem call env regions) sigs
  | _, _, [], [], _, _, _, _, _ => .nil
  | types, facts, region :: rest, sig :: sigs, call, env, checked, typedEnv, valid => by
    simp only [checkRegions, Bool.and_eq_true] at checked
    exact .cons (checkRegion_sound R producers checked.1 call env typedEnv valid)
      (checkRegions_sound R producers types facts rest sigs call env checked.2 typedEnv valid)
  | _, _, [], _ :: _, _, _, checked, _, _ => by cases checked
  | _, _, _ :: _, [], _, _, checked, _, _ => by cases checked

def checkFunction (R : Book sem ib) (m : Module) (fn : String) (args results : List VTy) : Bool :=
  match m.func? fn with
  | none => false
  | some function => !function.isDecl && checkRegion R [] [] function.body ⟨args, results, 0⟩

theorem checkFunction_sound (R : Book sem ib)
    (producers : PolicyW8A8SourceFacts.ProducerAgreement sem)
    {m : Module} {fn : String} {argTys resTys : List VTy}
    (checked : checkFunction R m fn argTys resTys = true) :
    FuncNotStuck sem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys) m fn := by
  intro ctx width args typed
  unfold checkFunction at checked
  cases found : m.func? fn with
  | none => simp [found] at checked
  | some function =>
    simp only [found, Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at checked
    obtain ⟨defined, checked⟩ := checked
    have emptyTypes : EnvTyped ib [] [] := by intro _ _ missing; cases missing
    have emptyFacts : BindingFacts.Valid model [] [] := by intro _ member; cases member
    have safe := checkRegion_sound R producers checked (evalFunc sem ctx m m.funcs.length)
      [] emptyTypes emptyFacts ctx width (Nat.zero_le _) args typed
    simpa only [denote, evalFunc, found, defined, Bool.false_eq_true, ↓reduceIte] using safe.1

#print axioms checkOps_sound
#print axioms checkRegion_sound
#print axioms checkRegions_sound
#print axioms checkFunction_sound

end MerlinVerify.Check.CalibratedProgress
