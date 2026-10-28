/- Whole-function typed support with calibration facts, including source Q/DQ within nested
regions. Captures use the actual enclosing facts; block arguments invalidate shadowed facts.
Internal calls and initialized execution remain separate obligations. -/
import MerlinVerify.Check.PolicyW8A8SourceProgress
import MerlinVerify.Check.PolicyW8A8Dequant
import MerlinVerify.Check.Rulebook
import MerlinVerify.Check.CalibratedProgress
import MerlinVerify.Check.RegionDepth

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

def baseBook (ib : Nat) : CalibratedProgress.Book sourceSem ib where
  rule depth types facts op args results := checkOp ib depth types facts op args results
  sound depth types facts op args results checked ctx call env values width loop typed valid fetched inputs :=
    checkOp_sound (ib := ib) (depth := depth) (types := types) (facts := facts)
      (op := op) (argTys := args) (resTys := results) (ctx := ctx) (call := call)
      (env := env) (args := values) checked width loop typed valid fetched inputs

/-- Each enclosing region consumes a budget unit. Exhaustion uses the established ordinary
region checker, so it can refuse source operations without compromising soundness. -/
def nestedBook (ib : Nat) : Nat → CalibratedProgress.Book sourceSem ib
  | 0 => baseBook ib
  | budget + 1 => {
      rule depth types facts op args results :=
        if W8A8SourceSemantics.isSource op.name then
          (baseBook ib).rule depth types facts op args results
        else match (ordinaryBook ib).rule depth op args results with
          | none => false
          | some sigs => CalibratedProgress.checkRegions (nestedBook ib budget) types facts op.regions sigs
      sound depth types facts op args results checked ctx call env values width loop typed valid fetched inputs := by
        split at checked
        · exact (baseBook ib).sound depth types facts op args results checked
            ctx call env values width loop typed valid fetched inputs
        · split at checked
          · cases checked
          · next sigs accepted =>
            exact (ordinaryBook ib).sound depth op args results sigs accepted ctx call values
              (evalRegions sourceSem call env op.regions) width loop inputs
              (CalibratedProgress.checkRegions_sound (nestedBook ib budget)
                W8A8SourceSemantics.producers types facts op.regions sigs call env checked typed valid) }

/-- Structural depth is computed from the checked AST. It affects coverage, never the claim. -/
def checkOps (ib depth : Nat) (types : TEnv) (facts : List Fact) (ops : List Op)
    (yields : List VTy) : Bool :=
  CalibratedProgress.checkOps (nestedBook ib (RegionDepth.ops ops)) depth types facts ops yields

theorem checkOps_progress (ib depth : Nat) (types : TEnv) (facts : List Fact)
    (ops : List Op) (yields : List VTy) (ctx : Ctx) (call : CallSem) (env : Env)
    (checked : checkOps ib depth types facts ops yields = true)
    (typed : EnvTyped ib types env) (valid : BindingFacts.Valid model facts env)
    (width : ctx.indexBits = ib) (loop : depth ≤ ctx.loopIndex.length) :
    (evalOps sourceSem ctx call env ops).NotStuck ∧
      ∀ results, evalOps sourceSem ctx call env ops = .ok results → HasTys ib results yields :=
  CalibratedProgress.checkOps_sound (nestedBook ib (RegionDepth.ops ops))
    W8A8SourceSemantics.producers depth types facts ops yields ctx call env checked typed valid width loop

def checkFunction (ib : Nat) (m : Module) (fn : String) (args results : List VTy) : Bool :=
  CalibratedProgress.checkFunction (nestedBook ib (RegionDepth.ops m.ops)) m fn args results

theorem checkFunction_sound {ib : Nat} {m : Module} {fn : String} {argTys resTys : List VTy}
    (checked : checkFunction ib m fn argTys resTys = true) :
    FuncNotStuck sourceSem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys) m fn :=
  CalibratedProgress.checkFunction_sound (nestedBook ib (RegionDepth.ops m.ops))
    W8A8SourceSemantics.producers checked

#print axioms checkOp_sound
#print axioms checkOps_progress
#print axioms checkFunction_sound

end MerlinVerify.Check.PolicyW8A8Support
