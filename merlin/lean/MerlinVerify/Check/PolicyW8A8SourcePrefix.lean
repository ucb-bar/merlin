/- Calibration facts across complete executed prefixes. The traversal follows actual SSA
writes, including arbitrary intervening operations; success of a cached diagnostic alone is
never an execution premise. Generic semantics must preserve the two calibration producers. -/
import MerlinVerify.Check.PolicyW8A8SourceFacts

namespace MerlinVerify.Check.PolicyW8A8SourcePrefix

open MerlinVerify PolicyW8A8SourceFacts

open ConstantFacts (Fact model)

def advanceOps (ib : Nat) : List Fact → List Op → List Fact
  | facts, [] => facts
  | facts, op :: rest => advanceOps ib (advance ib facts op) rest

/-- Every fact used at a later source operation describes its actual current binding, for
every successful execution of the prefix under a producer-compatible interpretation. -/
theorem advanceOps_sound {ib : Nat} {ops : List Op} {facts : List Fact}
    {sem : OpSem} {ctx : Ctx} {call : CallSem} {env env' : Env}
    (agree : ProducerAgreement sem) (valid : BindingFacts.Valid model facts env)
    (width : ctx.indexBits = ib) (executed : execOps sem ctx call env ops = .ok env') :
    BindingFacts.Valid model (advanceOps ib facts ops) env' := by
  induction ops generalizing facts env with
  | nil =>
    cases executed
    exact valid
  | cons op rest ih =>
    cases terminal : op.isTerminator with
    | true => simp [execOps, terminal] at executed
    | false =>
      rw [execOps_cons_step sem ctx call env op rest terminal] at executed
      cases stepped : stepOp sem ctx call env op with
      | ub => simp [stepped, Outcome.bind] at executed
      | stuck reason => simp [stepped, Outcome.bind] at executed
      | ok next =>
        simp only [stepped, Outcome.bind_ok] at executed
        exact ih (advance_sound agree valid width stepped) executed

def advanceTypes : List (ValueId × Ty) → List Op → List (ValueId × Ty)
  | types, [] => types
  | types, op :: rest => advanceTypes (op.results.map (fun value => (value.id, value.ty)) ++ types) rest

def checkAt (ib : Nat) (types : List (ValueId × Ty)) (facts : List Fact)
    (ops : List Op) (site : Op) : Bool :=
  checkParameters (advanceTypes types ops) (advanceOps ib facts ops) site

/-- Check all source calibration sites in one traversal. Unsupported quant_ext operations
refuse rather than disappearing from the coverage denominator. Nested regions are separate. -/
def checkOps (ib : Nat) : List (ValueId × Ty) → List Fact → List Op → Bool
  | _, _, [] => true
  | types, facts, op :: rest =>
    (!op.name.startsWith "quant_ext." || checkParameters types facts op) &&
      checkOps ib (op.results.map (fun value => (value.id, value.ty)) ++ types)
        (advance ib facts op) rest

/-- Whole-segment acceptance supplies the same check at every selected site, including sites
after arbitrary operations. It does not certify that the intervening operations execute. -/
theorem checkOps_at {ib : Nat} {types : List (ValueId × Ty)} {facts : List Fact}
    {pre rest : List Op} {site : Op}
    (checked : checkOps ib types facts (pre ++ site :: rest) = true)
    (selected : site.name.startsWith "quant_ext." = true) :
    checkAt ib types facts pre site = true := by
  induction pre generalizing types facts with
  | nil =>
    simp only [List.nil_append, checkOps, selected, Bool.not_true, Bool.false_or,
      Bool.and_eq_true] at checked
    exact checked.1
  | cons op pre ih =>
    simp only [List.cons_append, checkOps, Bool.and_eq_true] at checked
    exact ih checked.2

/-- Static prefix traversal plus a successful actual execution establishes the calibration
contract at the selected operation. The input data's numerical domain remains separate. -/
theorem checkAt_sound {ib : Nat} {types : List (ValueId × Ty)} {facts : List Fact}
    {ops : List Op} {site : Op} {sem : OpSem} {ctx : Ctx} {call : CallSem} {env env' : Env}
    (agree : ProducerAgreement sem) (valid : BindingFacts.Valid model facts env)
    (width : ctx.indexBits = ib) (executed : execOps sem ctx call env ops = .ok env')
    (checked : checkAt ib types facts ops site = true) :
    ∃ c dataId scaleId zeroId scale zero,
      Rules.W8A8SourceContract.inspect (advanceTypes types ops) site = some c ∧
      site.operands = [dataId, scaleId, zeroId] ∧
      env'.get? scaleId = some (.tensor scale) ∧ env'.get? zeroId = some (.tensor zero) ∧
      scale.shape = [] ∧ zero.shape = [] ∧
      Sem.Ops.Dequant.tensorTyped scale (.float .f32) = true ∧
      Sem.Ops.Dequant.tensorTyped zero (.int c.zeroWidth) = true ∧
      ∃ s z, Sem.Ops.Dequant.float32 (scale.get []) = some s ∧
        Sem.Ops.Dequant.bits c.zeroWidth (zero.get []) = some z ∧
        Rules.W8A8SourceQuantize.scaleAdmissible s = true :=
  checkParameters_sound checked env' (advanceOps_sound agree valid width executed)

/-- A fact-free entry is valid for every environment; no caller-provided calibration values
or declarations can masquerade as learned producer facts. -/
theorem empty_valid (env : Env) : BindingFacts.Valid model [] env := by
  intro fact member
  cases member

#print axioms advanceOps_sound
#print axioms checkOps_at
#print axioms checkAt_sound

end MerlinVerify.Check.PolicyW8A8SourcePrefix
