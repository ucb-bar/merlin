/- Actual-key scalar sin/cos checks for the separately named rounded-real model.
Unsupported keys, nonfinite inputs and absent numerical facts refuse. -/
import MerlinVerify.Rules.W8A8TranscendentalModel
import MerlinVerify.Check.PolicyW8A8OrderedPrefix

namespace MerlinVerify.Check.PolicyW8A8TrigScalar

open MerlinVerify.Rules.W8A8RoundedReal
open MerlinVerify Rules Check Definedness Sem.Ops PolicyW8A8RotaryFacts

inductive Kind where
  | sine
  | cosine
  deriving DecidableEq

def Kind.name : Kind → String
  | .sine => "math.sin"
  | .cosine => "math.cos"

noncomputable def Kind.apply : Kind → FBits .f32 → FBits .f32
  | .sine => MerlinVerify.Rules.W8A8RoundedReal.sine
  | .cosine => MerlinVerify.Rules.W8A8RoundedReal.cosine

theorem Kind.bounded (kind : Kind) {bits : FBits .f32} (finite : W8A8IEEE32.Finite bits) :
    W8A8IEEEBounds.ExponentLE 127 (kind.apply bits) := by
  cases kind
  · exact sine_cap finite
  · exact cosine_cap finite

def select (name : String) : Option Kind :=
  if name = "math.sin" then some .sine else if name = "math.cos" then some .cosine else none

theorem select_named {name : String} {kind : Kind} (selected : select name = some kind) :
    name = kind.name := by
  unfold select at selected
  split at selected
  · cases selected
    assumption
  · split at selected
    · cases selected
      assumption
    · cases selected

def allowed (ctx : Ctx) (_ : CallSem) : Prop :=
  ∃ base, ctx.M = MerlinVerify.Rules.W8A8RoundedReal.model base

theorem original_allowed {ctx : Ctx} {call : CallSem} (permitted : allowed ctx call) :
    PolicyW8A8OrderedRule.allowed ctx call := by
  obtain ⟨base, interpretation⟩ := permitted
  exact ⟨transcendentalFallback (W8A8UnarySign.model base), interpretation⟩

theorem unary_exec (op : Op) (kind : Kind) (modelled : unmodelledAttr? op.blind = none)
    (dialect : dialectOf op.name = "math") (named : op.name = kind.name) (regions : op.regions = [])
    (result : Arith.resultElt op.blind = some (.float .f32))
    (key : Arith.floatKey op.blind = Spec.Dequant.defaultKey op.name)
    (ctx : Ctx) (call : CallSem) (env : Env) (bits : FBits .f32) :
    W8A8SourceSemantics.opSem ctx call op [.scalar (.some (.flt .f32 bits))]
      (evalRegions W8A8SourceSemantics.opSem call env op.regions) =
      .ok [.scalar (.some (.flt .f32 (ctx.M.op (Spec.Dequant.defaultKey kind.name) .f32 [bits])))] := by
  have ordinary : W8A8SourceSemantics.isSource op.name = false := by
    cases kind <;> simp [Kind.name] at named <;>
      simp [W8A8SourceSemantics.isSource, W8A8SourceContract.name, named]
  rw [regions, evalRegions, W8A8SourceSemantics.unchanged ordinary, opSem_modelled modelled]
  simp only [dispatch, Op.blind_name, dialect]
  cases kind <;> simp only [Kind.name] at named ⊢
  all_goals simp [Arith.sem, Op.blind_name, named, result, Arith.pureN, Outcome.bindOp_eq,
    Arith.fltN, liftEager, key]

def rule (_ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "math" ∧ op.regions = [] ∧
      op.successors = [] ∧ Arith.floatKey op.blind = Spec.Dequant.defaultKey op.name then
    match facts, Arith.resultElt op.blind, select op.name with
    | [⟨⟨.scalar (.float .f32), true⟩, input⟩], some (.float .f32), some _ =>
      match upper input with
      | some _ => some [PolicyW8A8RotaryScalar.fact 127]
      | none => none
    | _, _, _ => none
  else none

theorem rule_sound (ib : Nat) : ∀ Γ op facts outputs,
    rule Γ op facts = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds PolicyW8A8RotaryFacts.model ib Γ env →
    Refined.FactsHold PolicyW8A8RotaryFacts.model ib args facts → ∃ values,
      W8A8SourceSemantics.opSem ctx call op args
        (evalRegions W8A8SourceSemantics.opSem call env op.regions) = .ok values ∧
      Refined.FactsHold PolicyW8A8RotaryFacts.model ib values outputs := by
  intro Γ op facts outputs checked ctx call env args _width permitted _valid holds
  unfold rule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, regions, _successors, key⟩ := header
    split at checked
    · next input kind resultType selected =>
      cases bounded : upper input with
      | none => simp [bounded] at checked
      | some cap =>
        simp only [bounded, Option.some.injEq] at checked
        subst outputs
        obtain _ | ⟨head, tail⟩ := holds
        cases tail
        obtain ⟨bits, rfl⟩ := initialized_float head.1
        have finite := (upper_sound bounded head).1
        have numerical := kind.bounded finite
        have execution := unary_exec op kind modelled dialect (select_named selected) regions resultType key
          ctx call env bits
        obtain ⟨base, interpretation⟩ := permitted
        rw [interpretation] at execution
        have actual : (MerlinVerify.Rules.W8A8RoundedReal.model base).op (Spec.Dequant.defaultKey kind.name) .f32 [bits] =
            kind.apply bits := by
          cases kind
          · exact model_sine base bits
          · exact model_cosine base bits
        rw [actual] at execution
        exact ⟨_, execution, .cons (PolicyW8A8RotaryScalar.result numerical) .nil⟩
    · cases checked
  · cases checked

def book (ib : Nat) : Contextual.Book PolicyW8A8RotaryFacts.model W8A8SourceSemantics.opSem ib allowed :=
  ({ rule := rule, sound := rule_sound ib } :
    Contextual.Book PolicyW8A8RotaryFacts.model W8A8SourceSemantics.opSem ib allowed).orElse
      ((PolicyW8A8OrderedPrefix.book ib).restrict (fun _ _ permitted => original_allowed permitted))

end MerlinVerify.Check.PolicyW8A8TrigScalar
