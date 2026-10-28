/- Bounded negation and pairwise addition for the named attention numerical model.
Source keys and live input facts are checked; arbitrary fallback negation is not assumed. -/
import MerlinVerify.Check.PolicyW8A8TrigScalar
import MerlinVerify.Rules.W8A8PairBounds

namespace MerlinVerify.Check.PolicyW8A8AttentionScalar

open MerlinVerify Rules Definedness Sem.Ops PolicyW8A8RotaryFacts

local notation "sourceSem" => W8A8SourceSemantics.opSem
local notation "allowed" => PolicyW8A8TrigScalar.allowed

theorem negation_exec (op : Op) (modelled : unmodelledAttr? op.blind = none)
    (dialect : dialectOf op.name = "arith") (named : op.name = "arith.negf") (regions : op.regions = [])
    (result : Arith.resultElt op.blind = some (.float .f32))
    (key : Arith.floatKey op.blind = Spec.Dequant.defaultKey op.name)
    (ctx : Ctx) (call : CallSem) (env : Env) (bits : FBits .f32) :
    sourceSem ctx call op [.scalar (.some (.flt .f32 bits))]
      (evalRegions sourceSem call env op.regions) =
      .ok [.scalar (.some (.flt .f32 (ctx.M.op (Spec.Dequant.defaultKey "arith.negf") .f32 [bits])))] := by
  have ordinary : W8A8SourceSemantics.isSource op.name = false := by
    simp [W8A8SourceSemantics.isSource, W8A8SourceContract.name, named]
  rw [regions, evalRegions, W8A8SourceSemantics.unchanged ordinary, opSem_modelled modelled]
  simp only [dispatch, Op.blind_name, dialect]
  simp [Arith.sem, Op.blind_name, named, result, Arith.unOp, Arith.pureN, Outcome.bindOp_eq,
    Arith.fltN, liftEager, key]

def negationRule (_ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "arith" ∧ op.name = "arith.negf" ∧
      op.regions = [] ∧ Arith.floatKey op.blind = Spec.Dequant.defaultKey op.name then
    match facts, Arith.resultElt op.blind with
    | [⟨⟨.scalar (.float .f32), true⟩, input⟩], some (.float .f32) =>
      (upper input).map fun cap => [PolicyW8A8RotaryScalar.fact cap]
    | _, _ => none
  else none

theorem negationRule_sound (ib : Nat) : ∀ Γ op facts outputs,
    negationRule Γ op facts = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds model ib Γ env →
    Refined.FactsHold model ib args facts → ∃ values,
      sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok values ∧
      Refined.FactsHold model ib values outputs := by
  intro Γ op facts outputs checked ctx call env args _width permitted _valid holds
  unfold negationRule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named, regions, key⟩ := header
    split at checked
    · next input resultType =>
      cases bounded : upper input with
      | none => simp [bounded] at checked
      | some cap =>
        simp only [bounded, Option.map_some, Option.some.injEq] at checked
        subst outputs
        obtain _ | ⟨head, tail⟩ := holds
        cases tail
        obtain ⟨bits, rfl⟩ := initialized_float head.1
        have numerical := W8A8UnarySign.bounded (upper_sound bounded head)
        have execution := negation_exec op modelled dialect named regions resultType key ctx call env bits
        obtain ⟨base, interpretation⟩ := permitted
        rw [interpretation, W8A8RoundedReal.model_negation] at execution
        exact ⟨_, execution, .cons (PolicyW8A8RotaryScalar.result numerical) .nil⟩
    · cases checked
  · cases checked

def addRule (_ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "arith" ∧ op.name = "arith.addf" ∧
      op.regions = [] ∧ Arith.floatKey op.blind = Spec.Dequant.defaultKey op.name then
    match facts, Arith.resultElt op.blind with
    | [⟨⟨.scalar (.float .f32), true⟩, left⟩, ⟨⟨.scalar (.float .f32), true⟩, right⟩], some (.float .f32) => do
      let leftCap ← upper left
      let rightCap ← upper right
      if max leftCap rightCap ≤ 252 then some [PolicyW8A8RotaryScalar.fact (W8A8PairBounds.addCap leftCap rightCap)]
      else none
    | _, _ => none
  else none

theorem addRule_sound (ib : Nat) : ∀ Γ op facts outputs,
    addRule Γ op facts = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds model ib Γ env →
    Refined.FactsHold model ib args facts → ∃ values,
      sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok values ∧
      Refined.FactsHold model ib values outputs := by
  intro Γ op facts outputs checked ctx call env args _width permitted _valid holds
  unfold addRule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named, regions, key⟩ := header
    split at checked
    · next left right resultType =>
      cases leftFound : upper left with
      | none => simp [leftFound] at checked
      | some leftCap =>
        cases rightFound : upper right with
        | none => simp [leftFound, rightFound] at checked
        | some rightCap =>
          simp only [leftFound, rightFound, bind, Option.bind] at checked
          split at checked
          · next safe =>
            cases checked
            obtain _ | ⟨leftHolds, tail⟩ := holds
            obtain _ | ⟨rightHolds, tail⟩ := tail
            cases tail
            obtain ⟨x, rfl⟩ := initialized_float leftHolds.1
            obtain ⟨y, rfl⟩ := initialized_float rightHolds.1
            have bounded := W8A8PairBounds.add_bounded (upper_sound leftFound leftHolds)
              (upper_sound rightFound rightHolds) safe
            have execution := PolicyW8A8NumericalOps.binary_exec op .add modelled dialect named regions
              resultType key ctx call env x y
            obtain ⟨base, interpretation⟩ := permitted
            have addition : (W8A8RoundedReal.model base).op (Spec.Dequant.defaultKey "arith.addf") .f32 [x, y] =
                W8A8IEEE32.add x y := by
              simp [W8A8RoundedReal.model, W8A8ByteModel.extend, W8A8IEEEPrimitives.model,
                W8A8IEEEPrimitives.key, Spec.Dequant.defaultKey]
            simp only [PolicyW8A8NumericalScalar.BinaryKind.name] at execution
            rw [interpretation, addition] at execution
            exact ⟨_, execution, .cons (PolicyW8A8RotaryScalar.result bounded) .nil⟩
          · cases checked
    · cases checked
  · cases checked

def book (ib : Nat) : Contextual.Book model sourceSem ib allowed :=
  ({ rule := negationRule, sound := negationRule_sound ib } : Contextual.Book model sourceSem ib allowed).orElse
    { rule := addRule, sound := addRule_sound ib }

end MerlinVerify.Check.PolicyW8A8AttentionScalar
