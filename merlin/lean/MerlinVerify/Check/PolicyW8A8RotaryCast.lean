/- Source-derived nonnegative integer intervals become checked f32 exponent
bounds through the exact signed conversion. Tables refuse oversized intervals. -/
import MerlinVerify.Check.PolicyW8A8RotaryScalar

namespace MerlinVerify.Check.PolicyW8A8RotaryCast

open MerlinVerify Rules Definedness Sem.Ops PolicyW8A8RotaryFacts

local notation "sourceSem" => W8A8SourceSemantics.opSem
local notation "allowed" => PolicyW8A8NumericalContext.allowed

def rule (_ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "arith" ∧ op.name = "arith.sitofp" ∧
      op.regions = [] ∧ Arith.floatKey op.blind = Spec.Dequant.defaultKey op.name then
    match facts, Arith.resultElt op.blind with
    | [⟨⟨.scalar (.int width), true⟩, input⟩], some (.float .f32) => do
      let range ← input.prior.prior.indices
      if range.upper ≤ 4096 then
        let cap := W8A8IntegerFloatBounds.cap width range
        if W8A8IntegerFloatBounds.check width range cap then some [PolicyW8A8RotaryScalar.fact cap] else none
      else none
    | _, _ => none
  else none

theorem cast_exec (op : Op) (modelled : unmodelledAttr? op.blind = none)
    (dialect : dialectOf op.name = "arith") (named : op.name = "arith.sitofp") (regions : op.regions = [])
    (result : Arith.resultElt op.blind = some (.float .f32))
    (key : Arith.floatKey op.blind = Spec.Dequant.defaultKey op.name)
    (ctx : Ctx) (call : CallSem) (env : Env) {width : Nat} (bits : BitVec width) :
    sourceSem ctx call op [.scalar (.some (.bits width bits))] (evalRegions sourceSem call env op.regions) =
      .ok [.scalar (.some (.flt .f32 (ctx.M.ofInt (Spec.Dequant.defaultKey "arith.sitofp") width .f32 bits)))] := by
  have ordinary : W8A8SourceSemantics.isSource op.name = false := by
    simp [W8A8SourceSemantics.isSource, W8A8SourceContract.name, named]
  rw [regions, evalRegions, W8A8SourceSemantics.unchanged ordinary, opSem_modelled modelled]
  simp only [dispatch, Op.blind_name, dialect]
  simp [Arith.sem, named, result, Arith.unOp, Arith.pureN,
    liftEager, key]

theorem rule_sound (ib : Nat) : ∀ Γ op facts outputs,
    rule Γ op facts = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds model ib Γ env →
    Refined.FactsHold model ib args facts → ∃ values,
      sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok values ∧
      Refined.FactsHold model ib values outputs := by
  intro Γ op facts outputs checked ctx call env args _width permitted _valid holds
  unfold rule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named, regions, key⟩ := header
    split at checked
    · next width input resultType =>
      cases found : input.prior.prior.indices with
      | none => simp [found] at checked
      | some range =>
        simp only [found, bind, Option.bind] at checked
        split at checked
        · split at checked
          · next accepted =>
            cases checked
            obtain _ | ⟨head, tail⟩ := holds
            cases tail
            obtain ⟨bits, rfl⟩ := initialized_int head.1
            have interval : range.Holds (.scalar (.some (.bits width bits))) := by
              simpa only [model, found, BoundedFloatFacts.Uniform] using head.2.1.1.2
            obtain ⟨z, decoded, lower, upper⟩ := interval
            have integer : bits.toInt = z := by
              simpa only [IndexRanges.value?, Sem.Ops.Tensor.indexVal?, Option.some.injEq] using decoded
            have bounded := W8A8IntegerFloatBounds.check_sound accepted bits (by omega)
            have execution := cast_exec op modelled dialect named regions resultType key ctx call env bits
            obtain ⟨base, _laws, agreement, interpretation⟩ := permitted
            have conversion : (W8A8ByteModel.extend base).ofInt
                (Spec.Dequant.defaultKey "arith.sitofp") width .f32 bits =
                base.ofInt (Spec.Dequant.defaultKey "arith.sitofp") width .f32 bits := by
              simp [W8A8ByteModel.extend, Spec.Dequant.defaultKey]
            rw [interpretation, conversion, agreement.signedInteger32] at execution
            exact ⟨_, execution, .cons (PolicyW8A8RotaryScalar.result bounded) .nil⟩
          · cases checked
        · cases checked
    · cases checked
  · cases checked

def book (ib : Nat) : Contextual.Book model sourceSem ib allowed where
  rule := rule
  sound := rule_sound ib

#print axioms rule_sound

end MerlinVerify.Check.PolicyW8A8RotaryCast
