/- Source quantization may consume infinities. The checker requires proved non-NaN data,
the actual literal scale, and a recomputed finite/nonzero binary64-derived reciprocal. -/
import MerlinVerify.Check.PolicyW8A8NumericalScalar
import MerlinVerify.Check.PolicyW8A8Dequant

namespace MerlinVerify.Check.PolicyW8A8NumericalQuantize

open MerlinVerify Rules Definedness Sem.Ops PolicyW8A8NumericalFacts
open PolicyW8A8SourceProgress

local notation "sourceSem" => W8A8SourceSemantics.opSem
local notation "allowed" => PolicyW8A8NumericalContext.allowed

def rule (_ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  match facts, W8A8SourceContract.inspect (typesAt op (facts.map (·.base.ty))) op with
  | [⟨⟨.tensor shape (.float .f32), true⟩, dataInfo⟩,
      ⟨⟨.tensor [] (.float .f32), true⟩, scaleInfo⟩, ⟨⟨.tensor [] (.int width), true⟩, _⟩], some config => do
    let scale ← scaleInfo.prior.floating.literal
    if config.kind = .quantize ∧ config.shape = shape ∧ config.zeroWidth = width ∧
        accepts dataInfo .nonNaN = true ∧ W8A8ArithmeticModel.inverseCheck scale = true then
      some [Refined.emptyFact model ⟨.tensor shape (.int 8), true⟩]
    else none
  | _, _ => none

theorem rule_sound (ib : Nat) : ∀ Γ op facts outputs,
    rule Γ op facts = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds model ib Γ env →
    Refined.FactsHold model ib args facts → ∃ values,
      sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok values ∧
      Refined.FactsHold model ib values outputs := by
  intro Γ op facts outputs checked ctx call env args _width permitted _valid holds
  unfold rule at checked
  split at checked
  · next shape dataInfo scaleInfo width zeroInfo config accepted =>
    cases decodedLiteral : scaleInfo.prior.floating.literal with
    | none => simp [decodedLiteral] at checked
    | some scaleBits =>
      simp only [decodedLiteral, bind, Option.bind] at checked
      split at checked
      · next conditions =>
        obtain ⟨kind, shapeEq, widthEq, inputAccepted, inverseChecked⟩ := conditions
        cases checked
        obtain ⟨base, laws, agreement, interpretation⟩ := permitted
        obtain ⟨declared, lookup, eligible⟩ := W8A8SourceContract.inspect_sound accepted
        have typed : HasTys ib args (argumentTys config) := by
          simpa [argumentTys, inputType, kind, shapeEq, widthEq] using holds.typed
        have actualTypes := PolicyW8A8Dequant.argumentTypes_typed typed
        rw [← eligible.2.1, ← lookup] at actualTypes
        have execution := W8A8SourceSemantics.recognized accepted actualTypes ctx call
        obtain _ | ⟨dataHead, tail⟩ := holds
        obtain _ | ⟨scaleHead, tail⟩ := tail
        obtain _ | ⟨zeroHead, tail⟩ := tail
        cases tail
        rename_i dataValue scaleValue zeroValue
        cases dataValue with
        | scalar _ => cases dataHead.1.1
        | tensor input =>
          cases scaleValue with
          | scalar _ => cases scaleHead.1.1
          | tensor scale =>
            cases zeroValue with
            | scalar _ => cases zeroHead.1.1
            | tensor zero =>
              have inputTyped := dequant_tensorTyped (Or.inr (Or.inr rfl)) dataHead.1.1
              have scaleTyped := dequant_tensorTyped (Or.inr (Or.inr rfl)) scaleHead.1.1
              have zeroTyped := PolicyW8A8Dequant.int_tensorTyped zeroHead.1.1
              have scaleBounds : Idx.InBounds scale.shape [] := by rw [scaleHead.1.1.1]; simp [Idx.InBounds]
              have zeroBounds : Idx.InBounds zero.shape [] := by rw [zeroHead.1.1.1]; simp [Idx.InBounds]
              have scaleLiteral : FloatFacts.Literal (some scaleBits) (.tensor scale) := by
                simpa only [decodedLiteral] using scaleHead.2.1.1.2
              have scaleDecoded : Dequant.float32 (scale.get []) = some scaleBits := by
                rw [scaleLiteral [] scaleBounds]; rfl
              obtain ⟨zeroBits, zeroDecoded⟩ := dequant_bits_defined
                (DequantSource.tensorTyped_get zeroTyped zeroBounds) (zeroHead.1.2 rfl [] zeroBounds)
              have validInputs : W8A8SourceQuantize.validInputs config.zeroWidth config.shape input scale zero = true := by
                simp only [W8A8SourceQuantize.validInputs, Bool.and_eq_true, Bool.or_eq_true,
                  beq_iff_eq, decide_eq_true_eq]
                rw [widthEq, shapeEq]
                exact ⟨⟨⟨⟨by simpa [widthEq] using eligible.2.2.2.2.2.2.1,
                  ⟨dataHead.1.1.1, scaleHead.1.1.1, zeroHead.1.1.1⟩⟩, inputTyped⟩, scaleTyped⟩, zeroTyped⟩
              have dataNN := accepts_sound inputAccepted dataHead.2
              have numericalDomain : W8A8SourceQuantize.Domain width base config.shape input scaleBits zeroBits := by
                intro point bounds x decoded
                have inBounds : Idx.InBounds input.shape point := by simpa [dataHead.1.1.1, ← shapeEq] using bounds
                have each := dataNN point inBounds
                have nonNaN : W8A8IEEE32.NonNaN x := by
                  simpa [Holds, pointCheck, bitsCheck, decoded] using each
                exact W8A8ArithmeticModel.shifted_nonNaN agreement laws.source
                  (by simpa [widthEq] using eligible.2.2.2.2.2.2.1) nonNaN inverseChecked
              obtain ⟨admissible, _, _⟩ := W8A8ArithmeticModel.inverseCheck_sound agreement inverseChecked
              obtain ⟨output, executed, outputTyped, outputDefined⟩ :=
                W8A8SourceQuantize.run_defined (ib := ib) (width := width) (base := base)
                  (by simpa only [widthEq] using validInputs) scaleDecoded zeroDecoded
                  admissible (dataHead.1.2 rfl) numericalDomain
              refine ⟨[.tensor output], ?_, Refined.FactsHold.empty
                (.cons ⟨by simpa [shapeEq] using outputTyped, fun _ => outputDefined⟩ .nil) model⟩
              rw [eligible.2.2.2.2.1, evalRegions, execution,
                W8A8SourceContract.eval_recognized accepted, W8A8SourceContract.run, kind, interpretation]
              simpa only [widthEq] using executed
      · cases checked
  · cases checked

def book (ib : Nat) : Contextual.Book model sourceSem ib allowed where
  rule := rule
  sound := rule_sound ib

#print axioms rule_sound
#print axioms book

end MerlinVerify.Check.PolicyW8A8NumericalQuantize
