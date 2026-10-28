/- Source-quantizer initialization from actual value classes under explicit primitive model
laws. The rule checks the source operation itself, not a cached calibration inventory. -/
import MerlinVerify.Check.DefinedContext
import MerlinVerify.Check.FiniteFacts
import MerlinVerify.Check.PolicyW8A8Dequant

namespace MerlinVerify.Check.PolicyW8A8FiniteQuantize

open MerlinVerify MerlinVerify.Sem.Ops Rules Definedness
open PolicyW8A8SourceProgress

local notation "sourceSem" => W8A8SourceSemantics.opSem

def ContextOk (ctx : Ctx) : Prop :=
  ∃ base, W8A8FiniteLaws.Laws base ∧ ctx.M = W8A8ByteModel.extend base

def allowed (ctx : Ctx) (_ : CallSem) : Prop := ContextOk ctx

def rule (_ : Refined.FEnv FiniteFacts.Class) (op : Op) (fs : List (Refined.Fact FiniteFacts.Class)) :
    Option (List (Refined.Fact FiniteFacts.Class)) :=
  match fs, W8A8SourceContract.inspect (typesAt op (fs.map (·.base.ty))) op with
  | [⟨⟨.tensor shape (.float .f32), _⟩, dataClass⟩,
      ⟨⟨.tensor [] (.float .f32), _⟩, .normal⟩, ⟨⟨.tensor [] (.int width), true⟩, _⟩], some config =>
    if config.kind = .quantize ∧ config.shape = shape ∧ config.zeroWidth = width ∧
        (dataClass = .finite ∨ dataClass = .normal) then
      some [Refined.emptyFact FiniteFacts.model ⟨.tensor shape (.int 8), true⟩]
    else none
  | _, _ => none

theorem rule_sound (ib : Nat) : ∀ Γ op fs rs,
    rule Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds FiniteFacts.model ib Γ env →
    Refined.FactsHold FiniteFacts.model ib args fs →
    ∃ vs, sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok vs ∧
      Refined.FactsHold FiniteFacts.model ib vs rs := by
  intro Γ op fs rs checked ctx call env args _width permitted _valid inputs
  unfold rule at checked
  split at checked
  · next shape dataInit dataClass scaleInit width zeroClass config accepted =>
    split at checked
    · next conditions =>
      obtain ⟨kind, shapeEq, widthEq, dataClassAllowed⟩ := conditions
      cases checked
      obtain ⟨base, laws, interpretation⟩ := permitted
      obtain ⟨declared, lookup, eligible⟩ := W8A8SourceContract.inspect_sound accepted
      have typed : HasTys ib args (argumentTys config) := by
        simpa [argumentTys, inputType, kind, shapeEq, widthEq] using inputs.typed
      have actualTypes := PolicyW8A8Dequant.argumentTypes_typed typed
      rw [← eligible.2.1, ← lookup] at actualTypes
      have execution := W8A8SourceSemantics.recognized accepted actualTypes ctx call
      obtain _ | ⟨dataHead, tail⟩ := inputs
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
            have scaleBounds : Idx.InBounds scale.shape [] := by
              rw [scaleHead.1.1.1]; simp [Idx.InBounds]
            have zeroBounds : Idx.InBounds zero.shape [] := by
              rw [zeroHead.1.1.1]; simp [Idx.InBounds]
            have normal := scaleHead.2 [] scaleBounds
            change FiniteFacts.pointCheck .normal (scale.get []) = true at normal
            unfold FiniteFacts.pointCheck at normal
            cases decodedScale : Dequant.float32 (scale.get []) with
            | none => simp [decodedScale] at normal
            | some scaleBits =>
              simp only [decodedScale] at normal
              obtain ⟨zeroBits, decodedZero⟩ := dequant_bits_defined
                (DequantSource.tensorTyped_get zeroTyped zeroBounds) (zeroHead.1.2 rfl [] zeroBounds)
              have validInputs : W8A8SourceQuantize.validInputs config.zeroWidth config.shape
                  input scale zero = true := by
                simp only [W8A8SourceQuantize.validInputs, Bool.and_eq_true, Bool.or_eq_true,
                  beq_iff_eq, decide_eq_true_eq]
                rw [widthEq, shapeEq]
                exact ⟨⟨⟨⟨by simpa [widthEq] using eligible.2.2.2.2.2.2.1,
                  ⟨dataHead.1.1.1, scaleHead.1.1.1, zeroHead.1.1.1⟩⟩, inputTyped⟩, scaleTyped⟩, zeroTyped⟩
              have finite := FiniteFacts.tensor_finite dataClassAllowed dataHead.2
              rw [dataHead.1.1.1] at finite
              have domain : PolicyW8A8FiniteDomain.check config.zeroWidth config.shape
                  [.tensor input, .tensor scale, .tensor zero] = true := by
                simp only [PolicyW8A8FiniteDomain.check, validInputs, ↓reduceIte, decodedScale]
                rw [widthEq, decodedZero]
                simp only [normal, Bool.true_and]
                simpa only [shapeEq] using finite
              obtain ⟨output, executed, outputTyped, outputDefined⟩ :=
                PolicyW8A8FiniteDomain.check_sound (ib := ib) domain base laws
              refine ⟨[.tensor output], ?_, Refined.FactsHold.empty
                (.cons ⟨by simpa [shapeEq] using outputTyped, fun _ => outputDefined⟩ .nil) FiniteFacts.model⟩
              rw [eligible.2.2.2.2.1, evalRegions, execution,
                W8A8SourceContract.eval_recognized accepted, W8A8SourceContract.run, kind, interpretation]
              exact executed
    · cases checked
  · cases checked

def book (ib : Nat) : Contextual.Book FiniteFacts.model sourceSem ib allowed where
  rule := rule
  sound := rule_sound ib

#print axioms rule_sound
#print axioms book

end MerlinVerify.Check.PolicyW8A8FiniteQuantize
