/- Fact-dependent support of actual source operations. This is a draft integration layer,
not a numerical-domain proof: typed source data may still produce poison. -/
import MerlinVerify.Rules.W8A8SourceSemantics
import MerlinVerify.Check.RulesDequant

namespace MerlinVerify.Check.PolicyW8A8SourceProgress

open MerlinVerify MerlinVerify.Sem.Ops Rules
open W8A8SourceContract
open PolicyW8A8SourceFacts
open ConstantFacts (Fact model)

def inputType (c : Config) : VTy :=
  .tensor c.shape (if c.kind = .quantize then .float .f32 else .int 8)

def argumentTys (c : Config) : List VTy :=
  [inputType c, .tensor [] (.float .f32), .tensor [] (.int c.zeroWidth)]

def resultTys (c : Config) : List VTy :=
  [.tensor c.shape (if c.kind = .quantize then .int 8 else .float .f32)]

def toTy : VTy → Ty
  | .scalar ty => ty
  | .tensor shape elt => tensor shape elt

def typesAt (op : Op) (argTys : List VTy) : List (ValueId × Ty) :=
  op.operands.zip (argTys.map toTy)

/-- The support claim refers to actual operands fetched from the current SSA environment.
Calibration values are supplied by proved facts, not by runtime arguments chosen by a hint. -/
def Safe (ib : Nat) (facts : List Fact) (op : Op) (argTys resTys : List VTy) : Prop :=
  ∀ ctx call env args, BindingFacts.Valid model facts env →
    env.getAll op.operands = .ok args → HasTys ib args argTys →
    ∃ results, W8A8SourceSemantics.opSem ctx call op args
      (evalRegions W8A8SourceSemantics.opSem call env op.regions) = .ok results ∧
      HasTys ib results resTys

def check (facts : List Fact) (op : Op) (argTys resTys : List VTy) : Bool :=
  match inspect (typesAt op argTys) op with
  | none => false
  | some c => argTys == argumentTys c && resTys == resultTys c &&
      checkParameters (typesAt op argTys) facts op

theorem check_sound {ib : Nat} {facts : List Fact} {op : Op} {argTys resTys : List VTy}
    (checked : check facts op argTys resTys = true) : Safe ib facts op argTys resTys := by
  unfold check at checked
  cases accepted : inspect (typesAt op argTys) op with
  | none => simp [accepted] at checked
  | some c =>
    simp only [accepted, Bool.and_eq_true, beq_iff_eq] at checked
    obtain ⟨⟨argTypes, resultTypes⟩, parameters⟩ := checked
    intro ctx call env args valid fetched typed
    obtain ⟨actual, dataId, scaleId, zeroId, scale, zero, same, operands,
      scaleBound, zeroBound, scaleShape, zeroShape, scaleTyped, zeroTyped,
      s, z, scaleDecoded, zeroDecoded, admissible⟩ := checkParameters_sound parameters env valid
    have sameConfig : actual = c := Option.some.inj (same.symm.trans accepted)
    subst actual
    obtain ⟨declared, lookup, eligible⟩ := inspect_sound accepted
    rw [argTypes] at typed
    unfold argumentTys at typed
    cases typed with
    | @cons dataValue _ _ _ dataTyped tail =>
      cases tail with
      | @cons scaleValue _ _ _ _scaleHasTy tail =>
        cases tail with
        | @cons zeroValue _ _ _ _zeroHasTy tail =>
          cases tail
          cases dataValue with
          | scalar data => cases dataTyped
          | tensor input =>
            rw [operands] at fetched
            cases found : env.get? dataId with
            | none => simp [Env.getAll, Outcome.mapM, found, Outcome.ofOption] at fetched
            | some value =>
              simp only [Env.getAll, Outcome.mapM, found, scaleBound, zeroBound,
                Outcome.ofOption, Outcome.bind_ok, Outcome.ok.injEq, List.cons.injEq] at fetched
              obtain ⟨_, scaleSame, zeroSame, _⟩ := fetched
              subst scaleValue
              subst zeroValue
              have scaleElt : scale.elt = .float .f32 := by
                have checkedScale := scaleTyped
                simp only [Dequant.tensorTyped, Bool.and_eq_true, beq_iff_eq] at checkedScale
                exact checkedScale.1
              have zeroElt : zero.elt = .int c.zeroWidth := by
                have checkedZero := zeroTyped
                simp only [Dequant.tensorTyped, Bool.and_eq_true, beq_iff_eq] at checkedZero
                exact checkedZero.1
              have inputShape : input.shape = c.shape := dataTyped.1
              have inputElt : input.elt = (if c.kind = .quantize then .float .f32 else .int 8) := dataTyped.2.1
              have inputTyped : Dequant.tensorTyped input
                  (if c.kind = .quantize then .float .f32 else .int 8) = true := by
                apply dequant_tensorTyped ?_ dataTyped
                cases c.kind <;> simp
              have actualTypes : W8A8SourceSemantics.argumentTypes
                  [.tensor input, .tensor scale, .tensor zero] = op.operands.mapM
                    (fun id => (typesAt op argTys).lookup id) := by
                rw [lookup, eligible.2.1]
                simp [W8A8SourceSemantics.argumentTypes, operandTypes, inputShape, inputElt,
                  scaleShape, zeroShape, scaleElt, zeroElt]
              have regionless := eligible.2.2.2.2.1
              have execution := W8A8SourceSemantics.recognized accepted actualTypes ctx call
              rw [eval_recognized accepted] at execution
              rw [regionless, evalRegions, execution]
              have widthAllowed := eligible.2.2.2.2.2.2.1
              cases kind : c.kind with
              | quantize =>
                have inputs : W8A8SourceQuantize.validInputs c.zeroWidth c.shape input scale zero = true := by
                  simp only [W8A8SourceQuantize.validInputs, Bool.and_eq_true, Bool.or_eq_true,
                    beq_iff_eq, decide_eq_true_eq]
                  exact ⟨⟨⟨⟨widthAllowed, ⟨inputShape, scaleShape, zeroShape⟩⟩,
                    by simpa [kind] using inputTyped⟩, scaleTyped⟩, zeroTyped⟩
                refine ⟨[.tensor (W8A8SourceQuantize.result c.zeroWidth ctx.M c.shape input s z)], ?_, ?_⟩
                · simp [W8A8SourceContract.run, kind, W8A8SourceQuantize.run, inputs,
                    scaleDecoded, zeroDecoded, admissible]
                · rw [resultTypes]
                  exact .cons (by simpa [resultTys, kind] using
                    W8A8SourceQuantize.result_typed ib c.zeroWidth ctx.M c.shape input s z) .nil
              | dequantize =>
                have inputs : W8A8DequantWidth.validInputs c.zeroWidth ⟨false, 0, c.shape⟩ input scale zero = true := by
                  simp only [W8A8DequantWidth.validInputs, Dequant.paramShape, Bool.false_eq_true,
                    ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq]
                  exact ⟨⟨⟨⟨inputShape, scaleShape, zeroShape⟩,
                    by simpa [kind] using inputTyped⟩, scaleTyped⟩, zeroTyped⟩
                refine ⟨[.tensor (W8A8DequantWidth.result c.zeroWidth ⟨false, 0, c.shape⟩ ctx.M input scale zero)], ?_, ?_⟩
                · simp [W8A8SourceContract.run, kind, W8A8DequantWidth.run, inputs]
                · rw [resultTypes]
                  exact .cons (by simpa [resultTys, kind] using
                    W8A8DequantWidth.result_typed ib c.zeroWidth ⟨false, 0, c.shape⟩ ctx.M input scale zero) .nil

#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8SourceProgress
