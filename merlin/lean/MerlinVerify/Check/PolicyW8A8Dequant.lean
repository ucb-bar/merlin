/- Total source dequantization with runtime calibration parameters. Unlike quantization,
dequantization needs no positivity or non-NaN premise to return initialized floating bits.
This is a theorem about the explicit provisional interpretation, not IEEE accuracy. -/
import MerlinVerify.Check.PolicyW8A8SourceProgress

namespace MerlinVerify.Check.PolicyW8A8Dequant

open MerlinVerify MerlinVerify.Sem.Ops Rules W8A8SourceContract PolicyW8A8SourceProgress

theorem int_tensorTyped {ib width : Nat} {t : Tensor} {shape : List Nat}
    (typed : RVal.HasTy ib (.tensor t) (.tensor shape (.int width))) :
    Dequant.tensorTyped t (.int width) = true := by
  obtain ⟨shapeEq, eltEq, elements⟩ := typed
  simp only [Dequant.tensorTyped, eltEq, beq_self_eq_true, Bool.true_and, List.all_eq_true]
  intro point member
  have value := elements point (shapeEq ▸ (mem_allPoints _ _).1 member)
  cases h : t.get point with
  | poison => rfl
  | some scalar =>
    rw [h] at value
    cases scalar <;> simp_all [Val.HasTy, Val.conforms, Scalar.conforms]

/-- Runtime tensor declarations follow from actual operand typing, including arbitrary integer
zero-point widths. No constant fact or proposed target supplies these declarations. -/
theorem argumentTypes_typed {ib : Nat} {c : Config} {args : List RVal}
    (typed : HasTys ib args (argumentTys c)) :
    W8A8SourceSemantics.argumentTypes args = some (operandTypes c) := by
  cases typed with
  | @cons q _ _ _ qt tail =>
    cases tail with
    | @cons s _ _ _ st tail =>
      cases tail with
      | @cons z _ _ _ zt tail =>
        cases tail
        cases q with
        | scalar _ => cases qt
        | tensor q =>
          cases s with
          | scalar _ => cases st
          | tensor s =>
            cases z with
            | scalar _ => cases zt
            | tensor z =>
              simp [W8A8SourceSemantics.argumentTypes, operandTypes,
                qt.1, qt.2.1, st.1, st.2.1, zt.1, zt.2.1]

/-- Typed dequantization always returns typed values. Initialized operands additionally give
initialized results, even when a scale is zero, negative, infinity or NaN. Such floating bits
are values in the semantics; this statement does not call them finite or numerically useful. -/
theorem run_total {ib : Nat} {c : Config} (kind : c.kind = .dequantize)
    (M : FloatModel) {args : List RVal} (typed : HasTys ib args (argumentTys c)) :
    ∃ results, run c M args = .ok results ∧ HasTys ib results (resultTys c) ∧
      (ResultsDefined args → ResultsDefined results) := by
  cases typed with
  | @cons q _ _ _ qt tail =>
    cases tail with
    | @cons s _ _ _ st tail =>
      cases tail with
      | @cons z _ _ _ zt tail =>
        cases tail
        cases q with
        | scalar _ => cases qt
        | tensor q =>
          cases s with
          | scalar _ => cases st
          | tensor s =>
            cases z with
            | scalar _ => cases zt
            | tensor z =>
              have qtyped : RVal.HasTy ib (.tensor q) (.tensor c.shape (.int 8)) := by
                simpa [argumentTys, inputType, kind] using qt
              have valid : W8A8DequantWidth.validInputs c.zeroWidth ⟨false, 0, c.shape⟩ q s z = true := by
                simp only [W8A8DequantWidth.validInputs, Dequant.paramShape, Bool.false_eq_true,
                  ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq]
                exact ⟨⟨⟨⟨qtyped.1, st.1, zt.1⟩, int_tensorTyped qtyped⟩,
                  dequant_tensorTyped (Or.inr (Or.inr rfl)) st⟩, int_tensorTyped zt⟩
              refine ⟨[.tensor (W8A8DequantWidth.result c.zeroWidth ⟨false, 0, c.shape⟩ M q s z)], ?_, ?_, ?_⟩
              · simp [run, kind, W8A8DequantWidth.run, valid]
              · exact .cons (by simpa [resultTys, kind] using
                  W8A8DequantWidth.result_typed ib c.zeroWidth ⟨false, 0, c.shape⟩ M q s z) .nil
              · intro initialized value member
                simp only [List.mem_singleton] at member
                subst value
                exact W8A8DequantWidth.result_defined M (Or.inl rfl) valid
                  (initialized _ (by simp)) (initialized _ (by simp)) (initialized _ (by simp))

def check (op : Op) (args results : List VTy) : Bool :=
  match inspect (typesAt op args) op with
  | none => false
  | some c => decide (c.kind = .dequantize) && args == argumentTys c && results == resultTys c

/-- The actual operation, metadata, runtime operand types and result declarations are checked.
The initialization conclusion is conditional on these same actual operands being initialized. -/
theorem check_sound {ib : Nat} {op : Op} {argTys resTys : List VTy}
    (checked : check op argTys resTys = true) (ctx : Ctx) (call : CallSem) (env : Env)
    {args : List RVal} (typed : HasTys ib args argTys) :
    ∃ results, W8A8SourceSemantics.opSem ctx call op args
        (evalRegions W8A8SourceSemantics.opSem call env op.regions) = .ok results ∧
      HasTys ib results resTys ∧ (ResultsDefined args → ResultsDefined results) := by
  unfold check at checked
  cases accepted : inspect (typesAt op argTys) op with
  | none => simp [accepted] at checked
  | some c =>
    simp only [accepted, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at checked
    obtain ⟨⟨kind, argTypes⟩, resultTypes⟩ := checked
    obtain ⟨declared, lookup, eligible⟩ := inspect_sound accepted
    have typed' : HasTys ib args (argumentTys c) := argTypes ▸ typed
    have actualTypes : W8A8SourceSemantics.argumentTypes args =
        op.operands.mapM (fun id => (typesAt op argTys).lookup id) := by
      rw [lookup, eligible.2.1]
      exact argumentTypes_typed typed'
    have execution := W8A8SourceSemantics.recognized accepted actualTypes ctx call
    rw [eval_recognized accepted] at execution
    rw [eligible.2.2.2.2.1, evalRegions, execution, resultTypes]
    exact run_total kind ctx.M typed'

#print axioms int_tensorTyped
#print axioms argumentTypes_typed
#print axioms run_total
#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8Dequant
