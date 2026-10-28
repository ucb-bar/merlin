/- Numerical facts for source dequantization, distinct from lower_quant_ext correspondence.
Literal scale and singleton zero-point facts are bound to the actual SSA operands. The finite
postcondition is recomputed over all 256 possible signed-byte inputs. -/
import MerlinVerify.Check.PolicyW8A8RotaryFacts
import MerlinVerify.Check.PolicyW8A8NumericalDequant
import MerlinVerify.Rules.W8A8ByteBounds

namespace MerlinVerify.Check.PolicyW8A8RotaryDequant

open MerlinVerify Rules Definedness Sem.Ops PolicyW8A8RotaryFacts PolicyW8A8SourceProgress

local notation "sourceSem" => W8A8SourceSemantics.opSem
local notation "allowed" => PolicyW8A8NumericalContext.allowed

def boundedInfo (cap : Nat) : Info := ⟨PolicyW8A8NumericalDequant.finiteInfo, some cap⟩

theorem point_finite {cap : Nat} {value : Val}
    (holds : pointCheck (some cap) value = true) : PolicyW8A8NumericalFacts.pointCheck .finite value = true := by
  cases decoded : Dequant.float32 value with
  | none => simp [pointCheck, decoded] at holds
  | some bits =>
    have bounded : W8A8IEEEBounds.ExponentLE cap bits := by simpa [pointCheck, decoded] using holds
    simpa [PolicyW8A8NumericalFacts.pointCheck, PolicyW8A8NumericalFacts.bitsCheck, decoded] using bounded.1

theorem boundedInfo_sound {cap : Nat} {value : RVal} (bounded : Holds (some cap) value) :
    model.holds (boundedInfo cap) value := by
  refine ⟨PolicyW8A8NumericalDequant.finiteInfo_sound ?_, bounded⟩
  cases value with
  | scalar _ => exact point_finite bounded
  | tensor _ => exact fun point bounds => point_finite (bounded point bounds)

def rule (_ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  match facts, W8A8SourceContract.inspect (typesAt op (facts.map (·.base.ty))) op with
  | [⟨⟨.tensor shape (.int 8), true⟩, _⟩,
      ⟨⟨.tensor [] (.float .f32), true⟩, scaleInfo⟩, ⟨⟨.tensor [] (.int width), true⟩, zeroInfo⟩], some config => do
    let scale ← scaleInfo.prior.prior.floating.literal
    let range ← zeroInfo.prior.prior.indices
    if config.kind = .dequantize ∧ config.shape = shape ∧ config.zeroWidth = width ∧
        range.upper = range.lower + 1 ∧
        W8A8ByteBounds.check (BitVec.ofNat width range.lower) scale
          (W8A8ByteBounds.cap (BitVec.ofNat width range.lower) scale) = true then
      some [⟨⟨.tensor shape (.float .f32), true⟩, boundedInfo (W8A8ByteBounds.cap (BitVec.ofNat width range.lower) scale)⟩]
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
  · next shape inputInfo scaleInfo width zeroInfo config accepted =>
    cases scaleFound : scaleInfo.prior.prior.floating.literal with
    | none => simp [scaleFound] at checked
    | some scaleBits =>
      cases rangeFound : zeroInfo.prior.prior.indices with
      | none => simp [scaleFound, rangeFound] at checked
      | some range =>
        simp only [scaleFound, rangeFound, bind, Option.bind] at checked
        split at checked
        · next conditions =>
          obtain ⟨kind, shapeEq, widthEq, singleton, numerical⟩ := conditions
          cases checked
          obtain ⟨base, _laws, agreement, interpretation⟩ := permitted
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
                have scaleBounds : Idx.InBounds scale.shape [] := by rw [scaleHead.1.1.1]; simp [Idx.InBounds]
                have zeroBounds : Idx.InBounds zero.shape [] := by rw [zeroHead.1.1.1]; simp [Idx.InBounds]
                have scaleLiteral : FloatFacts.Literal (some scaleBits) (.tensor scale) := by
                  simpa only [scaleFound] using scaleHead.2.1.1.1.2
                have scaleDecoded : Dequant.float32 (scale.get []) = some scaleBits := by
                  rw [scaleLiteral [] scaleBounds]; rfl
                have zeroTyped := PolicyW8A8Dequant.int_tensorTyped zeroHead.1.1
                obtain ⟨zeroBits, zeroDecoded⟩ := dequant_bits_defined
                  (DequantSource.tensorTyped_get zeroTyped zeroBounds) (zeroHead.1.2 rfl [] zeroBounds)
                have uniform : BoundedFloatFacts.Uniform (some range) (.tensor zero) := by
                  simpa only [rangeFound] using zeroHead.2.1.1.2
                have zeroSame := PolicyW8A8NumericalDequant.singleton_bits zeroDecoded (uniform [] zeroBounds) singleton
                have configTyped : HasTys ib [.tensor input, .tensor scale, .tensor zero] (argumentTys config) := typed
                obtain ⟨values, executed, outputTyped, outputInitialized⟩ :=
                  PolicyW8A8Dequant.run_total kind ctx.M configTyped
                have initialized : ResultsDefined [.tensor input, .tensor scale, .tensor zero] := by
                  intro value member
                  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
                  rcases member with rfl | rfl | rfl
                  · exact dataHead.1.2 rfl
                  · exact scaleHead.1.2 rfl
                  · exact zeroHead.1.2 rfl
                have exactResult : values = [.tensor (W8A8DequantWidth.result width ⟨false, 0, shape⟩ ctx.M input scale zero)] := by
                  simp only [W8A8SourceContract.run, kind, W8A8DequantWidth.run] at executed
                  split at executed
                  · simpa only [widthEq, shapeEq, Outcome.ok.injEq] using executed.symm
                  · cases executed
                subst values
                refine ⟨[.tensor (W8A8DequantWidth.result width ⟨false, 0, shape⟩ ctx.M input scale zero)], ?_,
                  .cons ⟨⟨?_, fun _ => outputInitialized initialized _ (by simp)⟩,
                  boundedInfo_sound ?_⟩ .nil⟩
                · rw [eligible.2.2.2.2.1, evalRegions, execution, W8A8SourceContract.eval_recognized accepted]
                  exact executed
                · have expected : HasTys ib
                      [.tensor (W8A8DequantWidth.result width ⟨false, 0, shape⟩ ctx.M input scale zero)]
                      [.tensor shape (.float .f32)] := by
                    simpa [resultTys, kind, shapeEq] using outputTyped
                  cases expected with
                  | cons head _ => exact head
                · intro point bounds
                  change Idx.InBounds shape point at bounds
                  have inputBounds : Idx.InBounds input.shape point := dataHead.1.1.1 ▸ bounds
                  have inputTyped := PolicyW8A8Dequant.int_tensorTyped dataHead.1.1
                  obtain ⟨q, qDecoded⟩ := dequant_bits_defined
                    (DequantSource.tensorTyped_get inputTyped inputBounds) (dataHead.1.2 rfl point inputBounds)
                  have bounded := W8A8ByteBounds.check_sound agreement numerical q
                  simp only [pointCheck, W8A8DequantWidth.result, Tensor.canon, bounds, ↓reduceIte,
                    Spec.Dequant.parameterIndex, Bool.false_eq_true,
                    W8A8DequantWidth.scalarValue, qDecoded, scaleDecoded, zeroDecoded,
                    W8A8DequantWidth.scalar, interpretation, zeroSame]
                  change decide (W8A8IEEEBounds.ExponentLE _ _) = true
                  exact decide_eq_true bounded
        · cases checked
  · cases checked

def book (ib : Nat) : Contextual.Book model sourceSem ib allowed where
  rule := rule
  sound := rule_sound ib

#print axioms rule_sound
#print axioms book

end MerlinVerify.Check.PolicyW8A8RotaryDequant
