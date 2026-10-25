/- Numerical facts for source dequantization, distinct from lower_quant_ext correspondence.
Literal scale and singleton zero-point facts are bound to the actual SSA operands. The finite
postcondition is recomputed over all 256 possible signed-byte inputs. -/
import MerlinVerify.Check.PolicyW8A8NumericalQuantize
import MerlinVerify.Rules.W8A8DequantNumerics

namespace MerlinVerify.Check.PolicyW8A8NumericalDequant

open MerlinVerify Rules Definedness Sem.Ops PolicyW8A8NumericalFacts PolicyW8A8SourceProgress

local notation "sourceSem" => W8A8SourceSemantics.opSem
local notation "allowed" => PolicyW8A8NumericalContext.allowed

def finiteInfo : Info := ⟨⟨⟨.finite, none⟩, none⟩, .finite⟩

theorem finiteInfo_sound {value : RVal} (finite : Holds .finite value) : model.holds finiteInfo value := by
  refine ⟨⟨⟨?_, trivial⟩, trivial⟩, finite⟩
  change FiniteFacts.Holds .finite value
  have point (v : Val) (holds : pointCheck .finite v = true) : FiniteFacts.pointCheck .finite v = true := by
    cases decoded : Dequant.float32 v with
    | none => simp [pointCheck, decoded] at holds
    | some _ => simpa [pointCheck, bitsCheck, FiniteFacts.pointCheck,
        PolicyW8A8FiniteDomain.pointCheck, decoded] using holds
  cases value with
  | scalar value => exact point value finite
  | tensor tensor => exact fun index bounds => point (tensor.get index) (finite index bounds)

def rule (_ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  match facts, W8A8SourceContract.inspect (typesAt op (facts.map (·.base.ty))) op with
  | [⟨⟨.tensor shape (.int 8), true⟩, _⟩,
      ⟨⟨.tensor [] (.float .f32), true⟩, scaleInfo⟩, ⟨⟨.tensor [] (.int width), true⟩, zeroInfo⟩], some config => do
    let scale ← scaleInfo.prior.floating.literal
    let range ← zeroInfo.prior.indices
    if config.kind = .dequantize ∧ config.shape = shape ∧ config.zeroWidth = width ∧
        range.upper = range.lower + 1 ∧
        W8A8DequantNumerics.finiteCheck (BitVec.ofNat width range.lower) scale = true then
      some [⟨⟨.tensor shape (.float .f32), true⟩, finiteInfo⟩]
    else none
  | _, _ => none

theorem singleton_bits {width : Nat} {bits : BitVec width} {value : Val} {range : IndexRanges.Interval}
    (decoded : Dequant.bits width value = some bits) (holds : range.Holds (.scalar value))
    (singleton : range.upper = range.lower + 1) : bits = BitVec.ofNat width range.lower := by
  obtain ⟨z, integer, lower, upper⟩ := holds
  have toInt : bits.toInt = z := by
    cases value with
    | poison => cases decoded
    | some scalar =>
      cases scalar with
      | flt _ _ => cases decoded
      | bits w original =>
        simp only [Dequant.bits] at decoded
        split at decoded
        · next equal =>
          subst w
          cases decoded
          simpa only [IndexRanges.value?, Sem.Ops.Tensor.indexVal?, Option.some.injEq] using integer
        · cases decoded
  have same : bits.toInt = (range.lower : Int) := by omega
  calc
    bits = BitVec.ofInt width bits.toInt := BitVec.ofInt_toInt.symm
    _ = BitVec.ofNat width range.lower := by rw [same, BitVec.ofInt_natCast]

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
    cases scaleFound : scaleInfo.prior.floating.literal with
    | none => simp [scaleFound] at checked
    | some scaleBits =>
      cases rangeFound : zeroInfo.prior.indices with
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
                  simpa only [scaleFound] using scaleHead.2.1.1.2
                have scaleDecoded : Dequant.float32 (scale.get []) = some scaleBits := by
                  rw [scaleLiteral [] scaleBounds]; rfl
                have zeroTyped := PolicyW8A8Dequant.int_tensorTyped zeroHead.1.1
                obtain ⟨zeroBits, zeroDecoded⟩ := dequant_bits_defined
                  (DequantSource.tensorTyped_get zeroTyped zeroBounds) (zeroHead.1.2 rfl [] zeroBounds)
                have uniform : BoundedFloatFacts.Uniform (some range) (.tensor zero) := by
                  simpa only [rangeFound] using zeroHead.2.1.2
                have zeroSame := singleton_bits zeroDecoded (uniform [] zeroBounds) singleton
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
                  finiteInfo_sound ?_⟩ .nil⟩
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
                  have finite := W8A8DequantNumerics.finiteCheck_sound agreement numerical q
                  simp only [pointCheck, W8A8DequantWidth.result, Tensor.canon, bounds, ↓reduceIte,
                    Spec.Dequant.parameterIndex, Bool.false_eq_true,
                    W8A8DequantWidth.scalarValue, qDecoded, scaleDecoded, zeroDecoded,
                    W8A8DequantWidth.scalar, interpretation, zeroSame]
                  change decide (W8A8IEEE32.Finite _) = true
                  exact decide_eq_true finite
        · cases checked
  · cases checked

def book (ib : Nat) : Contextual.Book model sourceSem ib allowed where
  rule := rule
  sound := rule_sound ib

#print axioms singleton_bits
#print axioms rule_sound
#print axioms book

end MerlinVerify.Check.PolicyW8A8NumericalDequant
