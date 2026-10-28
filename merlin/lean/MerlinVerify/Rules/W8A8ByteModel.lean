/- A concrete byte/clamp extension of any parametric float model. Only default f32 min/max
and signed-byte conversion are replaced by pure integer-field computations. This candidate
is not installed in frozen semantics; it supports explicit numerical qualification. -/
import MerlinVerify.Rules.W8A8ByteFloat

namespace MerlinVerify.Rules.W8A8ByteModel

open MerlinVerify

def extend (base : FloatModel) : FloatModel where
  op key kind values := match kind with
    | .f32 =>
      if key = Spec.Dequant.defaultKey "arith.minimumf" then
        match values with
        | [x, y] => W8A8ByteFloat.minimum x y
        | _ => base.op key .f32 values
      else if key = Spec.Dequant.defaultKey "arith.maximumf" then
        match values with
        | [x, y] => W8A8ByteFloat.maximum x y
        | _ => base.op key .f32 values
      else base.op key .f32 values
    | kind => base.op key kind values
  cmp := base.cmp
  convert := base.convert
  toInt key kind width value := match kind with
    | .f32 =>
      if key = Spec.Dequant.defaultKey "arith.fptosi" then
        if same : 8 = width then same ▸ W8A8ByteFloat.toByte value
        else base.toInt key .f32 width value
      else base.toInt key .f32 width value
    | kind => base.toInt key kind width value
  ofInt := base.ofInt

theorem minimum_eq (base : FloatModel) (x y : FBits .f32) :
    (extend base).op (Spec.Dequant.defaultKey "arith.minimumf") .f32 [x, y] =
      W8A8ByteFloat.minimum x y := by
  simp [extend]

theorem maximum_eq (base : FloatModel) (x y : FBits .f32) :
    (extend base).op (Spec.Dequant.defaultKey "arith.maximumf") .f32 [x, y] =
      W8A8ByteFloat.maximum x y := by
  simp [extend, Spec.Dequant.defaultKey]

theorem multiply_eq (base : FloatModel) (x y : FBits .f32) :
    (extend base).op (Spec.Dequant.defaultKey "arith.mulf") .f32 [x, y] =
      base.op (Spec.Dequant.defaultKey "arith.mulf") .f32 [x, y] := by
  simp [extend, Spec.Dequant.defaultKey]

theorem round_eq (base : FloatModel) (x : FBits .f32) :
    (extend base).op (Spec.Dequant.defaultKey "math.roundeven") .f32 [x] =
      base.op (Spec.Dequant.defaultKey "math.roundeven") .f32 [x] := by
  simp [extend, Spec.Dequant.defaultKey]

theorem conversion_eq (base : FloatModel) (x : FBits .f32) :
    (extend base).toInt (Spec.Dequant.defaultKey "arith.fptosi") .f32 8 x =
      W8A8ByteFloat.toByte x := by
  simp [extend]

def rounded (base : FloatModel) (x inverse : FBits .f32) : FBits .f32 :=
  base.op (Spec.Dequant.defaultKey "math.roundeven") .f32
    [base.op (Spec.Dequant.defaultKey "arith.mulf") .f32 [x, inverse]]

theorem clamped_formula (base : FloatModel) (x inverse upper lower : FBits .f32) :
    W8A8QuantizeDomain.clamped (extend base) x inverse upper lower =
      W8A8ByteFloat.maximum (W8A8ByteFloat.minimum (rounded base x inverse) upper) lower := by
  unfold W8A8QuantizeDomain.clamped
  dsimp only
  rw [maximum_eq, minimum_eq, round_eq, multiply_eq]
  rfl

theorem clamped_eq (base : FloatModel) (x inverse : FBits .f32)
    (nonNaN : W8A8ByteFloat.isNaN (rounded base x inverse) = false) :
    W8A8QuantizeDomain.clamped (extend base) x inverse 0x42fe0000#32 0xc2fe0000#32 =
      W8A8ByteFloat.clamp127 (rounded base x inverse) := by
  exact (clamped_formula base x inverse 0x42fe0000#32 0xc2fe0000#32).trans
    (W8A8ByteFloat.min_max_clamp nonNaN)

/-- Clamp-before-convert definedness for this explicit model, under the exact non-NaN
intermediate condition. It is not an assumption about every existing FloatModel. -/
theorem quantize_defined (base : FloatModel) (x inverse : FBits .f32)
    (nonNaN : W8A8ByteFloat.isNaN (rounded base x inverse) = false) :
    W8A8Scalar.quantize (extend base) (some x) (some inverse) 0x42fe0000#32 0xc2fe0000#32 ≠ .poison := by
  obtain ⟨byte, converted⟩ := W8A8ByteFloat.clamp_conversion_defined nonNaN
  apply (W8A8QuantizeDomain.defined_iff ..).2
  refine ⟨byte, ?_⟩
  simpa only [W8A8QuantizeDomain.conversion, clamped_eq base x inverse nonNaN, conversion_eq] using converted

theorem quantize_result (base : FloatModel) (x inverse : FBits .f32)
    (nonNaN : W8A8ByteFloat.isNaN (rounded base x inverse) = false) :
    ∃ byte, W8A8Scalar.quantize (extend base) (some x) (some inverse)
      0x42fe0000#32 0xc2fe0000#32 = .some (.bits 8 byte) := by
  obtain ⟨byte, converted⟩ := (W8A8QuantizeDomain.defined_iff ..).1 (quantize_defined base x inverse nonNaN)
  exact ⟨byte, W8A8QuantizeDomain.conversion_sufficient converted⟩

theorem region_result (ctx : Ctx) (call : CallSem) (env : Env)
    (x inverse : FBits .f32) (initial : Val)
    (nonNaN : W8A8ByteFloat.isNaN (rounded ctx.M x inverse) = false) :
    ∃ byte, evalRegion Sem.Ops.opSem { ctx with M := extend ctx.M } call
      (W8A8Scalar.quantizeCaptures 0x42fe0000#32 0xc2fe0000#32 env) W8A8Scalar.quantizeRegion
      [.scalar (W8A8Scalar.floatValue (some x)), .scalar (W8A8Scalar.floatValue (some inverse)), .scalar initial] =
      .ok [.scalar (.some (.bits 8 byte))] := by
  obtain ⟨byte, converted⟩ := quantize_result ctx.M x inverse nonNaN
  refine ⟨byte, ?_⟩
  exact (W8A8Scalar.quantize_sound { ctx with M := extend ctx.M } call env
    (some x) (some inverse) initial 0x42fe0000#32 0xc2fe0000#32).trans
      (congrArg (fun value => Outcome.ok [RVal.scalar value]) converted)

#print axioms conversion_eq
#print axioms clamped_eq
#print axioms quantize_defined
#print axioms region_result

end MerlinVerify.Rules.W8A8ByteModel
