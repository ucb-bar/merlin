/- Exact definedness obligation for the existing quantization component. This file adds no
float law and does not replace the independent source quantization contract. In particular,
clamp operation names do not establish conversion definedness under an arbitrary FloatModel. -/
import MerlinVerify.Rules.W8A8Scalar

namespace MerlinVerify.Rules.W8A8QuantizeDomain

open MerlinVerify

/-- The exact ordered value passed to signed-byte conversion. -/
def clamped (M : FloatModel) (x inverse upper lower : FBits .f32) : FBits .f32 :=
  let product := M.op (Spec.Dequant.defaultKey "arith.mulf") .f32 [x, inverse]
  let rounded := M.op (Spec.Dequant.defaultKey "math.roundeven") .f32 [product]
  let high := M.op (Spec.Dequant.defaultKey "arith.minimumf") .f32 [rounded, upper]
  M.op (Spec.Dequant.defaultKey "arith.maximumf") .f32 [high, lower]

def conversion (M : FloatModel) (x inverse upper lower : FBits .f32) : Option (BitVec 8) :=
  M.toInt (Spec.Dequant.defaultKey "arith.fptosi") .f32 8 (clamped M x inverse upper lower)

/-- A domain condition, not an assumed property of all floating models or all input bits. -/
theorem defined_iff (M : FloatModel) (x inverse upper lower : FBits .f32) :
    W8A8Scalar.quantize M (some x) (some inverse) upper lower ≠ .poison ↔
      ∃ byte, conversion M x inverse upper lower = some byte := by
  change (match conversion M x inverse upper lower with
    | some byte => Val.some (.bits 8 byte)
    | none => Val.poison) ≠ .poison ↔ _
  cases converted : conversion M x inverse upper lower <;> simp

theorem conversion_sufficient {M : FloatModel} {x inverse upper lower : FBits .f32} {byte : BitVec 8}
    (defined : conversion M x inverse upper lower = some byte) :
    W8A8Scalar.quantize M (some x) (some inverse) upper lower = .some (.bits 8 byte) := by
  change (match conversion M x inverse upper lower with
    | some value => Val.some (.bits 8 value)
    | none => Val.poison) = _
  simp [defined]

/-- This deliberately non-IEEE model is already permitted by the frozen semantics. -/
theorem counterexample (x inverse upper lower : FBits .f32) :
    W8A8Scalar.quantize W8A8Scale.separatingModel (some x) (some inverse) upper lower = .poison := rfl

theorem not_uniformly_defined :
    ¬ ∀ (M : FloatModel) (x inverse : FBits .f32),
      W8A8Scalar.quantize M (some x) (some inverse) 0x42fe0000#32 0xc2fe0000#32 ≠ .poison := by
  intro total
  exact total W8A8Scale.separatingModel 0#32 0x3f800000#32 (counterexample ..)

/-- The actual proved region can return an initialized outcome containing a poison result.
Non-stuck support therefore cannot substitute for a nonpoison-result obligation. -/
theorem region_counterexample (ctx : Ctx) (call : CallSem) (env : Env) :
    evalRegion Sem.Ops.opSem { ctx with M := W8A8Scale.separatingModel } call
      (W8A8Scalar.quantizeCaptures 0x42fe0000#32 0xc2fe0000#32 env) W8A8Scalar.quantizeRegion
      [.scalar (.some (.flt .f32 0#32)), .scalar (.some (.flt .f32 0x3f800000#32)), .scalar .poison] =
      .ok [.scalar .poison] := by
  simpa [W8A8Scalar.floatValue, W8A8Scalar.quantize, W8A8Scale.separatingModel] using
    W8A8Scalar.quantize_sound { ctx with M := W8A8Scale.separatingModel } call env
      (some 0#32) (some 0x3f800000#32) .poison 0x42fe0000#32 0xc2fe0000#32

#print axioms defined_iff
#print axioms conversion_sufficient
#print axioms not_uniformly_defined
#print axioms region_counterexample

end MerlinVerify.Rules.W8A8QuantizeDomain
