/- Explicit arithmetic interpretation for a separately named W8A8 numerical candidate.
Agreement names actual primitive keys and computations, not the desired classification of a
program result. Existing source/dynamic laws remain separate premises with unchanged meanings. -/
import MerlinVerify.Rules.W8A8IEEE32

namespace MerlinVerify.Rules.W8A8ArithmeticModel

open MerlinVerify
open Float.Model (Format UnpackedFloat)

def rounded (format : Format) : UnpackedFloat → UnpackedFloat
  | .finite sign mantissa exponent _ => UnpackedFloat.round format sign mantissa exponent
  | other => other

def widen (x : FBits .f32) : FBits .f64 :=
  UnpackedFloat.pack Format.binary64 (rounded Format.binary64 (W8A8IEEE32.unpack x))

def narrow (x : FBits .f64) : FBits .f32 :=
  W8A8IEEE32.pack (rounded Format.binary32 (UnpackedFloat.unpack Format.binary64 x))

def divide64 (x y : FBits .f64) : FBits .f64 :=
  UnpackedFloat.pack Format.binary64
    (UnpackedFloat.div Format.binary64
      (UnpackedFloat.unpack Format.binary64 x) (UnpackedFloat.unpack Format.binary64 y))

def inverse (scale : FBits .f32) : FBits .f32 :=
  narrow (divide64 0x3ff0000000000000#64 (widen scale))

def subtract32 (x y : FBits .f32) : FBits .f32 :=
  W8A8IEEE32.pack (UnpackedFloat.sub Format.binary32 (W8A8IEEE32.unpack x) (W8A8IEEE32.unpack y))

def signedInteger32 {width : Nat} (x : BitVec width) : FBits .f32 :=
  W8A8IEEE32.pack (UnpackedFloat.ofInt Format.binary32 x.toInt)

structure Agreement (base : FloatModel) : Prop where
  -- Arithmetic NaN sign/payload choices differ across implementations. Only non-NaN
  -- concrete results are used by this candidate, so exact NaN-bit agreement is unnecessary.
  multiply32 : ∀ x y, W8A8IEEE32.NonNaN (W8A8IEEE32.mul x y) →
    base.op (Spec.Dequant.defaultKey "arith.mulf") .f32 [x, y] = W8A8IEEE32.mul x y
  add32 : ∀ x y, W8A8IEEE32.NonNaN (W8A8IEEE32.add x y) →
    base.op (Spec.Dequant.defaultKey "arith.addf") .f32 [x, y] = W8A8IEEE32.add x y
  divide32 : ∀ x y, W8A8IEEE32.NonNaN (W8A8IEEE32.div x y) →
    base.op (Spec.Dequant.defaultKey "arith.divf") .f32 [x, y] = W8A8IEEE32.div x y
  rsqrt32 : ∀ x, W8A8IEEE32.NonNaN (W8A8IEEE32.rsqrt x) →
    base.op (Spec.Dequant.defaultKey "math.rsqrt") .f32 [x] = W8A8IEEE32.rsqrt x
  -- These three primitive identities follow the exact source reciprocal recipe, restricted
  -- to admissible f32 scales. They do not assert any resulting finite/nonzero property.
  widen32 : ∀ scale, W8A8SourceQuantize.scaleAdmissible scale = true →
    base.convert (Spec.Dequant.defaultKey "arith.extf") .f32 .f64 scale = widen scale
  divide64 : ∀ scale, W8A8SourceQuantize.scaleAdmissible scale = true →
    base.op (Spec.Dequant.defaultKey "arith.divf") .f64 [0x3ff0000000000000#64, widen scale] =
      divide64 0x3ff0000000000000#64 (widen scale)
  narrow64 : ∀ scale, W8A8SourceQuantize.scaleAdmissible scale = true →
    base.convert (Spec.Dequant.defaultKey "arith.truncf") .f64 .f32
      (W8A8ArithmeticModel.divide64 0x3ff0000000000000#64 (widen scale)) = inverse scale
  subtract32 : ∀ x y, W8A8IEEE32.NonNaN (subtract32 x y) →
    base.op (Spec.Dequant.defaultKey "arith.subf") .f32 [x, y] = subtract32 x y
  signedInteger32 : ∀ width x, base.ofInt (Spec.Dequant.defaultKey "arith.sitofp") width .f32 x = signedInteger32 x

theorem inverse_agrees {base : FloatModel} (agreement : Agreement base) (scale : FBits .f32)
    (admissible : W8A8SourceQuantize.scaleAdmissible scale = true) :
    W8A8SourceQuantize.inverse base scale = inverse scale := by
  simp only [W8A8SourceQuantize.inverse, agreement.widen32 scale admissible,
    agreement.divide64 scale admissible, agreement.narrow64 scale admissible]

/-- Literal calibration scales are checked by the actual concrete binary64-reciprocal/f32
narrowing computation. A tiny scale yielding infinity or a zero inverse is not admitted. -/
def inverseCheck (scale : FBits .f32) : Bool :=
  W8A8SourceQuantize.scaleAdmissible scale &&
    decide (W8A8IEEE32.Finite (inverse scale)) && decide (W8A8IEEE32.Nonzero (inverse scale))

theorem inverseCheck_sound {base : FloatModel} (agreement : Agreement base) {scale : FBits .f32}
    (checked : inverseCheck scale = true) :
    W8A8SourceQuantize.scaleAdmissible scale = true ∧
    W8A8IEEE32.Finite (W8A8SourceQuantize.inverse base scale) ∧
    W8A8IEEE32.Nonzero (W8A8SourceQuantize.inverse base scale) := by
  simp only [inverseCheck, Bool.and_eq_true, decide_eq_true_eq] at checked
  rw [inverse_agrees agreement scale checked.1.1]
  exact ⟨checked.1.1, checked.1.2, checked.2⟩

/-- Non-NaN inputs, including either infinity, have a nonpoison source conversion when the
actual inverse is finite/nonzero. Overflow of the product is permitted. -/
theorem shifted_nonNaN {base : FloatModel} (agreement : Agreement base)
    (laws : W8A8FiniteLaws.Laws base) {width : Nat} (allowed : width = 32 ∨ width = 64)
    {x scale : FBits .f32} {zero : BitVec width} (input : W8A8IEEE32.NonNaN x)
    (checked : inverseCheck scale = true) :
    W8A8ByteFloat.isNaN (W8A8SourceQuantize.shifted width base x scale zero) = false := by
  obtain ⟨_, finite, nonzero⟩ := inverseCheck_sound agreement checked
  have product : W8A8ByteFloat.isNaN
      (base.op (Spec.Dequant.defaultKey "arith.mulf") .f32 [x, W8A8SourceQuantize.inverse base scale]) = false := by
    have nonNaN := W8A8IEEE32.mul_nonNaN_nonzeroFinite input finite nonzero
    rw [agreement.multiply32 _ _ nonNaN]
    exact nonNaN
  exact laws.add _ _ (laws.round _ product) (laws.integer width allowed zero)

theorem scalar_defined {base : FloatModel} (agreement : Agreement base) (laws : W8A8FiniteLaws.Laws base)
    {width : Nat} (allowed : width = 32 ∨ width = 64) {x scale : FBits .f32} {zero : BitVec width}
    (input : W8A8IEEE32.NonNaN x) (checked : inverseCheck scale = true) :
    W8A8SourceQuantize.scalar width (W8A8ByteModel.extend base) (some x) scale zero ≠ .poison :=
  W8A8SourceQuantize.scalar_defined width base x scale zero (shifted_nonNaN agreement laws allowed input checked)

#print axioms inverse_agrees
#print axioms inverseCheck_sound
#print axioms shifted_nonNaN
#print axioms scalar_defined

end MerlinVerify.Rules.W8A8ArithmeticModel
