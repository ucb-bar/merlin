/- Provisional numerical premises for source quantization. Each field constrains one primitive
floating operation. No instance, axiom, IEEE fidelity claim or reviewed policy is introduced.
The normal-scale guard is conservative and excludes overflowing reciprocals of tiny scales. -/
import MerlinVerify.Rules.W8A8SourceQuantize

namespace MerlinVerify.Rules.W8A8FiniteLaws

open MerlinVerify MerlinVerify.Sem.Ops

abbrev key := Spec.Dequant.defaultKey

abbrev Finite (x : FBits .f32) : Prop := W8A8ByteFloat.exponent x < 255

def scaleCheck (scale : FBits .f32) : Bool :=
  W8A8SourceQuantize.scaleAdmissible scale && decide (0 < W8A8ByteFloat.exponent scale)

/-- The f64 interval [2^-126, +infinity) with +infinity excluded. These unsigned bit bounds
also exclude negative values and NaNs; they are not host floating-point comparisons. -/
def WideScale (x : FBits .f64) : Prop :=
  0x3810000000000000 ≤ x.toNat ∧ x.toNat < 0x7ff0000000000000

/-- Nonnegative finite f64 values at most 2^126, small enough to narrow to finite f32. -/
def Narrowable (x : FBits .f64) : Prop := x.toNat ≤ 0x47d0000000000000

structure Laws (M : FloatModel) : Prop where
  widen : ∀ scale, scaleCheck scale = true →
    WideScale (M.convert (key "arith.extf") .f32 .f64 scale)
  reciprocal : ∀ divisor, WideScale divisor →
    Narrowable (M.op (key "arith.divf") .f64 [0x3ff0000000000000#64, divisor])
  narrow : ∀ value, Narrowable value → Finite (M.convert (key "arith.truncf") .f64 .f32 value)
  integer : ∀ width, (width = 32 ∨ width = 64) → ∀ value : BitVec width,
    Finite (M.ofInt (key "arith.sitofp") width .f32 value)
  multiply : ∀ x y, Finite x → Finite y →
    W8A8ByteFloat.isNaN (M.op (key "arith.mulf") .f32 [x, y]) = false
  round : ∀ x, W8A8ByteFloat.isNaN x = false →
    W8A8ByteFloat.isNaN (M.op (key "math.roundeven") .f32 [x]) = false
  add : ∀ x y, W8A8ByteFloat.isNaN x = false → Finite y →
    W8A8ByteFloat.isNaN (M.op (key "arith.addf") .f32 [x, y]) = false

theorem scaleCheck_admissible {scale : FBits .f32} (checked : scaleCheck scale = true) :
    W8A8SourceQuantize.scaleAdmissible scale = true := by
  simp only [scaleCheck, Bool.and_eq_true] at checked
  exact checked.1

theorem inverse_finite {base : FloatModel} (laws : Laws base) {scale : FBits .f32}
    (checked : scaleCheck scale = true) : Finite (W8A8SourceQuantize.inverse base scale) :=
  laws.narrow _ (laws.reciprocal _ (laws.widen scale checked))

/-- Finite inputs and the checked scale class imply the exact intermediate non-NaN obligation
used by the byte-conversion theorem. Overflow to infinity is allowed; finiteness of a product
is deliberately not assumed. -/
theorem shifted_nonNaN {base : FloatModel} (laws : Laws base) {width : Nat}
    (allowed : width = 32 ∨ width = 64) {x scale : FBits .f32} {zero : BitVec width}
    (finite : Finite x) (checked : scaleCheck scale = true) :
    W8A8ByteFloat.isNaN (W8A8SourceQuantize.shifted width base x scale zero) = false :=
  laws.add _ _ (laws.round _ (laws.multiply _ _ finite (inverse_finite laws checked)))
    (laws.integer width allowed zero)

/-- The current abstract FloatModel interface does not entail these premises. This witness
prevents the conditional theorem from being promoted by treating the premises as automatic. -/
theorem separatingModel_not_lawful : ¬ Laws W8A8Scale.separatingModel := by
  intro laws
  have wide := laws.widen 0x3f800000#32 (by decide)
  change 0x3810000000000000 ≤ 0 ∧ 0 < 0x7ff0000000000000 at wide
  omega

#print axioms scaleCheck_admissible
#print axioms inverse_finite
#print axioms shifted_nonNaN
#print axioms separatingModel_not_lawful

end MerlinVerify.Rules.W8A8FiniteLaws
