/- An explicit arithmetic base for joint W8A8 numerical-law proofs. Every constrained
operation uses pinned unpacked IEEE arithmetic, bitwise absolute value, or explicit half-even
integer rounding. Remaining operations come from an explicitly supplied fallback; the
standalone fallback uses nonfinite/poison sentinels and is not a full native IEEE model. -/
import MerlinVerify.Rules.W8A8ArithmeticModel

namespace MerlinVerify.Rules.W8A8IEEEPrimitives

open MerlinVerify
open Float.Model (Format UnpackedFloat)

abbrev key := Spec.Dequant.defaultKey

def absolute (x : FBits .f32) : FBits .f32 := BitVec.ofNat 32 (W8A8ByteFloat.magnitudeWord x)

/-- Same half-even quotient/remainder recipe used by the frontend's explicit execution
instance. Signed zero is preserved. This defines a logical primitive, not native fidelity. -/
def roundeven (x : FBits .f32) : FBits .f32 :=
  match W8A8IEEE32.unpack x with
  | .notANumber => UnpackedFloat.packedNaN Format.binary32
  | .infinity _ | .zero _ => x
  | .finite s m e _ =>
    if 0 ≤ e then x else
    let d := 2^(-e).toNat
    let q := m / d
    let r := m % d
    let n := q + if d < 2*r ∨ (d = 2*r ∧ q % 2 = 1) then 1 else 0
    if n = 0 then W8A8IEEE32.pack (.zero s)
    else W8A8IEEE32.pack (UnpackedFloat.ofInt Format.binary32 (s.apply n))

def model (fallback : FloatModel) : FloatModel :=
  { fallback with
    op := fun k fmt values =>
      match fmt, values with
      | .f32, [x, y] =>
        if k = key "arith.mulf" then W8A8IEEE32.mul x y
        else if k = key "arith.addf" then W8A8IEEE32.add x y
        else if k = key "arith.subf" then W8A8ArithmeticModel.subtract32 x y
        else if k = key "arith.divf" then W8A8IEEE32.div x y
        else fallback.op k .f32 [x, y]
      | .f32, [x] =>
        if k = key "math.rsqrt" then W8A8IEEE32.rsqrt x
        else if k = key "math.absf" then absolute x
        else if k = key "math.roundeven" then roundeven x
        else fallback.op k .f32 [x]
      | .f64, [x, y] =>
        if k = key "arith.divf" then W8A8ArithmeticModel.divide64 x y
        else fallback.op k .f64 [x, y]
      | fmt, values => fallback.op k fmt values
    convert := fun k src dst value => match src, dst with
      | .f32, .f64 => if k = key "arith.extf" then W8A8ArithmeticModel.widen value
          else fallback.convert k .f32 .f64 value
      | .f64, .f32 => if k = key "arith.truncf" then W8A8ArithmeticModel.narrow value
          else fallback.convert k .f64 .f32 value
      | src, dst => fallback.convert k src dst value
    ofInt := fun k width fmt value => match fmt with
      | .f32 => if k = key "arith.sitofp" then W8A8ArithmeticModel.signedInteger32 value
          else fallback.ofInt k width .f32 value
      | fmt => fallback.ofInt k width fmt value }

/-- Unqualified primitives remain visibly unsupported. No numerical law is discharged by
these sentinels: all operations constrained by Agreement/Laws are overridden above. -/
def unsupported : FloatModel where
  op _ _ _ := -1
  cmp _ _ _ _ := false
  convert _ _ _ _ := -1
  toInt _ _ _ _ := none
  ofInt _ _ _ _ := -1

theorem agreement (fallback : FloatModel) : W8A8ArithmeticModel.Agreement (model fallback) where
  multiply32 _ _ _ := rfl
  add32 _ _ _ := rfl
  divide32 _ _ _ := rfl
  rsqrt32 _ _ := rfl
  widen32 _ _ := rfl
  divide64 _ _ := rfl
  narrow64 _ _ := rfl
  subtract32 _ _ _ := rfl
  signedInteger32 _ _ := rfl

theorem absolute_finite {x : FBits .f32} (finite : W8A8FiniteLaws.Finite x) :
    W8A8FiniteLaws.Finite (absolute x) := by
  have bound : W8A8ByteFloat.magnitudeWord x < 2^31 := Nat.mod_lt _ (by decide)
  unfold W8A8FiniteLaws.Finite W8A8ByteFloat.exponent absolute W8A8ByteFloat.magnitudeWord at *
  change (BitVec.ofNat 32 (x.toNat % 2^31)).toNat % 2^31 / 2^23 < 255
  have below32 : x.toNat % 2^31 < 2^32 := by omega
  simp only [BitVec.toNat_ofNat]
  rw [Nat.mod_eq_of_lt below32, Nat.mod_eq_of_lt bound]
  exact finite

theorem roundeven_nonNaN {x : FBits .f32} (nonNaN : W8A8IEEE32.NonNaN x) :
    W8A8IEEE32.NonNaN (roundeven x) := by
  have decoded := W8A8IEEE32.unpack_notNaN nonNaN
  have zero : ∀ s, W8A8IEEE32.NonNaN (W8A8IEEE32.pack (.zero s)) := by
    intro s
    apply W8A8IEEE32.pack_notNaN
    intro equal; cases equal
  have integer : ∀ n, W8A8IEEE32.NonNaN
      (W8A8IEEE32.pack (UnpackedFloat.ofInt Format.binary32 n)) := by
    intro n
    exact W8A8IEEE32.pack_notNaN (W8A8IEEE32.normalize_notNaN n 0 .positive)
  cases encoded : W8A8IEEE32.unpack x with
  | notANumber => exact False.elim (decoded encoded)
  | infinity s => simpa only [roundeven, encoded] using nonNaN
  | zero s => simpa only [roundeven, encoded] using nonNaN
  | finite s m e positive =>
    simp only [roundeven, encoded]
    by_cases integral : 0 ≤ e
    · simpa only [ite_eq_left integral] using nonNaN
    · simp only [ite_eq_right integral]
      let d := 2^(-e).toNat
      let n := m / d + if d < 2*(m % d) ∨ (d = 2*(m % d) ∧ (m / d) % 2 = 1) then 1 else 0
      change W8A8IEEE32.NonNaN
        (if n = 0 then W8A8IEEE32.pack (.zero s)
          else W8A8IEEE32.pack (UnpackedFloat.ofInt Format.binary32 (s.apply n)))
      split
      · exact zero s
      · exact integer (s.apply n)

theorem multiplication (fallback : FloatModel) {x y : FBits .f32}
    (hx : W8A8FiniteLaws.Finite x) (hy : W8A8FiniteLaws.Finite y) :
    W8A8ByteFloat.isNaN ((model fallback).op (key "arith.mulf") .f32 [x, y]) = false :=
  W8A8IEEE32.mul_finite_notNaN hx hy

theorem addition (fallback : FloatModel) {x y : FBits .f32}
    (hx : W8A8IEEE32.NonNaN x) (hy : W8A8FiniteLaws.Finite y) :
    W8A8ByteFloat.isNaN ((model fallback).op (key "arith.addf") .f32 [x, y]) = false :=
  W8A8IEEE32.add_nonNaN_finite hx hy

#print axioms agreement
#print axioms absolute_finite
#print axioms roundeven_nonNaN

end MerlinVerify.Rules.W8A8IEEEPrimitives
