/- Concrete binary32 classification lemmas for the numerical-domain candidate. Arithmetic
uses Lean's executable IEEE unpacked model. This does not identify an arbitrary FloatModel
with IEEE, change the frozen semantics, or admit the remaining source/dynamic laws. -/
import Init.Data.Float.Model.Float32
import MerlinVerify.Rules.W8A8DynamicLaws

namespace MerlinVerify.Rules.W8A8IEEE32

open MerlinVerify
open Float.Model (Format UnpackedFloat)
open Float.Model.UnpackedFloat

abbrev format := Format.binary32
abbrev unpack (x : FBits .f32) := UnpackedFloat.unpack format x
abbrev pack (x : UnpackedFloat) : FBits .f32 := UnpackedFloat.pack format x

abbrev NonNaN (x : FBits .f32) : Prop := W8A8ByteFloat.isNaN x = false
abbrev Finite (x : FBits .f32) : Prop := W8A8FiniteLaws.Finite x
abbrev Nonzero (x : FBits .f32) : Prop := W8A8ByteFloat.magnitudeWord x ≠ 0
abbrev Nonnegative (x : FBits .f32) : Prop := NonNaN x ∧ W8A8ByteFloat.negative x = false
abbrev Positive (x : FBits .f32) : Prop := Nonnegative x ∧ Nonzero x

def mul (x y : FBits .f32) : FBits .f32 := pack (UnpackedFloat.mul format (unpack x) (unpack y))
def add (x y : FBits .f32) : FBits .f32 := pack (UnpackedFloat.add format (unpack x) (unpack y))
def div (x y : FBits .f32) : FBits .f32 := pack (UnpackedFloat.div format (unpack x) (unpack y))
def sqrt (x : FBits .f32) : FBits .f32 := pack (UnpackedFloat.sqrt format (unpack x))
def rsqrt (x : FBits .f32) : FBits .f32 := div 0x3f800000#32 (sqrt x)

theorem exponent_eq_unpack (x : FBits .f32) :
    W8A8ByteFloat.exponent x = (unpackExponent (spec := format) x).toNat := by
  have bits := BitVec.extractLsb'_toNat 23 8 x
  simp only [Nat.shiftRight_eq_div_pow] at bits
  change (x.toNat % 2^31) / 2^23 = (BitVec.extractLsb' 23 8 x).toNat
  rw [bits]
  omega

theorem fraction_eq_unpack (x : FBits .f32) :
    W8A8ByteFloat.fraction x = (unpackMantissa (spec := format) x).toNat := by
  have bits := BitVec.extractLsb'_toNat 0 23 x
  simp only [Nat.shiftRight_eq_div_pow] at bits
  change (x.toNat % 2^31) % 2^23 = (BitVec.extractLsb' 0 23 x).toNat
  rw [bits]
  omega

theorem exponent_packComponents (s : Sign) (e : BitVec 8) (m : BitVec 23) :
    W8A8ByteFloat.exponent (packComponents format s e m) = e.toNat := by
  exact (exponent_eq_unpack _).trans (congrArg BitVec.toNat unpackExponent_packComponents)

theorem fraction_packComponents (s : Sign) (e : BitVec 8) (m : BitVec 23) :
    W8A8ByteFloat.fraction (packComponents format s e m) = m.toNat := by
  exact (fraction_eq_unpack _).trans (congrArg BitVec.toNat unpackMantissa_packComponents)

theorem pack_notNaN {x : UnpackedFloat} (h : x ≠ .notANumber) : NonNaN (pack x) := by
  cases x with
  | notANumber => exact False.elim (h rfl)
  | infinity s =>
    simp [NonNaN, pack, UnpackedFloat.pack, packedInfinity, W8A8ByteFloat.isNaN,
      exponent_packComponents, fraction_packComponents]
  | zero s =>
    simp [NonNaN, pack, UnpackedFloat.pack, packedZero, W8A8ByteFloat.isNaN,
      exponent_packComponents, fraction_packComponents]
  | finite s m e positive =>
    simp only [pack, UnpackedFloat.pack]
    split
    · simp [NonNaN, packedInfinity, W8A8ByteFloat.isNaN,
        exponent_packComponents, fraction_packComponents]
    · next bound =>
      split
      · have upper : (e + format.exponentBias + format.mantissaBitsWithoutImplicit).toNat < 255 := by
          simp only [format, Format.binary32] at bound ⊢
          omega
        have different : (BitVec.ofNat 8 (e + format.exponentBias + format.mantissaBitsWithoutImplicit).toNat).toNat ≠ 255 := by
          rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt (by omega)]
          omega
        change NonNaN (packComponents format s
          (BitVec.ofNat 8 (e + format.exponentBias + format.mantissaBitsWithoutImplicit).toNat)
          (BitVec.ofNat 23 m))
        simp only [NonNaN, W8A8ByteFloat.isNaN, exponent_packComponents]
        simp only [different, false_and, decide_false]
      · simp [NonNaN, W8A8ByteFloat.isNaN, exponent_packComponents]

theorem unpack_notNaN {x : FBits .f32} (h : NonNaN x) : unpack x ≠ .notANumber := by
  intro equal
  unfold unpack UnpackedFloat.unpack at equal
  dsimp only at equal
  split at equal
  · next exponent =>
    split at equal
    · cases equal
    · next mantissa =>
      have e : W8A8ByteFloat.exponent x = 255 := by
        exact (exponent_eq_unpack x).trans (congrArg BitVec.toNat exponent)
      have m : W8A8ByteFloat.fraction x ≠ 0 := by
        intro zero
        apply mantissa
        apply BitVec.eq_of_toNat_eq
        exact (fraction_eq_unpack x).symm.trans zero
      simp [NonNaN, W8A8ByteFloat.isNaN, e, m] at h
  · split at equal
    · split at equal <;> cases equal
    · cases equal

theorem unpack_notInf {x : FBits .f32} (h : Finite x) (s : Sign) : unpack x ≠ .infinity s := by
  intro equal
  unfold unpack UnpackedFloat.unpack at equal
  dsimp only at equal
  split at equal
  · next exponent =>
    have e : W8A8ByteFloat.exponent x = 255 :=
      (exponent_eq_unpack x).trans (congrArg BitVec.toNat exponent)
    exact (Nat.ne_of_lt h) e
  · split at equal
    · split at equal <;> cases equal
    · cases equal

theorem unpack_notZero {x : FBits .f32} (h : Nonzero x) (s : Sign) : unpack x ≠ .zero s := by
  intro equal
  unfold unpack UnpackedFloat.unpack at equal
  dsimp only at equal
  split at equal
  · split at equal <;> cases equal
  · split at equal
    · next exponent =>
      split at equal
      · next mantissa =>
        have e : W8A8ByteFloat.exponent x = 0 :=
          (exponent_eq_unpack x).trans (congrArg BitVec.toNat exponent)
        have m : W8A8ByteFloat.fraction x = 0 :=
          (fraction_eq_unpack x).trans (congrArg BitVec.toNat mantissa)
        apply h
        unfold W8A8ByteFloat.exponent at e
        unfold W8A8ByteFloat.fraction at m
        omega
      · cases equal
    · cases equal

theorem finite_notNaN {x : FBits .f32} (h : Finite x) : NonNaN x :=
  W8A8DynamicLaws.finite_nonNaN h

theorem round_notNaN (s : Sign) (m : Nat) (e : Int) (a : Accuracy) :
    roundWithAccuracy format s m e a ≠ .notANumber := by
  unfold roundWithAccuracy
  dsimp only
  split <;> intro h <;> cases h

/-- Finite operands may overflow to infinity, but their product is not NaN. -/
theorem mul_finite_notNaN {x y : FBits .f32} (hx : Finite x) (hy : Finite y) : NonNaN (mul x y) := by
  apply pack_notNaN
  have nx := unpack_notNaN (finite_notNaN hx)
  have ny := unpack_notNaN (finite_notNaN hy)
  have ix := unpack_notInf hx
  have iy := unpack_notInf hy
  cases dx : unpack x <;> cases dy : unpack y <;>
    simp_all [UnpackedFloat.mul, round_notNaN]

theorem mul_nonNaN_nonzeroFinite {x y : FBits .f32}
    (hx : NonNaN x) (hy : Finite y) (nz : Nonzero y) : NonNaN (mul x y) := by
  apply pack_notNaN
  have nx := unpack_notNaN hx
  have ny := unpack_notNaN (finite_notNaN hy)
  have iy := unpack_notInf hy
  have zy := unpack_notZero nz
  cases dx : unpack x <;> cases dy : unpack y <;>
    simp_all [UnpackedFloat.mul, round_notNaN]

theorem mul_nonzeroFinite_nonNaN {x y : FBits .f32}
    (hx : Finite x) (nz : Nonzero x) (hy : NonNaN y) : NonNaN (mul x y) := by
  apply pack_notNaN
  have nx := unpack_notNaN (finite_notNaN hx)
  have ny := unpack_notNaN hy
  have ix := unpack_notInf hx
  have zx := unpack_notZero nz
  cases dx : unpack x <;> cases dy : unpack y <;>
    simp_all [UnpackedFloat.mul, round_notNaN]

theorem rounded_notNaN (s : Sign) (m : Nat) (e : Int) :
    UnpackedFloat.round format s m e ≠ .notANumber := by
  unfold UnpackedFloat.round
  dsimp only
  apply round_notNaN

theorem normalize_notNaN (m : Int) (e : Int) (s : Sign) :
    normalize format m e s ≠ .notANumber := by
  unfold normalize
  split <;> simp [rounded_notNaN]

theorem add_nonNaN_finite {x y : FBits .f32} (hx : NonNaN x) (hy : Finite y) :
    NonNaN (add x y) := by
  apply pack_notNaN
  have nx := unpack_notNaN hx
  have ny := unpack_notNaN (finite_notNaN hy)
  have iy := unpack_notInf hy
  cases dx : unpack x <;> cases dy : unpack y <;>
    simp_all [UnpackedFloat.add, normalize_notNaN]
  all_goals split <;> simp

theorem div_nonNaN_nonzeroFinite {x y : FBits .f32}
    (hx : NonNaN x) (hy : Finite y) (nz : Nonzero y) : NonNaN (div x y) := by
  apply pack_notNaN
  have nx := unpack_notNaN hx
  have ny := unpack_notNaN (finite_notNaN hy)
  have iy := unpack_notInf hy
  have zy := unpack_notZero nz
  cases dx : unpack x <;> cases dy : unpack y <;>
    simp_all [UnpackedFloat.div, round_notNaN]

theorem append_toNat {a b : Nat} (x : BitVec a) (y : BitVec b) :
    (x ++ y).toNat = x.toNat * 2^b + y.toNat := by
  rw [BitVec.toNat_append, ← Nat.shiftLeft_add_eq_or_of_lt y.isLt, Nat.shiftLeft_eq]

theorem packComponents_positive (e : BitVec 8) (m : BitVec 23) :
    W8A8ByteFloat.negative (packComponents format .positive e m) = false := by
  change decide (2^31 ≤ ((0#1 ++ e) ++ m).toNat) = false
  rw [append_toNat, append_toNat]
  have he := e.isLt
  have hm := m.isLt
  simp only [BitVec.toNat_ofNat, Nat.zero_mod, Nat.zero_mul, Nat.zero_add, decide_eq_false_iff_not]
  omega

def PositiveSign : UnpackedFloat → Prop
  | .notANumber => False
  | .infinity s | .zero s | .finite s _ _ _ => s = .positive

theorem positiveSign_notNaN {x : UnpackedFloat} (h : PositiveSign x) : x ≠ .notANumber := by
  cases x <;> simp_all [PositiveSign]

theorem pack_nonnegative {x : UnpackedFloat} (h : PositiveSign x) : Nonnegative (pack x) := by
  refine ⟨pack_notNaN (positiveSign_notNaN h), ?_⟩
  cases x with
  | notANumber => cases h
  | infinity s =>
    cases h
    exact packComponents_positive _ _
  | zero s =>
    cases h
    exact packComponents_positive _ _
  | finite s m e positive =>
    cases h
    simp only [pack, UnpackedFloat.pack]
    split
    · exact packComponents_positive _ _
    · split <;> exact packComponents_positive _ _

theorem unpack_positiveSign {x : FBits .f32} (h : Nonnegative x) : PositiveSign (unpack x) := by
  have signBits : unpackSign (spec := format) x = 0#1 := by
    apply BitVec.eq_of_toNat_eq
    have bits := BitVec.extractLsb'_toNat 31 1 x
    simp only [Nat.shiftRight_eq_div_pow] at bits
    change (BitVec.extractLsb' 31 1 x).toNat = 0
    rw [bits]
    have signed := h.2
    simp only [W8A8ByteFloat.negative, decide_eq_false_iff_not] at signed
    omega
  have signEq : Sign.ofBitVec (unpackSign (spec := format) x) = .positive := by
    rw [signBits]; rfl
  have notNaN := unpack_notNaN h.1
  unfold unpack UnpackedFloat.unpack
  dsimp only
  rw [signEq]
  split
  · next exponent =>
    split
    · rfl
    · next mantissa =>
      have equal : unpack x = .notANumber := by
        simp [unpack, UnpackedFloat.unpack, exponent, mantissa]
      exact False.elim (notNaN equal)
  · split
    · split <;> rfl
    · rfl

theorem round_positiveSign (m : Nat) (e : Int) (a : Accuracy) :
    PositiveSign (roundWithAccuracy format .positive m e a) := by
  unfold roundWithAccuracy
  dsimp only
  split <;> rfl

theorem rounded_positiveSign (m : Nat) (e : Int) :
    PositiveSign (UnpackedFloat.round format .positive m e) := by
  unfold UnpackedFloat.round
  dsimp only
  apply round_positiveSign

theorem square_nonnegative {x : FBits .f32} (h : NonNaN x) : Nonnegative (mul x x) := by
  apply pack_nonnegative
  have nx := unpack_notNaN h
  cases d : unpack x with
  | notANumber => exact False.elim (nx d)
  | infinity s => cases s <;> rfl
  | zero s => cases s <;> rfl
  | finite s m e positive =>
    cases s <;> exact round_positiveSign _ _ _

theorem normalize_positiveSign {m : Int} (h : 0 ≤ m) (e : Int) :
    PositiveSign (normalize format m e .positive) := by
  unfold normalize
  split
  · next negative =>
    have less : m < 0 := by simpa only [Int.compare_eq_lt] using negative
    omega
  · rfl
  · apply rounded_positiveSign

theorem add_nonnegative {x y : FBits .f32} (hx : Nonnegative x) (hy : Nonnegative y) :
    Nonnegative (add x y) := by
  apply pack_nonnegative
  have px := unpack_positiveSign hx
  have py := unpack_positiveSign hy
  cases dx : unpack x <;> cases dy : unpack y <;>
    simp_all only [PositiveSign]
  all_goals simp only [UnpackedFloat.add, Sign.apply]
  all_goals first | rfl | apply normalize_positiveSign; omega

theorem div_nonnegative {x y : FBits .f32} (hx : Nonnegative x) (hy : Positive y) (finite : Finite y) :
    Nonnegative (div x y) := by
  apply pack_nonnegative
  have px := unpack_positiveSign hx
  have py := unpack_positiveSign hy.1
  have iy := unpack_notInf finite
  have zy := unpack_notZero hy.2
  cases dx : unpack x <;> cases dy : unpack y <;>
    simp_all only [PositiveSign]
  all_goals simp_all [UnpackedFloat.div, show Sign.positive / Sign.positive = Sign.positive from rfl]
  all_goals exact round_positiveSign _ _ _

theorem sqrt_nonnegative {x : FBits .f32} (h : Nonnegative x) : Nonnegative (sqrt x) := by
  apply pack_nonnegative
  have px := unpack_positiveSign h
  cases dx : unpack x <;> simp_all only [PositiveSign]
  all_goals first | rfl | apply round_positiveSign

theorem unpack_one : unpack 0x3f800000#32 = .finite .positive 8388608 (-23) (by decide) := by rfl

/-- Nonnegative zero is allowed: reciprocal square root of +0 is +infinity. This is a
non-NaN fact, deliberately not a false finite-output guarantee. -/
theorem rsqrt_nonnegative {x : FBits .f32} (h : Nonnegative x) : Nonnegative (rsqrt x) := by
  apply pack_nonnegative
  have px := unpack_positiveSign (sqrt_nonnegative h)
  change PositiveSign (UnpackedFloat.div format (unpack 0x3f800000#32) (unpack (sqrt x)))
  rw [unpack_one]
  cases dx : unpack (sqrt x) <;> simp_all only [PositiveSign]
  all_goals first | rfl | apply round_positiveSign

#print axioms mul_finite_notNaN
#print axioms mul_nonNaN_nonzeroFinite
#print axioms square_nonnegative
#print axioms add_nonnegative
#print axioms div_nonnegative
#print axioms rsqrt_nonnegative

end MerlinVerify.Rules.W8A8IEEE32
