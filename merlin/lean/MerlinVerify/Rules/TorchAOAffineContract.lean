/- PROVISIONAL reference contract, not an admitted Spec declaration.
Reference: TorchAO 0.15.0/0.16.0 quant_api._int8_symm_per_token_reduced_range_quant,
quant_primitives._choose_qparams_affine and _quantize_affine_no_dtype_cast.
This proposal is implementation-aware and requires an independent specification review.
It is applicable only when the actual capture receipt binds this dynamic per-token recipe;
the selected Gemmini static recipes do not establish that premise. -/
import MerlinVerify.Rules.DequantScalar

namespace MerlinVerify.Rules.TorchAOAffineContract
open MerlinVerify

def key (name : String) : FloatOpKey :=
  {op := name, modifiers := "fm1:0;rm4:none;pred4:none"}

def f (M : FloatModel) (name : String) (xs : List (FBits .f32)) : FBits .f32 :=
  M.op (key name) .f32 xs

/-- MLIR conversion extension. Agreement with Torch's integer cast requires a finite,
in-range result domain; Torch's observed NaN-to-int behavior is not an MLIR poison law. -/
def asInt (M : FloatModel) (x : FBits .f32) : Val :=
  match M.toInt (key "arith.fptosi") .f32 8 x with
  | none => .poison
  | some n => .some (.bits 8 n)

def clamp (M : FloatModel) (x : FBits .f32) : FBits .f32 :=
  f M "arith.minimumf" [f M "arith.maximumf" [x, 0xc2fe0000], 0x42fe0000]

/-- The reference includes the integer-domain zero point added AFTER half-even rounding.
No reassociation, division substitution or zero-addition law is assumed here. -/
def quantize (M : FloatModel) (x scale : FBits .f32) : Val :=
  let reciprocal := f M "arith.divf" [0x3f800000, scale]
  let product := f M "arith.mulf" [x, reciprocal]
  let rounded := f M "math.roundeven" [product]
  let shifted := f M "arith.addf" [rounded, 0]
  asInt M (clamp M shifted)

/-- The exact expression implemented without an explicit zero-point addition. Its equality
to the versioned reference requires the separately named elision obligation below. -/
def quantizeWithoutZero (M : FloatModel) (x scale : FBits .f32) : Val :=
  let reciprocal := f M "arith.divf" [0x3f800000, scale]
  let product := f M "arith.mulf" [x, reciprocal]
  let rounded := f M "math.roundeven" [product]
  asInt M (clamp M rounded)

/-- Exact remaining IEEE-model obligation. It concerns the final integer/poison result,
allowing the reference's zero addition to change a negative zero's sign. -/
def ZeroPointElisionCorrect (M : FloatModel) : Prop :=
  ∀ x scale, quantize M x scale = quantizeWithoutZero M x scale

def floorScale (M : FloatModel) (maxAbs : FBits .f32) : FBits .f32 :=
  f M "arith.maximumf" [f M "arith.divf" [maxAbs, 0x42fe0000], 0x3727c5ac]

/-- The source reference explicitly reduces min/max, includes zero, negates the lower bound,
and divides by 127 before applying the f32 1e-5 floor. Empty reduction admission is separate. -/
def choose (M : FloatModel) (xs : List (FBits .f32)) : FBits .f32 :=
  let lo := xs.foldl (fun a x => f M "arith.minimumf" [a, x]) 0x7f800000
  let hi := xs.foldl (fun a x => f M "arith.maximumf" [a, x]) 0xff800000
  let lo := f M "arith.minimumf" [lo, 0]
  let hi := f M "arith.maximumf" [hi, 0]
  floorScale M (f M "arith.maximumf" [f M "arith.negf" [lo], hi])

def chooseAbsFold (M : FloatModel) (xs : List (FBits .f32)) : FBits .f32 :=
  floorScale M (xs.foldl (fun a x => f M "arith.maximumf" [a, f M "math.absf" [x]]) 0)

/-- Exact remaining reduction obligation, including NaN, infinity and signed-zero behavior.
The reference's two reductions cannot be equated to abs/max by uninterpreted-float algebra. -/
def AbsReductionCorrect (M : FloatModel) : Prop :=
  ∀ xs, xs ≠ [] → choose M xs = chooseAbsFold M xs

/-- A countermodel documents why no theorem for every frozen FloatModel can drop the +zero.
This is a mathematical witness, not a proposed IEEE interpretation. -/
def countermodel : FloatModel where
  op := fun k _ xs => if k.op = "arith.addf" then 0 else xs.headD 0
  cmp := fun _ _ _ _ => false
  convert := fun _ _ _ x => BitVec.ofNat _ x.toNat
  toInt := fun _ _ w x => some (BitVec.ofNat w x.toNat)
  ofInt := fun _ _ _ x => BitVec.ofNat _ x.toNat

theorem elision_not_parametric : ¬ (∀ M, ZeroPointElisionCorrect M) := by
  intro h
  have he := h countermodel (1 : FBits .f32) (1 : FBits .f32)
  have hn : quantize countermodel 1 1 ≠ quantizeWithoutZero countermodel 1 1 := by decide
  exact hn he

end MerlinVerify.Rules.TorchAOAffineContract
