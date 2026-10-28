/- Per-tensor signed-i8 quantization contract, proposed for independent review.
PyTorch v2.10.0 torch/ao/quantization/fx/_decomposed.py:42-73,100-128.
Scalar tensor parameters are extracted before the binary64 reciprocal. The activation
arithmetic is f32, round-to-even, integer zero-point conversion, lower then upper clamp,
and signed conversion. Arithmetic keys are explicit; no floating algebra is assumed.
-/
import MerlinVerify.Spec.Dequant

namespace MerlinVerify.Spec.Quantize

def inverse (M : FloatModel) (scale : FBits .f32) : FBits .f32 :=
  M.convert (Dequant.defaultKey "arith.truncf") .f64 .f32
    (M.op (Dequant.defaultKey "arith.divf") .f64
      [0x3ff0000000000000#64,
       M.convert (Dequant.defaultKey "arith.extf") .f32 .f64 scale])

def withInverse (w : Nat) (M : FloatModel) (x inv : FBits .f32)
    (zero : BitVec w) (lower upper : FBits .f32) : Option (BitVec 8) :=
  let scaled := M.op (Dequant.defaultKey "arith.mulf") .f32 [x, inv]
  let rounded := M.op (Dequant.defaultKey "math.roundeven") .f32 [scaled]
  let shifted := M.op (Dequant.defaultKey "arith.addf") .f32
    [rounded, M.ofInt (Dequant.defaultKey "arith.sitofp") w .f32 zero]
  let low := M.op (Dequant.defaultKey "arith.maximumf") .f32 [shifted, lower]
  let high := M.op (Dequant.defaultKey "arith.minimumf") .f32 [low, upper]
  M.toInt (Dequant.defaultKey "arith.fptosi") .f32 8 high

def scalarValue (w : Nat) (M : FloatModel) (x scale : Option (FBits .f32))
    (zero : Option (BitVec w)) (lower upper : FBits .f32) : Val :=
  match x, scale, zero with
  | some x, some scale, some zero =>
    match withInverse w M x (inverse M scale) zero lower upper with
    | some q => .some (.bits 8 q)
    | none => .poison
  | _, _, _ => .poison

def tensor (w : Nat) (M : FloatModel) (shape : List Nat)
    (input : Idx → Option (FBits .f32)) (scale : Option (FBits .f32))
    (zero : Option (BitVec w)) (lower upper : FBits .f32) : Tensor :=
  Tensor.canon shape (.int 8) fun p => scalarValue w M (input p) scale zero lower upper

end MerlinVerify.Spec.Quantize
