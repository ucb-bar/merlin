/-
Proposed scalar contracts for signed-i8 dequantization to f32 with i32 zero points.
Source: PyTorch v2.10.0, torch/ao/quantization/fx/_decomposed.py,
https://github.com/pytorch/pytorch/blob/v2.10.0/torch/ao/quantization/fx/_decomposed.py

`dequantize_per_tensor` converts the input before subtracting its scalar zero point (lines
259–263); the tensor overload extracts scalar parameters (lines 307–323).
`dequantize_per_channel` subtracts the integer tensor zero point before multiplying by the scale
(lines 670–679). For i8 input and i32 zero point, PyTorch promotes the subtraction to i32.
The integer result wraps at i32; multiplication by an f32 scale converts that result to f32.

These distinct contracts deliberately do not call the lowering builder or Arith.sem. They name
the default MLIR float-operation keys explicitly so changing lowering attributes cannot silently
change the specification. No IEEE algebraic law is assumed. This is an initial, narrow contract;
capture-route binding and human specification review remain required before paper qualification.
Unsigned inputs, other precisions/zero-point widths, absent zero points and per-group semantics
are outside this contract.
-/
import MerlinVerify.Sem.Float
import MerlinVerify.Sem.Support

namespace MerlinVerify.Spec.Dequant

/-- A default-attribute operation in the parametric float model. -/
def defaultKey (op : String) : FloatOpKey :=
  { op, modifiers := "fm1:0;rm4:none;pred4:none" }

/-- Per-tensor: convert both integer operands to f32, subtract, then multiply by the scale. -/
def perTensor (M : FloatModel) (q : BitVec 8) (z : BitVec 32) (s : FBits .f32) : FBits .f32 :=
  M.op (defaultKey "arith.mulf") .f32
    [M.op (defaultKey "arith.subf") .f32
      [M.ofInt (defaultKey "arith.sitofp") 8 .f32 q,
       M.ofInt (defaultKey "arith.sitofp") 32 .f32 z], s]

/-- Per-channel: promote to i32, subtract in i32, convert once to f32, then multiply. -/
def perChannel (M : FloatModel) (q : BitVec 8) (z : BitVec 32) (s : FBits .f32) : FBits .f32 :=
  M.op (defaultKey "arith.mulf") .f32
    [M.ofInt (defaultKey "arith.sitofp") 32 .f32 (q.signExtend 32 - z), s]

/-- Source contract selection is part of the locked specification, not the rewrite checker. -/
def scalar (channel : Bool) (M : FloatModel) (q : BitVec 8) (z : BitVec 32) (s : FBits .f32) :
    FBits .f32 :=
  if channel then perChannel M q z s else perTensor M q z s

/-- Typed operands may be poison independently. Every arithmetic input is consumed eagerly;
an unused destination is deliberately absent from the source contract. -/
def scalarValue (channel : Bool) (M : FloatModel) (q : Option (BitVec 8))
    (z : Option (BitVec 32)) (s : Option (FBits .f32)) : Val :=
  match q, z, s with
  | some q, some z, some s => .some (.flt .f32 (scalar channel M q z s))
  | _, _, _ => .poison

/-- Source parameter addressing: a scalar broadcast or the selected channel coordinate. -/
def parameterIndex (channel : Bool) (axis : Nat) (p : Idx) : Idx :=
  if channel then [(p[axis]?).getD 0] else []

/-- Independent tensor contract: read q at the output coordinate and parameters at the source
coordinate. Canonicalization only fixes the irrelevant out-of-bounds values to poison. -/
def tensor (channel : Bool) (axis : Nat) (M : FloatModel) (shape : List Nat)
    (q : Idx → Option (BitVec 8)) (z : Idx → Option (BitVec 32))
    (s : Idx → Option (FBits .f32)) : Tensor :=
  Tensor.canon shape (.float .f32) fun p =>
    scalarValue channel M (q p) (z (parameterIndex channel axis p))
      (s (parameterIndex channel axis p))

end MerlinVerify.Spec.Dequant
