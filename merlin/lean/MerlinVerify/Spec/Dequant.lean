/-
Proposed scalar contracts for signed-i8 dequantization to f32.
Source: PyTorch v2.10.0, torch/ao/quantization/fx/_decomposed.py,
https://github.com/pytorch/pytorch/blob/v2.10.0/torch/ao/quantization/fx/_decomposed.py

`dequantize_per_tensor` (defined at line 243) computes `(input.to(out_dtype) - zero_point) * scale`
(line 289): the input is converted first, then the scalar zero point is subtracted and the scalar
scale multiplied, both in the f32 result type, so an i32 or i64 zero point is converted to f32
once. The `.tensor` overload (from line 321) extracts both scalars with `.item()` and calls it.
`quant_min`/`quant_max` are not read: neither range check nor clamp.
`dequantize_per_channel` (defined at line 680) checks `quant_min`/`quant_max` against the input
dtype's range (validation only, `_quant_min_max_bounds_check`), then computes
`(input - zero_points.view(...)) * scales` and `.to(out_dtype)` (lines 734–738). For i8 input and an
i32 zero point PyTorch promotes the subtraction to i32; the integer result wraps at i32, and
multiplication by an f32 scale converts it to f32.

These distinct contracts deliberately do not call the lowering builder or Arith.sem. They name
the default MLIR float-operation keys explicitly so changing lowering attributes cannot silently
change the specification. No IEEE algebraic law is assumed. This is an initial, narrow contract;
capture-route binding and human specification review remain required before paper qualification.

Covered: per-tensor with an i32 or i64 zero point; per-channel with an i32 zero point. Excluded:
unsigned inputs, other input or result precisions, per-channel i64 zero points, absent zero points,
per-group semantics, and f64 scales. An f64 scale is excluded because PyTorch then multiplies in
f64 and rounds the product to f32 (per-channel), a double rounding this f32-only model does not
express.
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

/-- Per-tensor with a zero point of any integer width (the contract admits i32 and i64): the
zero point is converted to f32 on its own, exactly as for i32. -/
def perTensorAt (width : Nat) (M : FloatModel) (q : BitVec 8) (z : BitVec width) (s : FBits .f32) :
    FBits .f32 :=
  M.op (defaultKey "arith.mulf") .f32
    [M.op (defaultKey "arith.subf") .f32
      [M.ofInt (defaultKey "arith.sitofp") 8 .f32 q,
       M.ofInt (defaultKey "arith.sitofp") width .f32 z], s]

theorem perTensorAt_32 (M : FloatModel) (q : BitVec 8) (z : BitVec 32) (s : FBits .f32) :
    perTensorAt 32 M q z s = perTensor M q z s := rfl

/-- `perTensorAt` with poison operands. -/
def perTensorValueAt (width : Nat) (M : FloatModel) (q : Option (BitVec 8))
    (z : Option (BitVec width)) (s : Option (FBits .f32)) : Val :=
  match q, z, s with
  | some q, some z, some s => .some (.flt .f32 (perTensorAt width M q z s))
  | _, _, _ => .poison

/-- Per-tensor tensor contract: the scalar parameters are read at the empty index. -/
def perTensorAtTensor (width : Nat) (M : FloatModel) (shape : List Nat)
    (q : Idx → Option (BitVec 8)) (z : Idx → Option (BitVec width))
    (s : Idx → Option (FBits .f32)) : Tensor :=
  Tensor.canon shape (.float .f32) fun p => perTensorValueAt width M (q p) (z []) (s [])

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
