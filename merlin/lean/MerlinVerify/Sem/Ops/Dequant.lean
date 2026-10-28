/- Narrow quant_ext source semantics, independent of the lowering and its scalar-body matcher:
per-tensor with i32 or i64 zero points, per-channel with i32 zero points (`Spec/Dequant.lean`).
The proposed PyTorch contract remains subject to specification and capture-route review. -/
import MerlinVerify.Spec.Dequant

namespace MerlinVerify.Sem.Ops.Dequant

open MerlinVerify

/-- Poison decodes as `none`; malformed values are rejected separately by `tensorTyped`. -/
def bits (w : Nat) : Val → Option (BitVec w)
  | .some (.bits w' x) => if h : w' = w then some (h ▸ x) else none
  | _ => none

def float32 : Val → Option (FBits .f32)
  | .some (.flt .f32 x) => some x
  | _ => none

/-- Check every observable element, including poison, against a fixed scalar type. -/
def tensorTyped (t : Tensor) (ty : Ty) : Bool :=
  t.elt == ty && (allPoints t.shape).all (fun p => (t.get p).conforms 0 ty)

structure Config where
  channel : Bool
  axis : Nat
  shape : List Nat
  deriving DecidableEq

/-- The quantization range as PyTorch reads it. Per-tensor dequantization ignores it entirely;
per-channel checks it against the signed-i8 range and raises otherwise (validation only, so a
narrower range such as [-127, 127] does not change the arithmetic). Other metadata is refused. -/
def metadataOK (op : Op) : Bool :=
  op.successors.isEmpty &&
  (match op.attr? "input_dtype" with
    | none => true
    | some (.str "i8") => true
    | _ => false) &&
  match op.attr? "quant_min", op.attr? "quant_max" with
  | none, none => true
  | some (.int lo (.int 64)), some (.int hi (.int 64)) =>
    op.name == "quant_ext.dequantize_per_tensor" || (decide (-128 ≤ lo) && decide (hi ≤ 127))
  | _, _ => false

def config? (op : Op) : Option Config := do
  if !metadataOK op then none else do
    let [r] := op.results | none
    let some (shape, .float .f32) := r.ty.staticTensor? | none
    if op.name == "quant_ext.dequantize_per_tensor" then
      some ⟨false, 0, shape⟩
    else if op.name == "quant_ext.dequantize_per_channel" then
      let axis ← attrInt? op "axis"
      if axis < 0 ∨ shape.length ≤ axis.toNat then none
      else some ⟨true, axis.toNat, shape⟩
    else none

def paramShape (c : Config) : List Nat :=
  if c.channel then (c.shape[c.axis]?).toList else []

/-- Successful source evaluation requires matching runtime shapes and typed elements. Invalid
inputs are unsupported, rather than being silently interpreted as poison or as a different type. -/
def validInputs (c : Config) (q s z : Tensor) : Bool :=
  decide (q.shape = c.shape ∧ s.shape = paramShape c ∧ z.shape = paramShape c) &&
  tensorTyped q (.int 8) && tensorTyped s (.float .f32) && tensorTyped z (.int 32)

def result (c : Config) (M : FloatModel) (q s z : Tensor) : Tensor :=
  Spec.Dequant.tensor c.channel c.axis M c.shape
    (fun p => bits 8 (q.get p)) (fun p => bits 32 (z.get p)) (fun p => float32 (s.get p))

/-- Per-tensor inputs with a zero point of integer width `width`. -/
def validPerTensorAt (width : Nat) (c : Config) (q s z : Tensor) : Bool :=
  !c.channel && decide (q.shape = c.shape ∧ s.shape = [] ∧ z.shape = []) &&
  tensorTyped q (.int 8) && tensorTyped s (.float .f32) && tensorTyped z (.int width)

def resultPerTensorAt (width : Nat) (c : Config) (M : FloatModel) (q s z : Tensor) : Tensor :=
  Spec.Dequant.perTensorAtTensor width M c.shape
    (fun p => bits 8 (q.get p)) (fun p => bits width (z.get p)) (fun p => float32 (s.get p))

/-- The i32 contract first, then per-tensor with an i64 zero point (PyTorch's int zero point). -/
def run (c : Config) (M : FloatModel) : List RVal → Outcome (List RVal)
  | [.tensor q, .tensor s, .tensor z] =>
    if validInputs c q s z then .ok [.tensor (result c M q s z)]
    else if validPerTensorAt 64 c q s z then .ok [.tensor (resultPerTensorAt 64 c M q s z)]
    else .stuck "dequantization: unsupported runtime types or parameter shapes"
  | _ => .stuck "dequantization: expected three tensor operands"

def sem : OpSem := fun ctx _ op args regions =>
  match config? op with
  | some c => if regions.isEmpty then run c ctx.M args
      else .stuck "dequantization: unexpected regions"
  | none => .stuck "unsupported quant_ext contract (requires signed i8 to f32 dequantization)"

end MerlinVerify.Sem.Ops.Dequant
