/- Narrow quant_ext source semantics, independent of the lowering and its scalar-body matcher.
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

/-- Captures may carry the full signed-i8 range. Other metadata/contracts are refused. -/
def metadataOK (op : Op) : Bool :=
  op.successors.isEmpty &&
  (match op.attr? "input_dtype" with
    | none => true
    | some (.str "i8") => true
    | _ => false) &&
  match op.attr? "quant_min", op.attr? "quant_max" with
  | none, none => true
  | some (.int lo (.int 64)), some (.int hi (.int 64)) => lo == -128 && hi == 127
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

def run (c : Config) (M : FloatModel) : List RVal → Outcome (List RVal)
  | [.tensor q, .tensor s, .tensor z] =>
    if validInputs c q s z then .ok [.tensor (result c M q s z)]
    else .stuck "dequantization: unsupported runtime types or parameter shapes"
  | _ => .stuck "dequantization: expected three tensor operands"

def sem : OpSem := fun ctx _ op args regions =>
  match config? op with
  | some c => if regions.isEmpty then run c ctx.M args
      else .stuck "dequantization: unexpected regions"
  | none => .stuck "unsupported quant_ext contract (requires signed i8/i32/f32 dequantization)"

end MerlinVerify.Sem.Ops.Dequant
