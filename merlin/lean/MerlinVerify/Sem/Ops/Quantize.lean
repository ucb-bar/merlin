/- Proposed signed-i8 per-tensor quantization source semantics. Arithmetic is
specified independently in Spec/Quantize.lean; specification/capture review remains
required. This module contains no lowering builder or rewrite checker. -/
import MerlinVerify.Spec.Quantize
import MerlinVerify.Sem.Ops.Dequant

namespace MerlinVerify.Sem.Ops.Quantize

open MerlinVerify

structure Config where
  shape : List Nat
  deriving DecidableEq

/-- The initial signed-byte subsection has f32 activations/scales and the full i8 range.
The scalar integer zero point is i32 or i64. Selection depends only on the operation's
actual metadata/types, never on the model name or a proposed replacement. -/
def config? (op : Op) : Option Config := do
  if op.name != "quant_ext.quantize_per_tensor" || !op.successors.isEmpty then none else do
    let [result] := op.results | none
    let some (shape, .int 8) := result.ty.staticTensor? | none
    if op.attr? "quant_min" != some (.int (-128) (.int 64)) ||
        op.attr? "quant_max" != some (.int 127 (.int 64)) then none else do
      match op.attr? "output_dtype" with
      | none | some (.str "int8") | some (.str "i8") => some ⟨shape⟩
      | _ => none

def validInputs (width : Nat) (c : Config) (input scale zero : Tensor) : Bool :=
  decide (input.shape = c.shape ∧ scale.shape = [] ∧ zero.shape = []) &&
    Dequant.tensorTyped input (.float .f32) && Dequant.tensorTyped scale (.float .f32) &&
    Dequant.tensorTyped zero (.int width)

def result (width : Nat) (c : Config) (M : FloatModel) (input scale zero : Tensor) : Tensor :=
  Spec.Quantize.tensor width M c.shape (fun p => Dequant.float32 (input.get p))
    (Dequant.float32 (scale.get [])) (Dequant.bits width (zero.get [])) 0xc3000000#32 0x42fe0000#32

def run (c : Config) (M : FloatModel) : List RVal → Outcome (List RVal)
  | [.tensor input, .tensor scale, .tensor zero] =>
    if validInputs 32 c input scale zero then .ok [.tensor (result 32 c M input scale zero)]
    else if validInputs 64 c input scale zero then .ok [.tensor (result 64 c M input scale zero)]
    else .stuck "quantization: requires f32 data/scale and a scalar i32/i64 zero point"
  | _ => .stuck "quantization: expected three tensor operands"

def sem : OpSem := fun ctx _ op args regions =>
  match config? op with
  | some c => if regions.isEmpty then run c ctx.M args
      else .stuck "quantization: unexpected regions"
  | none => .stuck "unsupported signed-i8 per-tensor quantization contract"

end MerlinVerify.Sem.Ops.Quantize
