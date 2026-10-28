/- Scalar dequantization contracts and adversarial body-shape checks. -/
import MerlinVerify.Rules.DequantScalar

namespace MerlinVerify.Tests.DequantScalar

open MerlinVerify MerlinVerify.Rules.DequantScalar

def args : List Value :=
  [{ id := 0, ty := .int 8 }, { id := 1, ty := .float .f32 },
   { id := 2, ty := .int 32 }, { id := 3, ty := .float .f32 }]

#guard checkBody false args perTensorBody
#guard checkBody true args perChannelBody
#guard !checkBody true args perTensorBody
#guard !checkBody false args perChannelBody
#guard !checkBody false args (perTensorBody.dropLast)
#guard !checkBody false args (perTensorBody ++ [yield 7])
#guard !checkBody false (args.map fun v => { v with id := 0 }) perTensorBody
#guard !checkBody false (args.map fun v => { v with ty := .float .f32 }) perTensorBody

/-- A legal parametric model in which floating subtraction is distinct from integer subtraction.
It witnesses why the two source contracts cannot be exchanged without extra float hypotheses.
This model is deliberately not advertised as IEEE-754. -/
def witnessFloat : FloatModel where
  op key k xs := if key.op == "arith.subf" then 0#k.width else xs.headD 0#k.width
  cmp _ _ _ _ := false
  convert _ _ k _ := 0#k.width
  toInt _ _ _ _ := none
  ofInt _ _ k x := BitVec.ofNat k.width x.toNat

theorem perTensor_ne_perChannel :
    Spec.Dequant.perTensor witnessFloat 1 16777217 1065353216 ≠
      Spec.Dequant.perChannel witnessFloat 1 16777217 1065353216 := by
  cbv
  decide

end MerlinVerify.Tests.DequantScalar
