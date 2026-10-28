/- Source evaluation is nonvacuous and refuses malformed runtime types and parameter layouts. -/
import MerlinVerify.Check.RulesDequant
import MerlinVerify.Tests.DequantScalar

namespace MerlinVerify.Tests.DequantSource

open MerlinVerify MerlinVerify.Sem.Ops
open MerlinVerify.Rules.DequantTensor

def c : Dequant.Config := ⟨false, 0, [2, 3]⟩
def q : Tensor := quantizedTensor [2, 3] (fun _ => some 127)
def s : Tensor := scaleTensor [] (fun _ => some 1065353216)
def z : Tensor := zeroTensor [] (fun _ => some 16777217)
def model := MerlinVerify.Tests.DequantScalar.witnessFloat

#guard (Dequant.run c model [.tensor q, .tensor s, .tensor z]).isOk
#guard (Dequant.run c model [.tensor (Tensor.full [2, 3] (.int 8) .poison),
  .tensor s, .tensor z]).isOk
#guard !(Dequant.run c model [.tensor { q with shape := [3, 2] }, .tensor s, .tensor z]).isOk
#guard !(Dequant.run c model [.tensor { q with elt := .int 16 }, .tensor s, .tensor z]).isOk
#guard !(Dequant.run c model [.tensor { q with get := fun _ => .some (.bits 32 0) },
  .tensor s, .tensor z]).isOk
#guard !(Dequant.run c model [.tensor q, .tensor { s with shape := [1] }, .tensor z]).isOk
#guard !(Dequant.run c model [.tensor q, .tensor s]).isOk
#guard !(Dequant.run c model [.scalar .poison, .tensor s, .tensor z]).isOk

/-- PyTorch's int zero point, as captured: a rank-0 i64 tensor. -/
def z64 : Tensor := { shape := [], elt := .int 64, get := fun _ => .some (.bits 64 (-3)) }
def channel : Dequant.Config := ⟨true, 1, [2, 3]⟩
def z64c : Tensor := { shape := [3], elt := .int 64, get := fun _ => .some (.bits 64 0) }
def sc : Tensor := scaleTensor [3] (fun _ => some 1065353216)

#guard (Dequant.run c model [.tensor q, .tensor s, .tensor z64]).isOk
#guard !(Dequant.run c model [.tensor q, .tensor s, .tensor { z64 with shape := [1] }]).isOk
-- Per-channel i64 zero points stay outside the contract.
#guard !(Dequant.run channel model [.tensor q, .tensor sc, .tensor z64c]).isOk

end MerlinVerify.Tests.DequantSource
