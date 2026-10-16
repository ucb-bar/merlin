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

end MerlinVerify.Tests.DequantSource
