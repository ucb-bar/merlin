/- Rank-zero tensors carry exactly one scalar. These equalities preserve every payload,
including poison and arbitrary floating-point bits; no scalar algebra is assumed. -/
import MerlinVerify.Check.UniformFacts

namespace MerlinVerify.Rules.ScalarCarrier

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

theorem from_elements_sem {s : Op} {element : Ty}
    (hn : s.name = "tensor.from_elements") (hm : unmodelledAttr? s.blind = none)
    (ht : Sem.Ops.Tensor.resultTensorTy s = some ([], element))
    (ctx : Ctx) (call : CallSem) (value : Val) :
    opSem ctx call s [.scalar value] [] = .ok [.tensor (Tensor.full [] element value)] := by
  have hd : dialectOf s.name = "tensor" := by rw [hn]; cbv
  rw [tensor_sem_of hm hd]
  simp [Sem.Ops.Tensor.sem, hn, resultTensorTy_blind, ht, shapeSize, Tensor.full, linearIndex]

theorem extract_sem {s : Op} {element : Ty}
    (hn : s.name = "tensor.extract") (hm : unmodelledAttr? s.blind = none)
    (ctx : Ctx) (call : CallSem) (value : Val) :
    opSem ctx call s [.tensor (Tensor.full [] element value)] [] = .ok [.scalar value] := by
  have hd : dialectOf s.name = "tensor" := by rw [hn]; cbv
  rw [tensor_sem_of hm hd]
  simp [Sem.Ops.Tensor.sem, hn, Tensor.full, Tensor.canon, Idx.InBounds]

end MerlinVerify.Rules.ScalarCarrier
