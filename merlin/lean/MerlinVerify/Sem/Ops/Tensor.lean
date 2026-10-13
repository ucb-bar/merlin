/-
Op family: tensor.empty, extract_slice, insert_slice, collapse_shape, expand_shape, concat, pad, cast, ....

Owned by its work package (merlin/lean/AGENT.md). The dispatcher in `Sem/Ops.lean` routes ops to
`sem` by name; at G0 every op is unsupported (`stuck`).
-/
import MerlinVerify.Sem.Program

namespace MerlinVerify.Sem.Ops.Tensor

def sem : OpSem := fun _ _ op _ _ => .stuck s!"unsupported op {op.name}"

end MerlinVerify.Sem.Ops.Tensor
