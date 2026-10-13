/-
Op family: named linalg ops (matmul, batch_matmul, convolutions, transpose, ...) defined by their OpDSL generic form.

Owned by its work package (merlin/lean/AGENT.md). The dispatcher in `Sem/Ops.lean` routes ops to
`sem` by name; at G0 every op is unsupported (`stuck`).
-/
import MerlinVerify.Sem.Program

namespace MerlinVerify.Sem.Ops.LinalgNamed

def sem : OpSem := fun _ _ op _ _ => .stuck s!"unsupported op {op.name}"

end MerlinVerify.Sem.Ops.LinalgNamed
