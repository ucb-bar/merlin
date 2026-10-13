/-
Op family: linalg.generic and linalg.fill (the loop-nest semantics, design §4.3).

Owned by its work package (merlin/lean/AGENT.md). The dispatcher in `Sem/Ops.lean` routes ops to
`sem` by name; at G0 every op is unsupported (`stuck`).
-/
import MerlinVerify.Sem.Program

namespace MerlinVerify.Sem.Ops.Linalg

def sem : OpSem := fun _ _ op _ _ => .stuck s!"unsupported op {op.name}"

end MerlinVerify.Sem.Ops.Linalg
