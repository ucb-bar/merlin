/-
Op family: arith.* and math.* scalar ops, arith.constant (scalar and dense), casts.

Owned by its work package (merlin/lean/AGENT.md). The dispatcher in `Sem/Ops.lean` routes ops to
`sem` by name; at G0 every op is unsupported (`stuck`).
-/
import MerlinVerify.Sem.Program

namespace MerlinVerify.Sem.Ops.Arith

def sem : OpSem := fun _ _ op _ _ => .stuck s!"unsupported op {op.name}"

end MerlinVerify.Sem.Ops.Arith
