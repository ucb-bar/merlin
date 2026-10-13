/-
Op family: linalg.generic and linalg.fill (the loop-nest semantics, design §4.3).

Owned by its work package (merlin/lean/AGENT.md). The dispatcher in `Sem/Ops.lean` routes ops to
`sem` by name; at G0 every op is unsupported (`stuck`).
-/
import MerlinVerify.Sem.Program

namespace MerlinVerify.Sem.Ops.Linalg

/-- Iterator kinds of a structured op (`#linalg.iterator_type<...>`). -/
inductive IterKind where
  | parallel | reduction
  deriving DecidableEq, Repr

/-- G0-FROZEN SIGNATURE. Loop-nest semantics of a structured op with explicit indexing maps
(one per input then per output operand) and iterator kinds, applied to input and output (`outs`)
values, with `body` the meaning of its single-block region (block args: one element per input, then
per output; yields one element per output). Named ops (`LinalgNamed`) are defined by calling this
with their OpDSL maps and their own region (design §4.3). Iteration is in lexicographic order of the
iteration space; `body` runs with `ctx.loopIndex` set to the current point. -/
def genericSem (_ctx : Ctx) (_maps : List AffineMap) (_iters : List IterKind)
    (_ins _outs : List RVal) (_body : Ctx → RegionSem) : Outcome (List RVal) :=
  .stuck "linalg generic semantics not implemented"

def sem : OpSem := fun _ _ op _ _ => .stuck s!"unsupported op {op.name}"

end MerlinVerify.Sem.Ops.Linalg
