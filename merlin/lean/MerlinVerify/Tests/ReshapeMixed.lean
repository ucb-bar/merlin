/- Direct rule probes include malformed SSA that authoritative MLIR parsing normally rejects. -/
import MerlinVerify.Rules.ReshapeMixed
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.ReshapeMixed

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics

def groups (gs : List (List Nat)) : Attr :=
  .array (gs.map fun g => .array (g.map fun i => .int (Int.ofNat i) (.int 64)))

def expanded : Op := mkOp "tensor.expand_shape" [v 3 (tensorTy [2, 3, 4] i32)] [0]
  [("reassociation", groups [[0, 1], [2]]), ("static_output_shape", .denseArray (.int 64) [2, 3, 4])]

def collapsed (result : Nat := 4) : Op := mkOp "tensor.collapse_shape" [v result (tensorTy [24] i32)] [3]
  [("reassociation", groups [[0, 1, 2]])]

def target (input : Nat := 10) (result : Nat := 7) : Op :=
  mkOp "tensor.collapse_shape" [v result (tensorTy [24] i32)] [input]
    [("reassociation", groups [[0, 1]])]

def accepts (ρ : Rho := [(0, 10)]) (a : Op := expanded) (b : Op := collapsed)
    (t : Op := target) : Bool := (Rules.ReshapeMixed.check ρ [a, b] [t]).isSome

#guard accepts
#guard !accepts (t := target (input := 11))
#guard !accepts (t := target (result := 10))
#guard !accepts (ρ := [(0, 10), (3, 12)])
#guard !accepts (b := collapsed (result := 3))
#guard !accepts (a := mkOp "tensor.expand_shape" [v 0 (tensorTy [2, 3, 4] i32)] [0]
  [("reassociation", groups [[0, 1], [2]])])

-- The runtime source shape is derived from reassociation products, not a function argument type.
#guard Rules.ReshapeMixed.inputShape [[0, 1], [2]] [2, 3, 4] == [6, 4]
#guard Rules.ReshapeMixed.inputShape [[0, 1], [2]] [0, 3, 4] == [0, 4]

end MerlinVerify.Tests.ReshapeMixed
