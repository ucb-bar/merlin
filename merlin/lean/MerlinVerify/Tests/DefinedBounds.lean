/- Bounds are learned from actual index constants, not from index types or requested outputs. -/
import MerlinVerify.Check.DefinedBounds
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.DefinedBounds

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics
open MerlinVerify.Check.Definedness

def source (shape : List Nat) (indices : List Int) : Module :=
  let constants := indices.zipIdx |>.map fun (n, i) =>
    mkOp "arith.constant" [v (i + 1) .index] [] [("value", .int n .index)]
  { ops := [func "f" [v 0 (tensorTy shape i32)]
    (constants ++ [mkOp "tensor.extract" [v 100 i32] (0 :: (List.range indices.length).map (· + 1)), ret [100]])] }

def accepts (ib : Nat) (shape : List Nat) (indices : List Int) : Bool :=
  Bounds.check ib (source shape indices) "f" [.tensor shape i32] [.scalar i32]

#guard accepts 32 [2, 3] [1, 2]
#guard accepts 64 [1] [0]
#guard accepts 64 [] []
#guard !accepts 64 [0] [0]
#guard !accepts 32 [2] [-1]
#guard !accepts 64 [2] [2]
#guard !accepts 64 [2, 3] [1]
#guard accepts 32 [2] [4294967297]
#guard !accepts 64 [2] [4294967297]
#guard !accepts 32 [2] [2147483648]
#guard accepts 64 [2] [18446744073709551617]

-- A later write replaces the old singleton. Its old initialized type cannot retain old bounds.
def rebound (n : Int) : Module :=
  { ops := [func "f" [v 0 (tensorTy [2] i32)]
    [mkOp "arith.constant" [v 1 .index] [] [("value", .int 0 .index)],
     mkOp "arith.constant" [v 1 .index] [] [("value", .int n .index)],
     mkOp "tensor.extract" [v 2 i32] [0, 1], ret [2]]] }
#guard Bounds.check 64 (rebound 1) "f" [.tensor [2] i32] [.scalar i32]
#guard !Bounds.check 64 (rebound 2) "f" [.tensor [2] i32] [.scalar i32]

end MerlinVerify.Tests.DefinedBounds
