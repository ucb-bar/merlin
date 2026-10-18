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

def captured (shadow : Bool := false) (dirty : Bool := false) : Module :=
  let tensor := tensorTy [2] .index
  let generic := mkOp "linalg.generic" [v 4 tensor] [3]
    [("indexing_maps", .array [amap 1 [0]]), ("iterator_types", .array [iter "parallel"]),
     ("operandSegmentSizes", .denseArray (.int 32) [0, 1])] []
    [.mk [.mk [v (if shadow then 1 else 5) .index]
      [mkOp "tensor.extract" [v 6 .index] [0, 1], mkOp "linalg.yield" [] [6]]]]
  { ops := [func "f" ([v 0 tensor] ++ if dirty then [] else [v 3 tensor])
    ([mkOp "arith.constant" [v 1 .index] [] [("value", .int 0 .index)]] ++
      (if dirty then [mkOp "tensor.empty" [v 3 tensor] []] else []) ++ [generic, ret [4]])] }

#guard Bounds.check 64 captured "f" [.tensor [2] .index, .tensor [2] .index] [.tensor [2] .index]
#guard Bounds.check 32 (captured false true) "f" [.tensor [2] .index] [.tensor [2] .index]
#guard !Bounds.check 64 (captured true) "f" [.tensor [2] .index, .tensor [2] .index] [.tensor [2] .index]

-- Exhaustive small shapes include zero/unit extents and both sides of each bounds interval.
#guard ([32, 64].all fun ib => (List.range 5).all fun size => (List.range 9).all fun raw =>
  let n : Int := (raw : Int) - 2
  accepts ib [size] [n] == (decide (0 ≤ n ∧ n < (size : Int))))

end MerlinVerify.Tests.DefinedBounds
