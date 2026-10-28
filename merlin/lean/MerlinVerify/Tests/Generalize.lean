/- Named-to-generic lowering preserves the actual body, maps, and complete SSA correspondence. -/
import MerlinVerify.Check.Module
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.Generalize

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics

def body : Region := .mk [.mk [v 10 i32, v 11 i32, v 12 i32]
  [mkOp "arith.muli" [v 13 i32] [10, 11], mkOp "arith.addi" [v 14 i32] [12, 13],
   mkOp "linalg.yield" [] [14]]]

def named (name : String := "linalg.matmul") : Op :=
  mkOp name [v 3 (tensorTy [2, 2] i32)] [0, 1, 2]
    [("operandSegmentSizes", .denseArray i32 [2, 1])] [] [body]

def moduleOf (op : Op) : Module := { ops := [func "f"
  [v 0 (tensorTy [2, 3] i32), v 1 (tensorTy [3, 2] i32), v 2 (tensorTy [2, 2] i32)] [op, ret [3]]] }

#guard (Rules.Generalize.expand named).isSome
#guard (Rules.Generalize.expand (named "linalg.batch_matmul")).isSome
#guard moduleCheck (moduleOf named) (moduleOf (Rules.Generalize.generic named))
#guard moduleCheck (moduleOf named) (moduleOf named)
#guard !(Rules.Generalize.expand (named "linalg.copy")).isSome

-- Changing a reduction iterator, affine map, body operation, or segment split is not generalization.
def corrupt (kind : Nat) : Op :=
  let g := Rules.Generalize.generic named
  let attrs := g.attrs.map fun (k, a) =>
    if kind = 0 ∧ k = "iterator_types" then (k, .array [iter "parallel", iter "parallel", iter "parallel"])
    else if kind = 1 ∧ k = "indexing_maps" then (k, .array [amap 3 [0, 1], amap 3 [2, 1], amap 3 [0, 1]])
    else if kind = 2 ∧ k = "operandSegmentSizes" then (k, .denseArray i32 [1, 2])
    else (k, a)
  .mk g.name g.results g.operands attrs g.props
    (if kind = 3 then [.mk [.mk [v 10 i32, v 11 i32, v 12 i32]
      [mkOp "arith.muli" [v 13 i32] [10, 11], mkOp "arith.subi" [v 14 i32] [12, 13],
       mkOp "linalg.yield" [] [14]]]] else g.regions) []

#guard !moduleCheck (moduleOf named) (moduleOf (corrupt 0))
#guard !moduleCheck (moduleOf named) (moduleOf (corrupt 1))
#guard !moduleCheck (moduleOf named) (moduleOf (corrupt 2))
#guard !moduleCheck (moduleOf named) (moduleOf (corrupt 3))

#guard (Rules.Generalize.expand (mkOp "linalg.matmul" [] []
  [("operandSegmentSizes", .denseArray i32 [1, 2])])).isNone
#guard (Rules.Generalize.expand (mkOp "linalg.matmul" [] []
  [("operandSegmentSizes", .denseArray i32 [2, 1]), ("unknown.numeric", .unit)])).isNone

end MerlinVerify.Tests.Generalize
