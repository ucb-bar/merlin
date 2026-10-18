/- Raw AST checks exercise provenance, hints, and SSA shadowing before MLIR validation. -/
import MerlinVerify.Check.ReshapeNormalize
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.ReshapeTyped

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics
def groups (gs : List (List Nat)) : Attr :=
  .array (gs.map fun g => .array (g.map fun i => .int (Int.ofNat i) (.int 64)))

def Γ : TEnv := [(0, .tensor [2, 3, 4] (.int 32))]

def first : Op := mkOp "tensor.collapse_shape" [v 3 (tensorTy [6, 4] i32)] [0]
  [("reassociation", groups [[0, 1], [2]])]

def second : Op := mkOp "tensor.expand_shape" [v 4 (tensorTy [2, 3, 2, 2] i32)] [3]
  [("reassociation", groups [[0, 1], [2, 3]]),
    ("static_output_shape", .denseArray (.int 64) [2, 3, 2, 2])]

def hint : Op := mkOp "tensor.expand_shape" [v 17 (tensorTy [2, 3, 2, 2] i32)] [10]
  [("reassociation", groups [[0], [1], [2, 3]]),
    ("static_output_shape", .denseArray (.int 64) [2, 3, 2, 2])]

def facts : List ReshapeFacts.Fact := ReshapeFacts.advance Γ [] first

#guard facts.length == 1
#guard (ReshapeNormalize.rewrite facts [hint] second).operands == [0]
#guard (ReshapeNormalize.rewrite facts [hint] second).results.map Value.id == [4]
#guard (ReshapeNormalize.rewrite facts [] second).operands == [3]
-- A same-shaped annotation on the first result cannot supply the missing runtime input shape.
#guard (ReshapeNormalize.rewrite (ReshapeFacts.advance [] [] first) [hint] second).operands == [3]
#guard (ReshapeNormalize.rewrite (ReshapeFacts.advance [(0, .tensor [3, 2, 4] (.int 32))] [] first)
  [hint] second).operands == [3]
#guard (ReshapeNormalize.rewrite facts [first] second).operands == [3]
#guard (ReshapeNormalize.rewrite facts [mkOp "tensor.expand_shape" hint.results hint.operands
  [("reassociation", groups [[0, 1], [2], [3]])]] second).operands == [3]
#guard (ReshapeNormalize.rewrite facts [hint] first).operands == [0]
-- Both root and result dependencies must remain live; arbitrary writes mask facts.
#guard (ReshapeFacts.advance Γ facts (mkOp "unknown.write" [v 0 i32] [])).isEmpty
#guard (ReshapeFacts.advance Γ facts (mkOp "unknown.write" [v 3 i32] [])).isEmpty
#guard (ReshapeFacts.advance Γ facts (mkOp "unknown.write" [v 99 i32] [])).length == 1
#guard (ReshapeFacts.learn Γ (mkOp "tensor.collapse_shape" [v 0 (tensorTy [6, 4] i32)] [0]
  [("reassociation", groups [[0, 1], [2]])])).isNone

end MerlinVerify.Tests.ReshapeTyped
