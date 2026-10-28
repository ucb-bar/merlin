/- Dead producers cannot escape through uses/captures, and region arguments supply actual local
runtime types rather than inheriting the type or provenance of a shadowed capture. -/
import MerlinVerify.Check.ReshapeNested
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.ReshapeNested

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics

def groups (gs : List (List Nat)) : Attr :=
  .array (gs.map fun g => .array (g.map fun i => .int (Int.ofNat i) (.int 64)))

def Γ : TEnv := [(0, .tensor [2, 3, 4] i32)]
def first : Op := mkOp "tensor.collapse_shape" [v 3 (tensorTy [6, 4] i32)] [0]
  [("reassociation", groups [[0, 1], [2]])]
def second : Op := mkOp "tensor.expand_shape" [v 4 (tensorTy [2, 3, 2, 2] i32)] [3]
  [("reassociation", groups [[0, 1], [2, 3]]), ("static_output_shape", .denseArray (.int 64) [2, 3, 2, 2])]
def hint : Op := mkOp "tensor.expand_shape" [v 17 (tensorTy [2, 3, 2, 2] i32)] [10]
  [("reassociation", groups [[0], [1], [2, 3]]), ("static_output_shape", .denseArray (.int 64) [2, 3, 2, 2])]

#guard ReshapeDead.drop? Γ first [ret [0]]
#guard !ReshapeDead.drop? Γ first [ret [3]]
#guard !ReshapeDead.drop? Γ first [mkOp "unknown.capture" [] [] [] [] [.mk [.mk [] [ret [3]]]], ret [0]]
#guard ReshapeDead.drop? Γ first [mkOp "tensor.empty" first.results [], ret [3]]
#guard !ReshapeDead.drop? ((3, .tensor [6, 4] i32) :: Γ) first [ret [3]]
#guard !ReshapeDead.drop? [] first [ret []]
#guard !ReshapeDead.drop? Γ (mkOp "tensor.collapse_shape" first.results [0]
  [("reassociation", groups [[0], [2, 1]])]) [ret [0]]
#guard !ReshapeDead.drop? Γ (mkOp "tensor.collapse_shape" [v 3 (tensorTy [6, 4] (.int 64))] [0]
  [("reassociation", groups [[0, 1], [2]])]) [ret [0]]
#guard !ReshapeDead.drop? [(0, .tensor [1, 1] i32)]
  (mkOp "tensor.collapse_shape" [v 3 (tensorTy [] i32)] [0] [("reassociation", groups [])]) [ret [0]]

def region (arg : Value) : Region := .mk [.mk [arg] [first, second, ret [4]]]
def run (arg : Value) (ty : VTy) : Option Region :=
  SplatNested.region (ReshapeNested.base 64 [hint]) Γ [] (region arg)
    ⟨[ty], [.tensor [2, 3, 2, 2] i32], 0⟩
def count (r : Option Region) : Option Nat := r.bind fun r => match r with
  | .mk [.mk _ ops] => some ops.length
  | _ => none

#guard count (run (v 9 i32) (.scalar i32)) == some 2
-- An actual scalar block argument shadows the captured tensor binding: source typing refuses.
#guard (run (v 0 i32) (.scalar i32)).isNone
-- A same-ID tensor argument supplies its own shape. The source is legal but direct fusion into
-- this target header is invalid; neither producer may be silently removed.
#guard count (run (v 0 (tensorTy [3, 2, 4] i32)) (.tensor [3, 2, 4] i32)) == some 3
#guard count (run (v 0 (tensorTy [2, 3, 4] i32)) (.tensor [2, 3, 4] i32)) == some 2

-- Both unit reassociations are semantically valid. A nested candidate in an earlier operation
-- must not displace the actual top-level candidate when the local source chain is normalized.
def nestedUnit : Op := mkOp "tensor.expand_shape" [v 18 (tensorTy [2, 1, 2, 4] i32)] [10]
  [("reassociation", groups [[0], [1, 2], [3]]), ("static_output_shape", .denseArray (.int 64) [2, 1, 2, 4])]
def localUnit : Op := mkOp "tensor.expand_shape" [v 19 (tensorTy [2, 1, 2, 4] i32)] [10]
  [("reassociation", groups [[0, 1], [2], [3]]), ("static_output_shape", .denseArray (.int 64) [2, 1, 2, 4])]
def unitHints := ReshapeNested.hintOps [mkOp "unknown.container" [] [] [] [] [.mk [.mk [] [nestedUnit]]], localUnit]
#guard unitHints.map Sem.Ops.Tensor.reassociation? == [some [[0, 1], [2], [3]], some [[0], [1, 2], [3]]]
def unitFirst := mkOp "tensor.collapse_shape" [v 3 (tensorTy [4, 4] i32)] [0]
  [("reassociation", groups [[0, 1], [2]])]
def unitSecond := mkOp "tensor.expand_shape" [v 4 (tensorTy [2, 1, 2, 4] i32)] [3]
  [("reassociation", groups [[0, 1, 2], [3]]), ("static_output_shape", .denseArray (.int 64) [2, 1, 2, 4])]
#guard Sem.Ops.Tensor.reassociation? (ReshapeNormalize.rewrite
  (ReshapeFacts.advance [(0, .tensor [2, 2, 4] i32)] [] unitFirst) unitHints unitSecond) ==
    some [[0, 1], [2], [3]]

end MerlinVerify.Tests.ReshapeNested
