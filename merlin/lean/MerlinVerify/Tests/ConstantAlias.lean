/- Exact payload equality, checked captures, binding invalidation, and literal-width effects. -/
import MerlinVerify.Check.ConstantAlias
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.ConstantAlias

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics

def constant (id : Nat) (ty : Ty) (value : Attr) : Op :=
  mkOp "arith.constant" [v id ty] [] [("value", value)]
def integer (id : Nat) (n : Int) (ty : Ty := i32) : Op := constant id ty (.int n ty)
def floating (id bits : Nat) : Op := constant id (.float .f32) (.float bits (.float .f32))
def normalized (ib : Nat) (ops : List Op) := Check.ConstantAlias.normalize ib [] [] ops

def a : Op := integer 1 7
def b : Op := integer 2 7
def different : Op := integer 2 8
def facts := Check.ConstantAlias.advance 64 [] a

def capture (args : List Value) (uses : List ValueId) : Op :=
  mkOp "unknown.capture" [] [] [] [] [.mk [.mk args [ret uses]]]

#guard (normalized 64 [a, b, ret [2]]).map (List.map Op.operands) == some [[], [1]]
#guard (normalized 64 [a, b, ret [2, 1]]).map (List.map Op.operands) == some [[], [1, 1]]
#guard (normalized 64 [a, b, integer 3 7, ret [1, 2, 3]]).map (List.map Op.operands) ==
  some [[], [1, 1, 1]]
#guard (normalized 64 [a, different, ret [2, 1]]).map (List.map Op.operands) == some [[], [], [2, 1]]
#guard (normalized 64 [integer 1 (-1) (.int 8), integer 2 255 (.int 8), ret [2, 1]]).map
  (List.map Op.operands) == some [[], [1, 1]]
#guard (normalized 32 [integer 1 (-9223372036854775808) .index, integer 2 0 .index, ret [2, 1]]).map
  (List.map Op.operands) == some [[], [1, 1]]
#guard (normalized 64 [integer 1 (-9223372036854775808) .index, integer 2 0 .index, ret [2, 1]]).map
  (List.map Op.operands) == some [[], [], [2, 1]]
#guard (normalized 64 [integer 1 7 (.int 64), integer 2 7 .index, ret [2, 1]]).map
  (List.map Op.operands) == some [[], [], [2, 1]]

#guard (normalized 64 [floating 1 0x7fc01234, floating 2 0x7fc01234, ret [2, 1]]).map
  (List.map Op.operands) == some [[], [1, 1]]
#guard (normalized 64 [floating 1 0x7fc01234, floating 2 0x7fc01235, ret [2, 1]]).map
  (List.map Op.operands) == some [[], [], [2, 1]]
#guard (normalized 64 [floating 1 0, floating 2 0x80000000, ret [2, 1]]).map
  (List.map Op.operands) == some [[], [], [2, 1]]
#guard (Check.ConstantAlias.advance 64 facts (mkOp "unknown.write" [v 1 i32] [])).isEmpty
#guard (Check.ConstantAlias.advance 64 facts (mkOp "unknown.write" [v 3 i32] [])).length == 1
-- The old value is genuinely dead before a later write; stale facts still cannot alias b.
#guard (normalized 64 [a, mkOp "unknown.write" [v 1 i32] [], b, ret [2, 1]]).map
  (List.map Op.name) == some ["unknown.write", "arith.constant", "func.return"]
#guard (normalized 64 [a, integer 1 8, b, ret [2, 1]]).map
  (List.map Op.operands) == some [[], [], [2, 1]]
#guard (Check.ConstantAlias.advance 64 [] (mkOp "arith.constant" [v 1 i32] []
  [("value", .int 7 i32), ("unknown.meaning", .unit)])).isEmpty
#guard (Check.ConstantAlias.advance 64 [] (mkOp "arith.constant" [v 1 i32] [] [])).isEmpty
#guard (Check.ConstantAlias.advance 64 [] (mkOp "arith.constant" [v 1 i32] []
  [("value", .str "poison")])).isEmpty
#guard (Check.ConstantAlias.advance 64 [] (mkOp "arith.constant" [v 1 (tensorTy [] i32)] []
  [("value", .dense (tensorTy [] i32) true [7, 0, 0, 0])])).isEmpty

-- The validated structural substitution updates captures but refuses capture-name collisions.
#guard (Check.ConstantAlias.eliminate 64 [(1, .scalar i32)] facts b [capture [] [2]]).isSome
#guard (Check.ConstantAlias.eliminate 64 [(1, .scalar i32)] facts b
  [capture [v 1 i32] [2]]).isNone
#guard (Check.ConstantAlias.eliminate 64 [(1, .scalar i32)] facts b
  [capture [v 2 i32] [2]]).isNone
#guard (Check.ConstantAlias.eliminate 64 [(1, .scalar i32)] facts b
  [mkOp "unknown.write" [v 1 i32] [], ret [2]]).isNone
#guard (Check.ConstantAlias.eliminate 64 [(1, .scalar i32)] facts b
  [mkOp "unknown.write" [v 3 i32] [], ret [2]]).isSome
#guard (Check.ConstantAlias.eliminate 64 [] facts b [ret [2]]).isNone

end MerlinVerify.Tests.ConstantAlias
