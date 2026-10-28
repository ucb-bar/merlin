/- Total scalar constant deletion checks exact runtime decoding, captures, and freshness. -/
import MerlinVerify.Tests.ConstantAlias

namespace MerlinVerify.Tests.ConstantDead

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics
open MerlinVerify.Tests.ConstantAlias

#guard (normalized 64 [a, ret []]).map (List.map Op.name) == some ["func.return"]
#guard (normalized 64 [a, b, ret []]).map (List.map Op.name) == some ["func.return"]
#guard (normalized 64 [a, different, ret [2]]).map (List.map Op.operands) == some [[], [2]]
#guard (normalized 64 [a, b, ret [1, 2]]).map (List.map Op.operands) == some [[], [1, 1]]
#guard Check.ConstantAlias.drop? 64 [] a [ret []]
#guard !(Check.ConstantAlias.drop? 64 [] a [ret [1]])
#guard !(Check.ConstantAlias.drop? 64 [(1, .scalar i32)] a [ret []])
#guard !(Check.ConstantAlias.drop? 64 [] a [capture [] [1]])
#guard !(Check.ConstantAlias.drop? 64 [] a [capture [v 99 i32] [1]])
#guard Check.ConstantAlias.drop? 64 [] a [capture [v 1 i32] [1]]
#guard Check.ConstantAlias.drop? 64 [] a [mkOp "unknown.write" [v 1 i32] [], ret [1]]
#guard !(Check.ConstantAlias.drop? 64 []
  (mkOp "arith.constant" [v 1 i32] [] [("value", .int 7 i32), ("unknown.meaning", .unit)]) [ret []])
#guard !(Check.ConstantAlias.drop? 64 [] (mkOp "arith.constant" [v 1 i32] [] []) [ret []])
#guard !(Check.ConstantAlias.drop? 64 []
  (mkOp "arith.constant" [v 1 i32] [] [("value", .str "poison")]) [ret []])
#guard !(Check.ConstantAlias.drop? 64 []
  (mkOp "arith.constant" [v 1 (tensorTy [] i32)] []
    [("value", .dense (tensorTy [] i32) true [7, 0, 0, 0])]) [ret []])
#guard Check.ConstantAlias.drop? 64 [] (floating 1 0x7fc01234) [ret []]
#guard Check.ConstantAlias.drop? 64 [] (floating 1 0x80000000) [ret []]
#guard Check.ConstantAlias.drop? 32 [] (integer 1 4294967296 .index) [ret []]
#guard Check.ConstantAlias.drop? 64 [] (integer 1 4294967296 .index) [ret []]

end MerlinVerify.Tests.ConstantDead
