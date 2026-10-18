/- Dense decoding, checked scalar materialization, runtime rank, current bindings and captures. -/
import MerlinVerify.Check.DenseExtract
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.DenseExtract

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics

def dense (id : Nat) (shape : List Nat) (e : Ty) (bytes : List UInt8) : Op :=
  mkOp "arith.constant" [v id (tensorTy shape e)] []
    [("value", .dense (tensorTy shape e) true bytes)]
def scalar (id : Nat) (e : Ty) (a : Attr) : Op := mkOp "arith.constant" [v id e] [] [("value", a)]
def ctor : Op := dense 3 [] i32 [7, 0, 0, 0]
def load : Op := mkOp "tensor.extract" [v 4 i32] [3]
def target : Op := scalar 99 i32 (.int 7 i32)
def badTarget : Op := scalar 99 i32 (.int 8 i32)
def facts := Check.DenseExtract.advance 64 [] ctor
def result (ops : List Op) := Check.DenseExtract.normalize 64 [] [] [target] ops

#guard facts.length == 1
#guard (result [ctor, load, ret [4]]).map (List.map Op.name) == some ["arith.constant", "func.return"]
#guard (result [ctor, load, ret [4, 3]]).map (List.map Op.name) ==
  some ["arith.constant", "arith.constant", "func.return"]
#guard (Check.DenseExtract.rewrite 64 facts [target] load).name == "arith.constant"
#guard (Check.DenseExtract.rewrite 64 facts [badTarget] load).name == "tensor.extract"
#guard (Check.DenseExtract.rewrite 64 facts [] load).name == "tensor.extract"
#guard (Check.DenseExtract.rewrite 64 [] [target] load).name == "tensor.extract"

#guard (Check.DenseExtract.advance 64 facts (mkOp "unknown.write" [v 3 (tensorTy [] i32)] [])).isEmpty
#guard (Check.DenseExtract.advance 64 facts (mkOp "unknown.write" [v 8 i32] [])).length == 1
#guard (Check.DenseExtract.advance 64 [] (dense 3 [2] i32 [7, 0, 0, 0])).isEmpty
#guard (Check.DenseExtract.advance 64 [] (scalar 3 i32 (.int 7 i32))).isEmpty
#guard (Check.DenseExtract.advance 64 [] (dense 3 [] i32 [7])).isEmpty
#guard (Check.DenseExtract.advance 64 []
  (mkOp "arith.constant" [v 3 (tensorTy [] i32)] []
    [("value", .dense (tensorTy [] i32) true [7, 0, 0, 0]), ("unknown.meaning", .unit)])).isEmpty

-- The actual decoded shape, rather than the result annotation, controls rank-zero extraction.
def falseRank : Op := mkOp "arith.constant" [v 3 (tensorTy [] i32)] []
  [("value", .dense (tensorTy [2] i32) true [7, 0, 0, 0])]
#guard (Check.DenseExtract.rewrite 64 (Check.DenseExtract.advance 64 [] falseRank) [target] load).name == "tensor.extract"
#guard (Check.DenseExtract.rewrite 64 facts [target]
  (mkOp "tensor.extract" [v 4 i32] [3, 1])).name == "tensor.extract"
#guard (Check.DenseExtract.rewrite 64 facts
  [scalar 99 (.int 64) (.int 7 (.int 64))] load).name == "tensor.extract"
#guard (Check.DenseExtract.rewrite 64 facts
  [mkOp "arith.constant" [v 99 i32] [] [("value", .int 7 i32), ("unknown.meaning", .unit)]] load).name == "tensor.extract"

-- Captured tensor uses and numeric-name collisions both prevent exact producer deletion.
#guard !(Check.DenseExtract.drop? 64 [(3, .tensor [] i32)] ctor [ret []])
#guard !(Check.DenseExtract.drop? 64 [] ctor
  [mkOp "unknown.capture" [] [] [] [] [.mk [.mk [] [ret [3]]]]])
#guard Check.DenseExtract.drop? 64 [] ctor
  [mkOp "unknown.capture" [] [] [] [] [.mk [.mk [v 8 i32] [ret [8]]]]]

def indexCtor : Op := dense 3 [] .index [255, 255, 255, 255, 255, 255, 255, 255]
def indexLoad : Op := mkOp "tensor.extract" [v 4 .index] [3]
def indexTarget : Op := scalar 99 .index (.int (-1) .index)
#guard (Check.DenseExtract.rewrite 32 (Check.DenseExtract.advance 32 [] indexCtor) [indexTarget] indexLoad).name == "arith.constant"
#guard (Check.DenseExtract.rewrite 64 (Check.DenseExtract.advance 64 [] indexCtor) [indexTarget] indexLoad).name == "arith.constant"

def floatCtor : Op := dense 3 [] (.float .f32) [0x34, 0x12, 0xc0, 0x7f]
def floatLoad : Op := mkOp "tensor.extract" [v 4 (.float .f32)] [3]
def floatTarget : Op := scalar 99 (.float .f32) (.float 0x7fc01234 (.float .f32))
#guard (Check.DenseExtract.rewrite 64 (Check.DenseExtract.advance 64 [] floatCtor) [floatTarget] floatLoad).name == "arith.constant"
#guard (Check.DenseExtract.rewrite 64 (Check.DenseExtract.advance 64 [] floatCtor)
  [scalar 99 (.float .f32) (.float 0x7fc01235 (.float .f32))] floatLoad).name == "tensor.extract"

end MerlinVerify.Tests.DenseExtract
