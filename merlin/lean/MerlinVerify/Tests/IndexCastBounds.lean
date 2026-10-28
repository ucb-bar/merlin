/- Unsigned widening bounds follow from actual initialized fixed-width values, including high bits. -/
import MerlinVerify.Check.DefinedBounds
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.IndexCastBounds

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics
open MerlinVerify.Check.Definedness

def source (width size : Nat) (name := "arith.index_castui") (flags : List (String × Attr) := [])
    (extract := true) : Module :=
  { ops := [func "f" [v 0 (tensorTy [size] i32), v 1 (.int width)]
    ([mkOp name [v 2 .index] [1] flags] ++
     (if extract then [mkOp "tensor.extract" [v 3 i32] [0, 2], ret [3]] else [ret [2]]))] }

def accepts (ib width size : Nat) (name := "arith.index_castui") : Bool :=
  Bounds.check ib (source width size name) "f" [.tensor [size] i32, .scalar (.int width)] [.scalar i32]

#guard accepts 32 1 2
#guard accepts 64 8 256
#guard !accepts 64 1 1
#guard !accepts 32 8 255
#guard !accepts 32 1 0
#guard !accepts 32 32 2
#guard !accepts 32 64 2
#guard !accepts 64 8 256 "arith.index_cast"
#guard Bounds.check 32 (source 8 2 "arith.index_cast" [] false) "f"
  [.tensor [2] i32, .scalar (.int 8)] [.scalar .index]
#guard !Bounds.check 64 (source 8 256 "arith.index_castui" [("nonNeg", .unit)]) "f"
  [.tensor [256] i32, .scalar (.int 8)] [.scalar i32]

-- Singleton range evidence can survive a signed roundtrip from a known actual index value.
def roundtrip : Module :=
  { ops := [func "f" [v 0 (tensorTy [2] i32)]
    [mkOp "arith.constant" [v 1 .index] [] [("value", .int 1 .index)],
     mkOp "arith.index_cast" [v 2 (.int 8)] [1],
     mkOp "arith.index_cast" [v 3 .index] [2],
     mkOp "tensor.extract" [v 4 i32] [0, 3], ret [4]]] }
#guard Bounds.check 64 roundtrip "f" [.tensor [2] i32] [.scalar i32]

-- A generic argument shadows the captured cast result with an unconstrained index value.
def shadowed : Module :=
  let generic := mkOp "linalg.generic" [v 4 (tensorTy [2] .index)] [3]
    [("indexing_maps", .array [amap 1 [0]]), ("iterator_types", .array [iter "parallel"]),
     ("operandSegmentSizes", .denseArray (.int 32) [0, 1])] []
    [.mk [.mk [v 2 .index]
      [mkOp "tensor.extract" [v 5 i32] [0, 2],
       mkOp "arith.constant" [v 6 .index] [] [("value", .int 0 .index)],
       mkOp "linalg.yield" [] [6]]]]
  { ops := [func "f" [v 0 (tensorTy [256] i32), v 1 (.int 8), v 3 (tensorTy [2] .index)]
    [mkOp "arith.index_castui" [v 2 .index] [1], generic, ret [4]]] }
#guard !Bounds.check 64 shadowed "f"
  [.tensor [256] i32, .scalar (.int 8), .tensor [2] .index] [.tensor [2] .index]

-- Actual source width must fit strictly below target width; no unsigned/sign coincidence inferred.
#guard ([2, 3, 32, 64].all fun ib => (List.range 5).all fun raw =>
  let w := raw + 1
  let size := 2 ^ w
  accepts ib w size == decide (w < ib))

end MerlinVerify.Tests.IndexCastBounds
