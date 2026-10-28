/- Scope guards for exact generic-body integer ordering. Raw malformed SSA is deliberately
covered here because the authoritative MLIR parser rejects it before export. -/
import MerlinVerify.Check.IntegerBody
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.IntegerBody

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics

def shaped : Ty := tensorTy [2] i32

def generic (reverse : Bool) (arg : Nat := 5) (operand : Nat := 3)
    (result : Nat := 7) (argTy : Ty := i32) : Op :=
  mkOp "linalg.generic" [v 4 shaped] [0, 1]
    [("indexing_maps", .array [amap 1 [0], amap 1 [0]]),
     ("iterator_types", .array [iter "parallel"]),
     ("operandSegmentSizes", .denseArray (.int 32) [1, 1])] []
    [.mk [.mk [v arg argTy, v 6 i32]
      [mkOp "arith.addi" [v result i32] (if reverse then [arg, operand] else [operand, arg]),
       mkOp "linalg.yield" [] [result]]]]

def module (reverse : Bool) (arg : Nat := 5) (operand : Nat := 3)
    (result : Nat := 7) (argTy : Ty := i32) : Module :=
  { ops := [func "f" [v 0 shaped, v 1 shaped]
    [mkOp "arith.constant" [v 3 i32] [] [("value", .int 7 i32)],
     generic reverse arg operand result argTy, ret [4]]] }

def accepts (target : Module) : Bool :=
  Check.IntegerBody.check 64 (module false) target "f"
    [.tensor [2] i32, .tensor [2] i32] [.tensor [2] i32]

#guard accepts (module true)
#guard !accepts (module true (arg := 3))
#guard !accepts (module true (result := 3))
#guard !accepts (module true (operand := 999))
#guard !accepts (module true (argTy := i8))

-- Missing bindings and poison are retained, rather than excluded by an initializedness premise.
def arithmetic : Op := mkOp "arith.addi" [v 4 i32] [3, 1]
#guard (Rules.IntegerOrder.operation arithmetic).operands == [1, 3]
#guard !(Rules.IntegerOrder.operation
  (mkOp "arith.addf" [v 4 (.float .f32)] [3, 1])).operands == [1, 3]
#guard !(Rules.IntegerOrder.eligible
  (mkOp "arith.addi" [v 4 i32] [3, 1] [("overflowFlags", .enum "arith" "overflow" "nsw")]))

#guard readInts (Sem.Ops.denote ctx (module false) "f"
  [intTensor [2] 32 [], intTensor [2] 32 [0, 0]]) == some [[none, none]]
#guard readInts (Sem.Ops.denote ctx (module true) "f"
  [intTensor [2] 32 [], intTensor [2] 32 [0, 0]]) == some [[none, none]]

end MerlinVerify.Tests.IntegerBody
