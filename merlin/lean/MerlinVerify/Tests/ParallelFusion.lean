/- Raw admission, scope and execution controls for the bounded fusion checker. -/
import MerlinVerify.Check.ParallelFusion
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.ParallelFusion

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Tests.Semantics
open MerlinVerify.Rules.ParallelFusionScalar

def generic (kind : Kind) (shape : List Nat) (input destination result arg old scalar : Nat)
    (operand : Option Nat := none) (attrs : List (String × Attr) := []) : Op :=
  mkOp "linalg.generic" [v result (tensorTy shape i32)] [input, destination]
    [("indexing_maps", .array [amap shape.length (List.range shape.length),
        amap shape.length (List.range shape.length)]),
     ("iterator_types", .array (List.replicate shape.length (iter "parallel"))),
     ("operandSegmentSizes", .denseArray (.int 32) [1, 1])] []
    [.mk [.mk [v arg i32, v old i32]
      [mkOp kind.name [v scalar i32] [arg, operand.getD arg] attrs, mkOp "linalg.yield" [] [scalar]]]]

def producer (shape : List Nat := [2]) := generic .add shape 0 1 3 5 6 7
def consumer (shape : List Nat := [2]) := generic .mul shape 3 2 4 8 9 10
def Γ (shape : List Nat := [2]) : TEnv :=
  [(0, .tensor shape i32), (1, .tensor shape i32), (2, .tensor shape i32)]
def accepts (first second : Op) (tail : List Op := [ret [4]]) (shape : List Nat := [2]) :=
  (ParallelFusionSyntax.attempt 64 (Γ shape) first second tail).isSome

#guard accepts (producer [2]) (consumer [2])
#guard accepts (producer []) (consumer []) (shape := [])
#guard accepts (producer [0, 3]) (consumer [0, 3]) (shape := [0, 3])
#guard accepts (producer [1, 2, 1]) (consumer [1, 2, 1]) (shape := [1, 2, 1])
#guard !(accepts (producer [2]) (consumer [2]) [ret [3, 4]])
#guard !(accepts (producer [2]) (consumer [2]) [ret [3]])
#guard !(accepts (producer [2]) (generic .mul [2] 0 2 4 8 9 10))
#guard !(accepts (producer [2]) (generic .mul [2] 3 3 4 8 9 10))
#guard !(accepts (generic .add [2] 0 1 0 5 6 7) (consumer [2]))
#guard !(accepts (generic .add [2] 0 1 3 5 5 7) (consumer [2]))
#guard !(accepts (generic .add [2] 0 1 3 5 6 0) (consumer [2]))
#guard !(accepts (producer [2]) (generic .mul [2] 3 2 4 8 9 7))
#guard !(accepts (producer [2]) (consumer [3]))
#guard !(accepts (generic .add [2] 0 1 3 5 6 7 (some 6)) (consumer [2]))
#guard !(accepts (producer [2]) (generic .mul [2] 3 2 4 8 9 10 (some 9)))
#guard !(accepts (generic .add [2] 0 1 3 5 6 7 none
  [("overflowFlags", .enum "arith" "overflow" "nsw")]) (consumer [2]))
#guard !(ParallelFusionSyntax.attempt 64 ((42, .scalar i32) :: Γ [2])
  (generic .add [2] 0 1 3 5 6 7 (some 42)) (consumer [2]) [ret [4]]).isSome
#guard !(accepts (producer [2]) (consumer [2])
  [mkOp "test.unknown" [v 11 i32] [] [] []
    [.mk [.mk [] [mkOp "tensor.extract" [v 12 i32] [3], ret [12]]]], ret [4]])
-- A genuinely fresh later rebinding of the removed id remains safe.
#guard accepts (producer [2]) (consumer [2])
  [mkOp "arith.constant" [v 3 i32] [] [("value", .int 9 i32)], ret [3, 4]]

def kinds : List Kind := [.add, .mul, .band, .bor, .bxor]
#guard kinds.all fun p => kinds.all fun c =>
  accepts (generic p [2] 0 1 3 5 6 7) (generic c [2] 3 2 4 8 9 10)
#guard (ParallelFusionSyntax.attempt 32 (Γ [2]) (producer [2]) (consumer [2]) [ret [4]]).isSome

def env (shape : List Nat) (xs : List Int) : Env :=
  [(0, intTensor shape 32 xs), (1, intTensor shape 32 [17, 19]), (2, intTensor shape 32 [23, 29])]
def source (shape : List Nat) : List Op := [producer shape, consumer shape, ret [4]]
def target (shape : List Nat) : List Op :=
  (ParallelFusionSyntax.attempt 64 (Γ shape) (producer shape) (consumer shape) [ret [4]]).getD []
def run (ops : List Op) (values : Env) := readInts (evalOps opSem ctx (fun _ _ => .ub) values ops)

#guard run (source [2]) (env [2] [2, 3]) == some [[some 16, some 36]]
#guard run (target [2]) (env [2] [2, 3]) == some [[some 16, some 36]]
#guard run (source [2]) (env [2] []) == some [[none, none]]
#guard run (target [2]) (env [2] []) == some [[none, none]]
#guard run (source []) (env [] [3]) == some [[some 36]]
#guard run (target []) (env [] [3]) == some [[some 36]]
#guard run (source [0, 3]) (env [0, 3] []) == some [[]]
#guard run (target [0, 3]) (env [0, 3] []) == some [[]]

end MerlinVerify.Tests.ParallelFusion
