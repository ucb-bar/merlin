/- Lexical SSA proposals are checked across captures, shadowing and arbitrary outcomes. -/
import MerlinVerify.Check.AlphaNormalize
import MerlinVerify.Check.RegionExact
import MerlinVerify.Rules.IntegerCompareOrder
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.AlphaNormalize

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Tests.Semantics

def Γ : TEnv := [(0, .tensor [3] (.int 13)), (1, .tensor [3] (.int 13)), (2, .scalar i32)]
def shaped : Ty := tensorTy [3] (.int 13)
def const (id : Nat) : Op :=
  mkOp "arith.constant" [v id (.int 13)] [] [("value", .int 5 (.int 13))]
def generic (result input old constant : Nat) (reverse : Bool) : Op :=
  mkOp "linalg.generic" [v result shaped] [0, 1]
    [("indexing_maps", .array [amap 1 [0], amap 1 [0]]),
     ("iterator_types", .array [iter "parallel"]),
     ("operandSegmentSizes", .denseArray (.int 32) [1, 1])] []
    [.mk [.mk [v input (.int 13), v old (.int 13)]
      [mkOp "arith.cmpi" [v 7 (.int 1)]
        (if reverse then [input, constant] else [constant, input])
        [("predicate", .int 0 (.int 64))],
       mkOp "arith.extui" [v 8 (.int 13)] [7],
       mkOp "arith.xori" [v 9 (.int 13)] [8, old],
       mkOp "linalg.yield" [] [9]]]]

-- Hoisting preserves original IDs, whereas the printer assigns new lexical IDs. Raw numeric
-- operand ordering therefore disagrees; canonicalizing internal bindings first resolves it.
def source := [const 6, generic 3 4 5 6 false, ret [3, 2]]
def target := [const 3, generic 4 5 6 3 true, ret [4, 2]]
def order := RegionExact.recursive Rules.IntegerCompareOrder.operations 3
#guard !(alphaOps (SplatNormalize.diagonal Γ) (order source) (order target))
#guard match AlphaNormalize.normalize Γ source, AlphaNormalize.normalize Γ target with
  | some left, some right => alphaOps (SplatNormalize.diagonal Γ) (order left) (order right)
  | _, _ => false
#guard (AlphaNormalize.candidate Γ source).head!.results.map Value.id == [3]
#guard (AlphaNormalize.candidate Γ source)[1]!.results.map Value.id == [4]
#guard (AlphaNormalize.candidate Γ source)[1]!.regions.head!.blocks.head!.args.map Value.id == [5, 6]
example : AlphaNormalize.candidate Γ (AlphaNormalize.candidate Γ source) =
    AlphaNormalize.candidate Γ source := rfl

def tensor (values : List Int) : RVal :=
  .tensor ⟨[3], .int 13,
    fun point => .some (.bits 13 (BitVec.ofInt 13 (values[point.headD 0]?.getD 0)))⟩
def keep : RVal := .scalar (.some (.bits 32 37))
def env : Env := [(0, tensor [5, 7, -1]), (1, tensor [3, 5, 17]), (2, keep)]
def run (ops : List Op) (bindings : Env := env) :=
  readInts (evalOps opSem ctx (fun _ _ => .ub) bindings ops)
#guard run source == some [[some 2, some 5, some 17], [some 37]]
#guard run (AlphaNormalize.candidate Γ source) == run source
#guard run (order (AlphaNormalize.candidate Γ target)) == run source
#guard run source [(0, .tensor (Tensor.full [3] (.int 13) .poison)), (1, tensor [3, 5, 17]), (2, keep)] ==
  some [[none, none, none], [some 37]]
#guard run (AlphaNormalize.candidate Γ source) [] == run source []

-- Entry names remain fixed; internal allocation starts above every entry ID.
def scalarΓ : TEnv := [(10, .scalar i32), (2, .scalar i32)]
def scalarSource := [mkOp "arith.addi" [v 99 i32] [10, 2], ret [99]]
#guard AlphaNormalize.firstFresh scalarΓ == 11
#guard (AlphaNormalize.candidate scalarΓ scalarSource).head!.operands == [10, 2]
#guard (AlphaNormalize.candidate scalarΓ scalarSource).head!.results.map Value.id == [11]
#guard (AlphaNormalize.normalize scalarΓ scalarSource).isSome
#guard (AlphaNormalize.normalize [] [ret [42]]).isNone

def block (args : List Value) (ops : List Op) : Block := .mk args ops
def box (result : Nat) (blocks : List Block) : Op :=
  mkOp "test.unknown" [v result i32] [] [] [] [.mk blocks]
def captureOld := [box 10 [block [] [ret [10]]], ret [10]]
def shadow := [box 11 [block [v 10 i32] [ret [10]]], ret [11]]
#guard (AlphaNormalize.normalize scalarΓ captureOld).isSome
#guard (AlphaNormalize.candidate scalarΓ captureOld).head!.regions.head!.blocks.head!.ops.head!.operands == [10]
#guard (AlphaNormalize.normalize scalarΓ shadow).isSome
#guard (AlphaNormalize.candidate scalarΓ shadow).head!.regions.head!.blocks.head!.ops.head!.operands == [12]
#guard (AlphaNormalize.normalize [] [box 1 [block [v 100 i32] [ret [100]], block [] [ret [100]]], ret [1]]).isNone
#guard (AlphaNormalize.normalize [] [box 1 [], ret [1]]).isSome
#guard (AlphaNormalize.normalize [] [ret [], const 6]).isSome

example (Γ : TEnv) (ss ns : List Op) (h : AlphaNormalize.normalize Γ ss = some ns)
    (ctx : Ctx) (call : CallSem) (env : Env) :
    evalOps opSem ctx call env ns = evalOps opSem ctx call env ss :=
  AlphaNormalize.normalize_sound h ctx call env

end MerlinVerify.Tests.AlphaNormalize
