/-
Build-time tests of the op semantics (`#guard` runs during `lake build`): hand-built programs are
evaluated with `Sem.Ops.denote` and compared with hand-computed results.
-/
import MerlinVerify.Sem.Ops

namespace MerlinVerify.Tests.Semantics

open MerlinVerify

/-! ### A tiny builder for test programs -/

def v (id : Nat) (ty : Ty) : Value := { id, ty }
def mkOp (name : String) (results : List Value) (operands : List ValueId)
    (attrs : List (String × Attr) := []) (props : List (String × Attr) := [])
    (regions : List Region := []) : Op :=
  .mk name results operands attrs props regions []
def ret (ids : List ValueId) : Op := mkOp "func.return" [] ids
def func (name : String) (args : List Value) (ops : List Op) : Op :=
  mkOp "func.func" [] [] [] [("sym_name", .str name)] [.mk [.mk args ops]]

def i8 : Ty := .int 8
def i32 : Ty := .int 32
def tensorTy (shape : List Nat) (elt : Ty) : Ty := .tensor (shape.map some) elt none

def intAttr (n : Int) (ty : Ty) : Attr := .int n ty
def iter (k : String) : Attr := .enum "linalg" "iterator_type" k
def amap (n : Nat) (res : List Nat) : Attr :=
  .affineMap { numDims := n, numSyms := 0, results := res.map AffineExpr.dim }

/-- An arbitrary float model: these tests are integer-only, so its values are irrelevant. -/
def anyFloat : FloatModel where
  op _ k _ := 0#k.width
  cmp _ _ _ _ := false
  convert _ _ k _ := 0#k.width
  toInt _ _ _ _ := none
  ofInt _ _ k _ := 0#k.width

def ctx : Ctx := { M := anyFloat, ext := fun _ _ => .stuck "no externals", indexBits := 64 }

def intTensor (shape : List Nat) (w : Nat) (vals : List Int) : RVal :=
  let arr := vals.toArray
  .tensor { shape, elt := .int w,
            get := fun i => match arr[linearIndex shape i]? with
              | some x => .some (.bits w (BitVec.ofInt w x))
              | none => .poison }

/-- Read every element of every tensor result as a signed integer (`none` for poison). -/
def readInts : Outcome (List RVal) → Option (List (List (Option Int)))
  | .ok rs => rs.mapM fun
    | .tensor t => some ((allPoints t.shape).map fun i => match t.get i with
        | .some (.bits _ x) => some x.toInt
        | _ => none)
    | .scalar (.some (.bits _ x)) => some [some x.toInt]
    | .scalar _ => some [none]
  | _ => none

def isUb : Outcome (List RVal) → Bool
  | .ub => true
  | _ => false

def isStuck : Outcome (List RVal) → Bool
  | .stuck _ => true
  | _ => false

/-! ### i8 x i8 → i32 matmul as `linalg.generic` -/

/-- `C = A(2x3) · B(3x2)` with sign-extended i8 operands and an explicitly zeroed accumulator. -/
def matmulModule : Module :=
  let a := v 0 (tensorTy [2, 3] i8)
  let b := v 1 (tensorTy [3, 2] i8)
  let init := v 2 (tensorTy [2, 2] i32)
  let out := v 3 (tensorTy [2, 2] i32)
  let body : Region := .mk [.mk [v 10 i8, v 11 i8, v 12 i32] [
      mkOp "arith.extsi" [v 13 i32] [10],
      mkOp "arith.extsi" [v 14 i32] [11],
      mkOp "arith.muli" [v 15 i32] [13, 14],
      mkOp "arith.addi" [v 16 i32] [12, 15],
      mkOp "linalg.yield" [] [16]]]
  { ops := [func "mm" [a, b] [
      mkOp "arith.constant" [init] [] [] [("value", .dense (tensorTy [2, 2] i32) true [0, 0, 0, 0])],
      mkOp "linalg.generic" [out] [0, 1, 2] []
        [("indexing_maps", .array [amap 3 [0, 2], amap 3 [2, 1], amap 3 [0, 1]]),
         ("iterator_types", .array [iter "parallel", iter "parallel", iter "reduction"]),
         ("operandSegmentSizes", .denseArray (.int 32) [2, 1])] [body],
      ret [3]]] }

-- A = [[1, -2, 3], [4, 5, -6]], B = [[7, 8], [-9, 10], [11, -12]]
-- C = [[1*7 + -2*-9 + 3*11, 1*8 + -2*10 + 3*-12], [4*7 + 5*-9 + -6*11, 4*8 + 5*10 + -6*-12]]
--   = [[58, -48], [-83, 154]]
#guard readInts (Sem.Ops.denote ctx matmulModule "mm"
    [intTensor [2, 3] 8 [1, -2, 3, 4, 5, -6], intTensor [3, 2] 8 [7, 8, -9, 10, 11, -12]])
  == some [[some 58, some (-48), some (-83), some 154]]

-- i8 extremes: (-128)·(-128)·3 accumulates exactly in i32.
#guard readInts (Sem.Ops.denote ctx matmulModule "mm"
    [intTensor [2, 3] 8 [-128, -128, -128, 127, 127, 127],
     intTensor [3, 2] 8 [-128, 127, -128, 127, -128, 127]])
  == some [[some 49152, some (-48768), some (-48768), some 48387]]

/-! ### Scalar arith: UB, poison, flags -/

def scalarModule (name : String) (attrs : List (String × Attr) := []) : Module :=
  { ops := [func "f" [v 0 i8, v 1 i8] [mkOp name [v 2 i8] [0, 1] attrs, ret [2]]] }

def i8v (x : Int) : RVal := .scalar (.some (.bits 8 (BitVec.ofInt 8 x)))

#guard isUb (Sem.Ops.denote ctx (scalarModule "arith.divsi") "f" [i8v 5, i8v 0])
#guard isUb (Sem.Ops.denote ctx (scalarModule "arith.divsi") "f" [i8v (-128), i8v (-1)])
#guard readInts (Sem.Ops.denote ctx (scalarModule "arith.divsi") "f" [i8v (-7), i8v 2]) == some [[some (-3)]]
#guard readInts (Sem.Ops.denote ctx (scalarModule "arith.remsi") "f" [i8v (-7), i8v 2]) == some [[some (-1)]]
#guard readInts (Sem.Ops.denote ctx (scalarModule "arith.remsi") "f" [i8v (-128), i8v (-1)]) == some [[some 0]]
#guard readInts (Sem.Ops.denote ctx (scalarModule "arith.floordivsi") "f" [i8v (-7), i8v 2]) == some [[some (-4)]]
#guard readInts (Sem.Ops.denote ctx (scalarModule "arith.ceildivsi") "f" [i8v 7, i8v 2]) == some [[some 4]]
#guard readInts (Sem.Ops.denote ctx (scalarModule "arith.addi") "f" [i8v 127, i8v 1]) == some [[some (-128)]]
#guard readInts (Sem.Ops.denote ctx
    (scalarModule "arith.addi" [("overflowFlags", .enum "arith" "overflow" "nsw")]) "f" [i8v 127, i8v 1])
  == some [[none]]
#guard readInts (Sem.Ops.denote ctx
    (scalarModule "arith.addi" [("overflowFlags", .enum "arith" "overflow" "nsw, nuw")]) "f" [i8v 100, i8v 20])
  == some [[some 120]]
#guard readInts (Sem.Ops.denote ctx (scalarModule "arith.shli") "f" [i8v 1, i8v 8]) == some [[none]]
#guard readInts (Sem.Ops.denote ctx (scalarModule "arith.shrsi") "f" [i8v (-128), i8v 7]) == some [[some (-1)]]
#guard readInts (Sem.Ops.denote ctx (scalarModule "arith.shrui") "f" [i8v (-128), i8v 7]) == some [[some 1]]
#guard readInts (Sem.Ops.denote ctx (scalarModule "arith.maxsi") "f" [i8v (-3), i8v 2]) == some [[some 2]]
#guard readInts (Sem.Ops.denote ctx (scalarModule "arith.maxui") "f" [i8v (-3), i8v 2]) == some [[some (-3)]]
-- An attribute the semantics does not read makes the op stuck (fail closed) ...
#guard isStuck (Sem.Ops.denote ctx (scalarModule "arith.addi" [("frobnicate", .unit)]) "f" [i8v 1, i8v 2])
-- ... unless the reviewed annotation spec declares it semantics-free.
#guard readInts (Sem.Ops.denote ctx (scalarModule "arith.addi" [("prov.fqn", .str "layer1")]) "f"
    [i8v 1, i8v 2]) == some [[some 3]]
-- cf.assert: a true condition is a no-op; a false or poison condition aborts (modelled as UB).
def assertModule : Module := { ops := [func "f" [v 0 (.int 1), v 1 i8] [
  mkOp "cf.assert" [] [0] [] [("msg", .str "must hold")], ret [1]]] }
def i1v (b : Nat) : RVal := .scalar (.some (.bits 1 (BitVec.ofNat 1 b)))
#guard readInts (Sem.Ops.denote ctx assertModule "f" [i1v 1, i8v 7]) == some [[some 7]]
#guard isUb (Sem.Ops.denote ctx assertModule "f" [i1v 0, i8v 7])
#guard isUb (Sem.Ops.denote ctx assertModule "f" [.scalar .poison, i8v 7])
-- An op outside the semantics is stuck, never a value.
#guard isStuck (Sem.Ops.denote ctx (scalarModule "arith.frobnicate") "f" [i8v 1, i8v 2])

/-! ### Named ops and tensor ops -/

def matmulBody : Region := .mk [.mk [v 10 i8, v 11 i8, v 12 i32] [
    mkOp "arith.extsi" [v 13 i32] [10], mkOp "arith.extsi" [v 14 i32] [11],
    mkOp "arith.muli" [v 15 i32] [13, 14], mkOp "arith.addi" [v 16 i32] [12, 15],
    mkOp "linalg.yield" [] [16]]]

/-- The same matmul as a named `linalg.matmul` over a `linalg.fill`-zeroed `tensor.empty`. -/
def namedMatmul : Module :=
  let a := v 0 (tensorTy [2, 3] i8)
  let b := v 1 (tensorTy [3, 2] i8)
  let fillBody : Region := .mk [.mk [v 20 i32, v 21 i32] [mkOp "linalg.yield" [] [20]]]
  { ops := [func "mm" [a, b] [
      mkOp "tensor.empty" [v 2 (tensorTy [2, 2] i32)] [],
      mkOp "arith.constant" [v 3 i32] [] [] [("value", intAttr 0 i32)],
      mkOp "linalg.fill" [v 4 (tensorTy [2, 2] i32)] [3, 2] []
        [("operandSegmentSizes", .denseArray i32 [1, 1])] [fillBody],
      mkOp "linalg.matmul" [v 5 (tensorTy [2, 2] i32)] [0, 1, 4] []
        [("operandSegmentSizes", .denseArray i32 [2, 1])] [matmulBody],
      ret [5]]] }

#guard readInts (Sem.Ops.denote ctx namedMatmul "mm"
    [intTensor [2, 3] 8 [1, -2, 3, 4, 5, -6], intTensor [3, 2] 8 [7, 8, -9, 10, 11, -12]])
  == some [[some 58, some (-48), some (-83), some 154]]

/-- Without the fill, the accumulator is `tensor.empty`: every result element is poison. -/
def unfilledMatmul : Module :=
  { ops := [func "mm" [v 0 (tensorTy [2, 3] i8), v 1 (tensorTy [3, 2] i8)] [
      mkOp "tensor.empty" [v 2 (tensorTy [2, 2] i32)] [],
      mkOp "linalg.matmul" [v 5 (tensorTy [2, 2] i32)] [0, 1, 2] []
        [("operandSegmentSizes", .denseArray i32 [2, 1])] [matmulBody],
      ret [5]]] }

#guard readInts (Sem.Ops.denote ctx unfilledMatmul "mm"
    [intTensor [2, 3] 8 [1, 2, 3, 4, 5, 6], intTensor [3, 2] 8 [1, 2, 3, 4, 5, 6]])
  == some [[none, none, none, none]]

def yieldIn : Region := .mk [.mk [v 30 i32, v 31 i32] [mkOp "linalg.yield" [] [30]]]

/-- transpose(2x3 → 3x2), then collapse to 6, expand back to 3x2. -/
def transposeReshape : Module :=
  { ops := [func "f" [v 0 (tensorTy [2, 3] i32)] [
      mkOp "tensor.empty" [v 1 (tensorTy [3, 2] i32)] [],
      mkOp "linalg.transpose" [v 2 (tensorTy [3, 2] i32)] [0, 1] []
        [("permutation", .denseArray (.int 64) [1, 0])] [yieldIn],
      mkOp "tensor.collapse_shape" [v 3 (tensorTy [6] i32)] [2] []
        [("reassociation", .array [.array [.int 0 (.int 64), .int 1 (.int 64)]])],
      mkOp "tensor.expand_shape" [v 4 (tensorTy [3, 2] i32)] [3] []
        [("reassociation", .array [.array [.int 0 (.int 64), .int 1 (.int 64)]]),
         ("static_output_shape", .denseArray (.int 64) [3, 2])],
      ret [3, 4]]] }

#guard readInts (Sem.Ops.denote ctx transposeReshape "f" [intTensor [2, 3] 32 [1, 2, 3, 4, 5, 6]])
  == some [[some 1, some 4, some 2, some 5, some 3, some 6],
           [some 1, some 4, some 2, some 5, some 3, some 6]]

/-- Row sums with `linalg.reduce` over dimension 1. -/
def rowSum : Module :=
  let addBody : Region := .mk [.mk [v 30 i32, v 31 i32] [
    mkOp "arith.addi" [v 32 i32] [30, 31], mkOp "linalg.yield" [] [32]]]
  { ops := [func "f" [v 0 (tensorTy [2, 3] i32)] [
      mkOp "arith.constant" [v 1 (tensorTy [2] i32)] [] [] [("value", .dense (tensorTy [2] i32) true [0, 0, 0, 0])],
      mkOp "linalg.reduce" [v 2 (tensorTy [2] i32)] [0, 1] []
        [("dimensions", .denseArray (.int 64) [1])] [addBody],
      ret [2]]] }

#guard readInts (Sem.Ops.denote ctx rowSum "f" [intTensor [2, 3] 32 [1, 2, 3, 4, 5, 6]])
  == some [[some 6, some 15]]

def sliceAttrs (offs sizes strides : List Int) : List (String × Attr) :=
  [("static_offsets", .denseArray (.int 64) offs), ("static_sizes", .denseArray (.int 64) sizes),
   ("static_strides", .denseArray (.int 64) strides),
   ("operandSegmentSizes", .denseArray i32 [1, 0, 0, 0])]

/-- Rank-reducing slice row 1 of a 2x3 (→ tensor<3>), and a concat of two 1x3 slices. -/
def slices : Module :=
  { ops := [func "f" [v 0 (tensorTy [2, 3] i32)] [
      mkOp "tensor.extract_slice" [v 1 (tensorTy [3] i32)] [0] [] (sliceAttrs [1, 0] [1, 3] [1, 1]),
      mkOp "tensor.extract_slice" [v 2 (tensorTy [1, 3] i32)] [0] [] (sliceAttrs [0, 0] [1, 3] [1, 1]),
      mkOp "tensor.extract_slice" [v 3 (tensorTy [1, 3] i32)] [0] [] (sliceAttrs [1, 0] [1, 3] [1, 1]),
      mkOp "tensor.concat" [v 4 (tensorTy [1, 6] i32)] [3, 2] [] [("dim", .int 1 (.int 64))],
      ret [1, 4]]] }

#guard readInts (Sem.Ops.denote ctx slices "f" [intTensor [2, 3] 32 [1, 2, 3, 4, 5, 6]])
  == some [[some 4, some 5, some 6], [some 4, some 5, some 6, some 1, some 2, some 3]]

/-- A slice that leaves its source is stuck, not clamped. -/
def badSlice : Module :=
  { ops := [func "f" [v 0 (tensorTy [2, 3] i32)] [
      mkOp "tensor.extract_slice" [v 1 (tensorTy [1, 3] i32)] [0] [] (sliceAttrs [2, 0] [1, 3] [1, 1]),
      ret [1]]] }

#guard isStuck (Sem.Ops.denote ctx badSlice "f" [intTensor [2, 3] 32 [1, 2, 3, 4, 5, 6]])

end MerlinVerify.Tests.Semantics
