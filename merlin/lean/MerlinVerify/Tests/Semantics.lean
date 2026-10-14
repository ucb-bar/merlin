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
-- An op outside the semantics is stuck, never a value.
#guard isStuck (Sem.Ops.denote ctx (scalarModule "arith.frobnicate") "f" [i8v 1, i8v 2])

end MerlinVerify.Tests.Semantics
