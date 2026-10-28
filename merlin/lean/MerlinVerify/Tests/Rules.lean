/-
Build-time tests of the typing rules: each rule must accept the typings the capture uses and refuse
the ones its semantics would get stuck on. A rule that accepts nothing is trivially sound, so the
acceptance tests are what make the soundness theorems meaningful (design §1.2 rule 7).
-/
import MerlinVerify.Check.Rulebook
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.Rules

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics

def i1 : Ty := .int 1
def i64 : Ty := .int 64
def f16 : Ty := .float .f16
def f32 : Ty := .float .f32

def s (t : Ty) : VTy := .scalar t
def T (sh : List Nat) (t : Ty) : VTy := .tensor sh t

/-- An op `name` with one result of type `ty` (operands are irrelevant to the rule). -/
def op1 (name : String) (ty : Ty) (attrs : List (String × Attr) := []) : Op :=
  mkOp name [v 9 ty] [] attrs

def accepts (op : Op) (argTys resTys : List VTy) : Bool := (arithRule 64 0 op argTys resTys).isSome

/-! ### Integer arithmetic -/

#guard accepts (op1 "arith.addi" i32) [s i32, s i32] [s i32]
#guard accepts (op1 "arith.muli" .index) [s .index, s .index] [s .index]
#guard accepts (op1 "arith.subi" (tensorTy [2, 3] i32)) [T [2, 3] i32, T [2, 3] i32] [T [2, 3] i32]
#guard accepts (op1 "arith.divsi" i32) [s i32, s i32] [s i32]
#guard accepts (op1 "arith.remsi" i8) [s i8, s i8] [s i8]
#guard accepts (op1 "arith.shrsi" i32) [s i32, s i32] [s i32]
-- operand types differ / wrong arity / float operands / shapes differ / scalar-tensor mix
#guard !accepts (op1 "arith.addi" i32) [s i32, s i8] [s i32]
#guard !accepts (op1 "arith.addi" i32) [s i32, s i32, s i32] [s i32]
#guard !accepts (op1 "arith.addi" f32) [s f32, s f32] [s f32]
#guard !accepts (op1 "arith.addi" (tensorTy [2, 3] i32)) [T [2, 3] i32, T [3, 2] i32] [T [2, 3] i32]
#guard !accepts (op1 "arith.addi" i32) [s i32, T [] i32] [s i32]
-- the result type in the op must be the one the rule is asked about
#guard !accepts (op1 "arith.addi" i64) [s i32, s i32] [s i32]

/-! ### Comparisons and selection -/

def pred (n : Int) : List (String × Attr) := [("predicate", .int n i64)]

#guard accepts (op1 "arith.cmpi" i1 (pred 2)) [s i32, s i32] [s i1]
#guard accepts (op1 "arith.cmpf" i1 (pred 4)) [s f32, s f32] [s i1]
#guard !accepts (op1 "arith.cmpi" i1) [s i32, s i32] [s i1]
#guard !accepts (op1 "arith.cmpi" i1 (pred 10)) [s i32, s i32] [s i1]
#guard !accepts (op1 "arith.cmpi" i32 (pred 2)) [s i32, s i32] [s i32]
#guard !accepts (op1 "arith.cmpf" i1 (pred 4)) [s i32, s i32] [s i1]

#guard accepts (op1 "arith.select" f32) [s i1, s f32, s f32] [s f32]
#guard accepts (op1 "arith.select" (tensorTy [4] f32)) [s i1, T [4] f32, T [4] f32] [T [4] f32]
#guard accepts (op1 "arith.select" (tensorTy [4] f32)) [T [4] i1, T [4] f32, T [4] f32] [T [4] f32]
#guard !accepts (op1 "arith.select" f32) [s i32, s f32, s f32] [s f32]
#guard !accepts (op1 "arith.select" f32) [s i1, s f32, s f16] [s f32]
#guard !accepts (op1 "arith.select" (tensorTy [4] f32)) [T [2] i1, T [4] f32, T [4] f32] [T [4] f32]

/-! ### Floats -/

#guard accepts (op1 "arith.addf" f32) [s f32, s f32] [s f32]
#guard accepts (op1 "arith.maximumf" (tensorTy [8] f32)) [T [8] f32, T [8] f32] [T [8] f32]
#guard accepts (op1 "arith.negf" f32) [s f32] [s f32]
#guard !accepts (op1 "arith.addf" f32) [s f32, s f16] [s f32]
#guard !accepts (op1 "arith.negf" i32) [s i32] [s i32]

/-! ### Casts -/

#guard accepts (op1 "arith.extsi" i32) [s i8] [s i32]
#guard accepts (op1 "arith.trunci" i8) [s i32] [s i8]
#guard accepts (op1 "arith.index_cast" i64) [s .index] [s i64]
#guard accepts (op1 "arith.sitofp" f32) [s i32] [s f32]
#guard accepts (op1 "arith.fptosi" (tensorTy [2] i32)) [T [2] f32] [T [2] i32]
#guard accepts (op1 "arith.extf" f32) [s f16] [s f32]
#guard !accepts (op1 "arith.sitofp" f32) [s f32] [s f32]
#guard !accepts (op1 "arith.extsi" f32) [s i8] [s f32]
#guard !accepts (op1 "arith.extf" i32) [s f16] [s i32]

/-! ### `math` -/

#guard accepts (op1 "math.rsqrt" f32) [s f32] [s f32]
#guard accepts (op1 "math.roundeven" (tensorTy [3] f32)) [T [3] f32] [T [3] f32]
#guard accepts (op1 "math.absi" i32) [s i32] [s i32]
#guard !accepts (op1 "math.rsqrt" i32) [s i32] [s i32]
#guard !accepts (op1 "math.absi" .index) [s .index] [s .index]
#guard !accepts (op1 "math.fpowi" f32) [s f32, s i32] [s f32]

/-! ### Constants -/

#guard accepts (op1 "arith.constant" i32 [("value", .int 5 i32)]) [] [s i32]
#guard accepts (op1 "arith.constant" f32 [("value", .float 0x3f800000 f32)]) [] [s f32]
#guard !accepts (op1 "arith.constant" i32 [("value", .int 5 i64)]) [] [s i32]
#guard !accepts (op1 "arith.constant" i32 [("value", .int 5 i32)]) [s i32] [s i32]
#guard !accepts (op1 "arith.constant" i32) [] [s i32]
-- dense: a splat holds one element, a full attribute holds every element
#guard accepts (op1 "arith.constant" (tensorTy [2, 2] f32)
  [("value", .dense (tensorTy [2, 2] f32) true [0, 0, 128, 63])]) [] [T [2, 2] f32]
#guard accepts (op1 "arith.constant" (tensorTy [2] i8)
  [("value", .dense (tensorTy [2] i8) false [1, 255])]) [] [T [2] i8]
#guard !accepts (op1 "arith.constant" (tensorTy [2] i8)
  [("value", .dense (tensorTy [2] i8) false [1])]) [] [T [2] i8]
#guard !accepts (op1 "arith.constant" (tensorTy [2] i8)
  [("value", .dense (tensorTy [3] i8) false [1, 2, 3])]) [] [T [2] i8]

/-! ### Outside the family -/

#guard !accepts (op1 "arith.bitcast" f32) [s i32] [s f32]
#guard !accepts (op1 "linalg.fill" (tensorTy [2] f32)) [s f32, T [2] f32] [T [2] f32]

/-! ### Progress checking a block -/

def book : Rulebook Sem.Ops.opSem 64 := (arithRule_sound 64).book

/-- `%2 = addi %0, %1; %3 = cmpi slt %2, %0; %4 = select %3, %2, %0; return %4`. -/
def body : List Op := [
  mkOp "arith.addi" [v 2 i32] [0, 1],
  mkOp "arith.cmpi" [v 3 i1] [2, 0] (pred 2),
  mkOp "arith.select" [v 4 i32] [3, 2, 0],
  ret [4]]

def Γ : TEnv := [(0, s i32), (1, s i32)]

#guard checkOps book 0 Γ body [s i32]
-- wrong yield type, unknown operand, an unmodelled attribute
#guard !checkOps book 0 Γ body [s i64]
#guard !checkOps book 0 [(0, s i32)] body [s i32]
#guard !checkOps book 0 Γ [mkOp "arith.addi" [v 2 i32] [0, 1] [("foo", .unit)], ret [2]] [s i32]

/-! ### Affine bounds -/

#guard affineBounds [4, 6] (.add (.mul (.dim 0) (.const 2)) (.dim 1)) == some (0, 11)
#guard affineBounds [4, 6] (.mul (.const (-3)) (.dim 1)) == some (-15, 0)
#guard affineBounds [4, 6] (.mod (.dim 1) (.const 4)) == some (0, 3)
#guard affineBounds [4, 6] (.floordiv (.dim 1) (.const 4)) == some (0, 1)
#guard affineBounds [4, 6] (.ceildiv (.dim 1) (.const 4)) == some (0, 2)
#guard affineBounds [4, 6] (.sym 0) == none
#guard affineBounds [4, 6] (.mul (.dim 0) (.dim 1)) == none
#guard affineBounds [4, 6] (.mod (.dim 1) (.const 0)) == none
#guard affineBounds [4, 6] (.dim 2) == none

/-! ### Structured ops -/

theorem fragment_sound : RuleSound 64 (firstRule [arithRule 64, linalgRule]) :=
  firstRule_sound (by
    simp only [List.forall_mem_cons, List.not_mem_nil, IsEmpty.forall_iff, implies_true, and_true]
    exact ⟨arithRule_sound 64, linalgRule_sound 64⟩)

def book2 : Rulebook Sem.Ops.opSem 64 := fragment_sound.book

/-- The ops of the first function of a module. -/
def funcBody (m : Module) : List Op :=
  match m.ops with
  | .mk _ _ _ _ _ [.mk [.mk _ ops]] _ :: _ => ops
  | _ => []

def mmΓ : TEnv := [(0, T [2, 3] i8), (1, T [3, 2] i8)]

-- the i8 matmul of `Tests.Semantics`: `arith.constant` (dense splat), then `linalg.generic` with an
-- arith body
#guard checkOps book2 0 mmΓ (funcBody matmulModule) [T [2, 2] i32]
#guard !checkOps book2 0 mmΓ (funcBody matmulModule) [T [2, 2] i8]

/-- The matmul with `B` read through `(i, j, k) ↦ (j, k)`: `k < 3` overruns `B`'s second dim (2). -/
def mmSwapped (maps : List Attr) (iters : List Attr) : List Op :=
  let body : Region := .mk [.mk [v 10 i8, v 11 i8, v 12 i32] [
      mkOp "arith.extsi" [v 13 i32] [10],
      mkOp "arith.extsi" [v 14 i32] [11],
      mkOp "arith.muli" [v 15 i32] [13, 14],
      mkOp "arith.addi" [v 16 i32] [12, 15],
      mkOp "linalg.yield" [] [16]]]
  [mkOp "arith.constant" [v 2 (tensorTy [2, 2] i32)] [] []
     [("value", .dense (tensorTy [2, 2] i32) true [0, 0, 0, 0])],
   mkOp "linalg.generic" [v 3 (tensorTy [2, 2] i32)] [0, 1, 2] []
     [("indexing_maps", .array maps), ("iterator_types", .array iters),
      ("operandSegmentSizes", .denseArray (.int 32) [2, 1])] [body],
   ret [3]]

def mmIters : List Attr := [iter "parallel", iter "parallel", iter "reduction"]

#guard checkOps book2 0 mmΓ (mmSwapped [amap 3 [0, 2], amap 3 [2, 1], amap 3 [0, 1]] mmIters)
  [T [2, 2] i32]
-- out-of-bounds read, missing loop, a map with the wrong number of dims
#guard !checkOps book2 0 mmΓ (mmSwapped [amap 3 [0, 2], amap 3 [1, 2], amap 3 [0, 1]] mmIters)
  [T [2, 2] i32]
#guard !checkOps book2 0 mmΓ
  (mmSwapped [amap 3 [0, 2], amap 3 [2, 1], amap 3 [0, 1]] (mmIters.take 2)) [T [2, 2] i32]
#guard !checkOps book2 0 mmΓ (mmSwapped [amap 2 [0, 1], amap 3 [2, 1], amap 3 [0, 1]] mmIters)
  [T [2, 2] i32]

/-- `linalg.fill` of a scalar into an output argument. -/
def fillOps : List Op :=
  [mkOp "linalg.fill" [v 2 (tensorTy [4] i32)] [0, 1] []
     [("operandSegmentSizes", .denseArray (.int 32) [1, 1])]
     [.mk [.mk [v 10 i32, v 11 i32] [mkOp "linalg.yield" [] [10]]]],
   ret [2]]

#guard checkOps book2 0 [(0, s i32), (1, T [4] i32)] fillOps [T [4] i32]
-- the value's type must be the body's first argument type
#guard !checkOps book2 0 [(0, s i8), (1, T [4] i32)] fillOps [T [4] i32]

/-- A 2-loop `linalg.generic` whose body reads `linalg.index` `k`. -/
def indexOps (k : Int) : List Op :=
  [mkOp "linalg.generic" [v 2 (tensorTy [2, 3] .index)] [1] []
     [("indexing_maps", .array [amap 2 [0, 1]]),
      ("iterator_types", .array [iter "parallel", iter "parallel"]),
      ("operandSegmentSizes", .denseArray (.int 32) [0, 1])]
     [.mk [.mk [v 10 .index] [
        mkOp "linalg.index" [v 11 .index] [] [("dim", .int k i64)],
        mkOp "linalg.yield" [] [11]]]],
   ret [2]]

#guard checkOps book2 0 [(1, T [2, 3] .index)] (indexOps 1) [T [2, 3] .index]
#guard !checkOps book2 0 [(1, T [2, 3] .index)] (indexOps 2) [T [2, 3] .index]
#guard !checkOps book2 0 [(1, T [2, 3] .index)] (indexOps (-1)) [T [2, 3] .index]
-- outside a structured op there is no iteration point to read
#guard !checkOps book2 0 [] [mkOp "linalg.index" [v 11 .index] [] [("dim", .int 0 i64)], ret [11]]
  [s .index]

/-! ### Named structured ops -/

/-- The i8 × i8 → i32 multiply-accumulate body named matmuls carry in generic form. -/
def macBody : Region := .mk [.mk [v 10 i8, v 11 i8, v 12 i32] [
    mkOp "arith.extsi" [v 13 i32] [10],
    mkOp "arith.extsi" [v 14 i32] [11],
    mkOp "arith.muli" [v 15 i32] [13, 14],
    mkOp "arith.addi" [v 16 i32] [12, 15],
    mkOp "linalg.yield" [] [16]]]

def seg (n m : Int) : List (String × Attr) := [("operandSegmentSizes", .denseArray (.int 32) [n, m])]

def mm (name : String) (outTy : Ty) (operands : List ValueId) (n : Int) : List Op :=
  [mkOp name [v 3 outTy] operands (seg n 1) [] [macBody], ret [3]]

#guard checkOps book2 0 [(0, T [2, 3] i8), (1, T [3, 4] i8), (2, T [2, 4] i32)]
  (mm "linalg.matmul" (tensorTy [2, 4] i32) [0, 1, 2] 2) [T [2, 4] i32]
-- the contraction runs over A's 5 columns but B has only 3 rows: B would be read out of bounds
#guard !checkOps book2 0 [(0, T [2, 5] i8), (1, T [3, 4] i8), (2, T [2, 4] i32)]
  (mm "linalg.matmul" (tensorTy [2, 4] i32) [0, 1, 2] 2) [T [2, 4] i32]
#guard checkOps book2 0 [(0, T [5, 2, 3] i8), (1, T [5, 3, 4] i8), (2, T [5, 2, 4] i32)]
  (mm "linalg.batch_matmul" (tensorTy [5, 2, 4] i32) [0, 1, 2] 2) [T [5, 2, 4] i32]
-- the batch loop runs over A's 6 batches but B has 5
#guard !checkOps book2 0 [(0, T [6, 2, 3] i8), (1, T [5, 3, 4] i8), (2, T [6, 2, 4] i32)]
  (mm "linalg.batch_matmul" (tensorTy [6, 2, 4] i32) [0, 1, 2] 2) [T [6, 2, 4] i32]

/-- `quantized_matmul` with scalar zero points: the body receives `a, b, azp, bzp, acc`. -/
def qmmOps : List Op :=
  [mkOp "linalg.quantized_matmul" [v 5 (tensorTy [2, 4] i32)] [0, 1, 2, 3, 4] (seg 4 1) []
     [.mk [.mk [v 10 i8, v 11 i8, v 12 i32, v 13 i32, v 14 i32] [
        mkOp "arith.extsi" [v 15 i32] [10],
        mkOp "arith.subi" [v 16 i32] [15, 12],
        mkOp "arith.extsi" [v 17 i32] [11],
        mkOp "arith.subi" [v 18 i32] [17, 13],
        mkOp "arith.muli" [v 19 i32] [16, 18],
        mkOp "arith.addi" [v 20 i32] [14, 19],
        mkOp "linalg.yield" [] [20]]]],
   ret [5]]

#guard checkOps book2 0
  [(0, T [2, 3] i8), (1, T [3, 4] i8), (2, s i32), (3, s i32), (4, T [2, 4] i32)] qmmOps [T [2, 4] i32]
-- a zero point given as a vector is not read at every point
#guard !checkOps book2 0
  [(0, T [2, 3] i8), (1, T [3, 4] i8), (2, T [3] i32), (3, s i32), (4, T [2, 4] i32)] qmmOps
  [T [2, 4] i32]

def copyBody : Region := .mk [.mk [v 10 f32, v 11 f32] [mkOp "linalg.yield" [] [10]]]

#guard checkOps book2 0 [(0, T [3, 2] f32), (1, T [3, 2] f32)]
  [mkOp "linalg.copy" [v 2 (tensorTy [3, 2] f32)] [0, 1] (seg 1 1) [] [copyBody], ret [2]] [T [3, 2] f32]
#guard !checkOps book2 0 [(0, T [2, 3] f32), (1, T [3, 2] f32)]
  [mkOp "linalg.copy" [v 2 (tensorTy [3, 2] f32)] [0, 1] (seg 1 1) [] [copyBody], ret [2]] [T [3, 2] f32]

def dims (xs : List Int) : List (String × Attr) := [("dimensions", .denseArray i64 xs)]

#guard checkOps book2 0 [(0, T [4] f32), (1, T [3, 4] f32)]
  [mkOp "linalg.broadcast" [v 2 (tensorTy [3, 4] f32)] [0, 1] (dims [0]) [] [copyBody], ret [2]]
  [T [3, 4] f32]
-- with `dimensions = [1]` the 4-vector feeds loop 0, whose range (4, read off the input) overruns
-- the output's 3 rows; a rank mismatch is refused outright
#guard !checkOps book2 0 [(0, T [4] f32), (1, T [3, 4] f32)]
  [mkOp "linalg.broadcast" [v 2 (tensorTy [3, 4] f32)] [0, 1] (dims [1]) [] [copyBody], ret [2]]
  [T [3, 4] f32]
#guard !checkOps book2 0 [(0, T [4] f32), (1, T [3, 4] f32)]
  [mkOp "linalg.broadcast" [v 2 (tensorTy [3, 4] f32)] [0, 1] (dims []) [] [copyBody], ret [2]]
  [T [3, 4] f32]

def perm (xs : List Int) : List (String × Attr) := [("permutation", .denseArray i64 xs)]

#guard checkOps book2 0 [(0, T [2, 3] f32), (1, T [3, 2] f32)]
  [mkOp "linalg.transpose" [v 2 (tensorTy [3, 2] f32)] [0, 1] (perm [1, 0]) [] [copyBody], ret [2]]
  [T [3, 2] f32]
#guard !checkOps book2 0 [(0, T [2, 3] f32), (1, T [3, 2] f32)]
  [mkOp "linalg.transpose" [v 2 (tensorTy [3, 2] f32)] [0, 1] (perm [0, 1]) [] [copyBody], ret [2]]
  [T [3, 2] f32]
#guard !checkOps book2 0 [(0, T [2, 3] f32), (1, T [3, 2] f32)]
  [mkOp "linalg.transpose" [v 2 (tensorTy [3, 2] f32)] [0, 1] (perm [1, 1]) [] [copyBody], ret [2]]
  [T [3, 2] f32]

/-! ### Tensor ops (with the full rulebook) -/

def book3 : Rulebook Sem.Ops.opSem 64 := merlinBook 64

def arr (xs : List Int) : Attr := .denseArray i64 xs
def slice (offs sizes strides : List Int) : List (String × Attr) :=
  [("static_offsets", arr offs), ("static_sizes", arr sizes), ("static_strides", arr strides),
   ("operandSegmentSizes", .denseArray (.int 32) [1, 0, 0, 0])]

-- rows 1..2 (stride 1) and columns 0, 2 (stride 2) of a 4x3 tensor
#guard checkOps book3 0 [(0, T [4, 3] f32)]
  [mkOp "tensor.extract_slice" [v 1 (tensorTy [2, 2] f32)] [0] (slice [1, 0] [2, 2] [1, 2]), ret [1]]
  [T [2, 2] f32]
-- rank-reducing: one row of a 4x3 tensor as a 3-vector
#guard checkOps book3 0 [(0, T [4, 3] f32)]
  [mkOp "tensor.extract_slice" [v 1 (tensorTy [3] f32)] [0] (slice [2, 0] [1, 3] [1, 1]), ret [1]]
  [T [3] f32]
-- a slice past the end, and a result shape that does not match the sizes
#guard !checkOps book3 0 [(0, T [4, 3] f32)]
  [mkOp "tensor.extract_slice" [v 1 (tensorTy [2, 2] f32)] [0] (slice [3, 0] [2, 2] [1, 2]), ret [1]]
  [T [2, 2] f32]
#guard !checkOps book3 0 [(0, T [4, 3] f32)]
  [mkOp "tensor.extract_slice" [v 1 (tensorTy [2, 3] f32)] [0] (slice [1, 0] [2, 2] [1, 2]), ret [1]]
  [T [2, 3] f32]

def islice (offs sizes strides : List Int) : List (String × Attr) :=
  [("static_offsets", arr offs), ("static_sizes", arr sizes), ("static_strides", arr strides),
   ("operandSegmentSizes", .denseArray (.int 32) [1, 1, 0, 0, 0])]

#guard checkOps book3 0 [(0, T [3] f32), (1, T [4, 3] f32)]
  [mkOp "tensor.insert_slice" [v 2 (tensorTy [4, 3] f32)] [0, 1] (islice [2, 0] [1, 3] [1, 1]), ret [2]]
  [T [4, 3] f32]
#guard !checkOps book3 0 [(0, T [3] f32), (1, T [4, 3] f32)]
  [mkOp "tensor.insert_slice" [v 2 (tensorTy [4, 3] f32)] [0, 1] (islice [4, 0] [1, 3] [1, 1]), ret [2]]
  [T [4, 3] f32]
#guard !checkOps book3 0 [(0, T [3] i32), (1, T [4, 3] f32)]
  [mkOp "tensor.insert_slice" [v 2 (tensorTy [4, 3] f32)] [0, 1] (islice [2, 0] [1, 3] [1, 1]), ret [2]]
  [T [4, 3] f32]

#guard checkOps book3 0 [(0, T [2, 3] f32), (1, T [5, 3] f32)]
  [mkOp "tensor.concat" [v 2 (tensorTy [7, 3] f32)] [0, 1] [("dim", .int 0 i64)], ret [2]]
  [T [7, 3] f32]
-- extents along the concatenated dim must add up; the others must agree
#guard !checkOps book3 0 [(0, T [2, 3] f32), (1, T [5, 3] f32)]
  [mkOp "tensor.concat" [v 2 (tensorTy [8, 3] f32)] [0, 1] [("dim", .int 0 i64)], ret [2]]
  [T [8, 3] f32]
#guard !checkOps book3 0 [(0, T [2, 3] f32), (1, T [5, 4] f32)]
  [mkOp "tensor.concat" [v 2 (tensorTy [7, 3] f32)] [0, 1] [("dim", .int 0 i64)], ret [2]]
  [T [7, 3] f32]

#guard checkOps book3 0 [(0, s f32)]
  [mkOp "tensor.splat" [v 1 (tensorTy [2, 2] f32)] [0], ret [1]] [T [2, 2] f32]
#guard checkOps book3 0 [(0, T [2, 2] f32)]
  [mkOp "tensor.cast" [v 1 (tensorTy [2, 2] f32)] [0], ret [1]] [T [2, 2] f32]
#guard !checkOps book3 0 [(0, T [2, 3] f32)]
  [mkOp "tensor.cast" [v 1 (tensorTy [3, 2] f32)] [0], ret [1]] [T [3, 2] f32]

#guard checkOps book3 0 [(0, T [2, 3] f32), (1, s .index), (2, s .index)]
  [mkOp "tensor.extract" [v 3 f32] [0, 1, 2], ret [3]] [s f32]
#guard !checkOps book3 0 [(0, T [2, 3] f32), (1, s .index)]
  [mkOp "tensor.extract" [v 3 f32] [0, 1], ret [3]] [s f32]
#guard checkOps book3 0 [(0, s f32), (1, T [2, 3] f32), (2, s .index), (3, s .index)]
  [mkOp "tensor.insert" [v 4 (tensorTy [2, 3] f32)] [0, 1, 2, 3], ret [4]] [T [2, 3] f32]
#guard checkOps book3 0 [(0, s i32), (1, s i32)]
  [mkOp "tensor.from_elements" [v 2 (tensorTy [2] i32)] [0, 1], ret [2]] [T [2] i32]
#guard !checkOps book3 0 [(0, s i32), (1, s i32)]
  [mkOp "tensor.from_elements" [v 2 (tensorTy [3] i32)] [0, 1], ret [2]] [T [3] i32]

end MerlinVerify.Tests.Rules
