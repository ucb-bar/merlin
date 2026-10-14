/-
Build-time tests of the typing rules: each rule must accept the typings the capture uses and refuse
the ones its semantics would get stuck on. A rule that accepts nothing is trivially sound, so the
acceptance tests are what make the soundness theorems meaningful (design §1.2 rule 7).
-/
import MerlinVerify.Check.RulesArith
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

def accepts (op : Op) (argTys resTys : List VTy) : Bool := (arithRule 64 op argTys resTys).isSome

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

#guard checkOps book Γ body [s i32]
-- wrong yield type, unknown operand, an unmodelled attribute
#guard !checkOps book Γ body [s i64]
#guard !checkOps book [(0, s i32)] body [s i32]
#guard !checkOps book Γ [mkOp "arith.addi" [v 2 i32] [0, 1] [("foo", .unit)], ret [2]] [s i32]

end MerlinVerify.Tests.Rules
