/- Actual modular index arithmetic may establish extraction bounds, never type-only bounds. -/
import MerlinVerify.Check.DefinedBounds
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.IndexArithmetic

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics
open MerlinVerify.Check.Definedness

def source (name : String) (lhs rhs : Int) (size : Nat) (flags : List (String × Attr) := [])
    (rebind : Nat := 0) : Module :=
  { ops := [func "f" [v 0 (tensorTy [size] i32), v 9 .index]
    ([mkOp "arith.constant" [v 1 .index] [] [("value", .int lhs .index)],
      mkOp "arith.constant" [v 2 .index] [] [("value", .int rhs .index)],
      mkOp name [v 3 .index] [1, 2] flags] ++
     (if rebind = 1 then [mkOp "arith.constant" [v 1 .index] [] [("value", .int 7 .index)]]
      else if rebind = 2 then [mkOp "arith.addi" [v 3 .index] [9, 9]] else []) ++
     [mkOp "tensor.extract" [v 4 i32] [0, 3], ret [4]])] }

def accepts (ib : Nat) (name : String) (a b : Int) (size : Nat) (rebind := 0) : Bool :=
  Bounds.check ib (source name a b size [] rebind) "f" [.tensor [size] i32, .scalar .index] [.scalar i32]

#guard accepts 64 "arith.addi" 1 1 3
#guard accepts 32 "arith.subi" 1 0 2
#guard accepts 64 "arith.muli" 1 2 3
#guard accepts 64 "arith.muli" 0 0 1
#guard !accepts 64 "arith.muli" 0 0 0
#guard !accepts 32 "arith.subi" 0 1 2
#guard !accepts 32 "arith.addi" 2147483647 1 2
#guard accepts 32 "arith.muli" 2147483647 2147483647 2
#guard !accepts 64 "arith.muli" 2147483647 2147483647 2
#guard accepts 32 "arith.addi" 4294967296 1 2
#guard !accepts 64 "arith.addi" 4294967296 1 2
-- Evidence is about the computed value, so changing an old operand does not invalidate it.
#guard accepts 64 "arith.addi" 0 1 2 1
-- Replacing the result by an unknown initialized index must discard its old singleton.
#guard !accepts 64 "arith.addi" 0 1 2 2

#guard !Bounds.check 64
  (source "arith.addi" 0 1 2 [("overflowFlags", .enum "arith" "overflow" "nsw")])
  "f" [.tensor [2] i32, .scalar .index] [.scalar i32]
#guard !Bounds.check 64 (source "arith.divsi" 0 1 2)
  "f" [.tensor [2] i32, .scalar .index] [.scalar i32]

-- Exhaustive small cases exercise signed wrap at widths2/3 as well as32/64.
-- Only nonnegative signed singleton operands contribute arithmetic interval evidence.
#guard ([2, 3, 32, 64].all fun ib =>
  ["arith.addi", "arith.subi", "arith.muli"].all fun name =>
  (List.range 4).all fun a => (List.range 4).all fun b => (List.range 5).all fun size =>
    let lhs := (BitVec.ofNat ib a).toInt
    let rhs := (BitVec.ofNat ib b).toInt
    let result := (match name with
      | "arith.addi" => BitVec.ofNat ib a + BitVec.ofNat ib b
      | "arith.subi" => BitVec.ofNat ib a - BitVec.ofNat ib b
      | _ => BitVec.ofNat ib a * BitVec.ofNat ib b).toInt
    accepts ib name a b size == decide (0 ≤ lhs ∧ 0 ≤ rhs ∧ 0 ≤ result ∧ result < (size : Int)))

end MerlinVerify.Tests.IndexArithmetic
