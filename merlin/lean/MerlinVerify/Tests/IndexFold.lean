/- Checked-width replay, hint independence, flags, stale bindings and excluded width-sensitive ops. -/
import MerlinVerify.Rules.IndexFold
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.IndexFold

open MerlinVerify MerlinVerify.Check MerlinVerify.Rules.IndexFold

def constant (id : Nat) (n : Int) : Op := .mk "arith.constant" [⟨id,.index⟩] [] [("value",.int n .index)] [] [] []
def binary (name : String) (x y r : Nat) (attrs : List (String × Attr) := []) : Op :=
  .mk name [⟨r,.index⟩] [x,y] attrs [] [] []
def add := binary "arith.addi" 1 2 3
def facts (ib : Nat) (a b : Int) := advance ib (advance ib [] (constant 1 a)) (constant 2 b)
def bits : Option RVal → Option Nat
  | some (.scalar (.some (.bits _ x))) => some x.toNat
  | _ => none
def materialize (ib : Nat) (a b target : Int) := replacement? ib (facts ib a b) [constant 99 target] add

#guard bits (value? 32 (facts 32 17 5) add) == some 22
#guard bits (value? 64 (facts 64 17 5) add) == some 22
#guard (materialize 32 17 5 22).isSome
#guard (materialize 64 17 5 23).isNone
#guard (materialize 32 4294967295 1 0).isSome
#guard (materialize 64 4294967295 1 0).isNone
#guard (materialize 64 4294967295 1 4294967296).isSome
#guard (value? 32 [] add).isNone
#guard (value? 32 (facts 32 17 5) (binary "arith.divui" 1 2 3)).isNone
#guard (value? 32 (facts 32 17 5) (binary "arith.shli" 1 2 3)).isNone
#guard (value? 32 (facts 32 17 5) (.mk "arith.cmpi" [⟨3,.int 1⟩] [1,2]
  [("predicate",.int 2 (.int 64))] [] [] [])).isNone
#guard (value? 32 (facts 32 17 5) (binary "arith.addi" 1 2 3 [("unknown.meaning",.unit)])).isNone

-- Source computations accumulate independently of absent intermediate target materializations.
def chain := [constant 1 17, constant 2 5, add, binary "arith.muli" 3 2 4,
  binary "arith.subi" 4 1 5, .mk "func.return" [] [5] [] [] [] []]
def normalized := FactNormalize.normalize family 32 [] [] [constant 99 93] chain
#guard (normalized.bind (·[2]?)).map (·.name) == some "arith.addi"
#guard (normalized.bind (·[3]?)).map (·.name) == some "arith.muli"
#guard (normalized.bind (·[4]?)).map (·.name) == some "arith.constant"
#guard (normalized.bind (·[4]?)).map (·.results.map (·.id)) == some [5]
#guard (normalized.bind (·[4]?)).map (·.operands) == some []

-- Every source write invalidates stale facts, even when the writer is unrecognized.
def clobber : Op := .mk "unknown.write" [⟨1,.index⟩] [] [] [] [] []
#guard (lookup (advance 32 (facts 32 17 5) clobber) 1).isNone
#guard bits (lookup (advance 32 (facts 32 17 5) (constant 1 23)) 1) == some 23
#guard (value? 32 (advance 32 (facts 32 17 5) clobber) add).isNone

-- Invalid overflow promises produce poison in the actual semantics and cannot become bits.
def flagged := binary "arith.addi" 1 2 3 [("overflowFlags",.enum "arith" "overflow" "nsw")]
#guard (match value? 32 (facts 32 2147483647 1) flagged with | some (.scalar .poison) => true | _ => false)
#guard (replacement? 32 (facts 32 2147483647 1) [constant 99 2147483648] flagged).isNone
#guard (replacement? 64 (facts 64 2147483647 1) [constant 99 2147483648] flagged).isSome
#guard (tryHints 32 add (.some (.bits 32 22)) [constant 99 23]).isNone
#guard (tryHints 64 add (.some (.bits 32 22)) [constant 99 22]).isNone

end MerlinVerify.Tests.IndexFold
