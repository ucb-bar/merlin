/- Source-derived computed coordinates, including absent target hints and shadowing. -/
import MerlinVerify.Tests.FromElementsSelect

namespace MerlinVerify.Tests.FromElementsIndex

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics
open MerlinVerify.Tests.FromElementsSelect
open MerlinVerify.Tests.FromElementsFacts

def number (id : Nat) (n : Int) : Op := mkOp "arith.constant" [v id .index] [] [("value", .int n .index)]
def binary (name : String) (left right : Nat := 7) : Op :=
  mkOp name [v 5 .index] [left, right]
def initialOps : List Op := [ctor 3 [0, 1], number 7 0, number 8 1]
def learned (ib : Nat) (ops : List Op) : Check.FromElementsSelect.Facts :=
  ops.foldl (Check.FromElementsSelect.advance ib) ⟨[], []⟩
def computed (ib : Nat) (s : Op) := proposeAlias (learned ib (initialOps ++ [s]))

#guard (computed 64 (binary "arith.addi" 7 8)).isSome
#guard (computed 32 (binary "arith.addi" 7 8)).isSome
#guard (computed 64 (binary "arith.subi" 8 7)).isSome
#guard (computed 64 (binary "arith.muli" 8 8)).isSome
#guard (computed 64 (binary "arith.andi" 8 8)).isSome
#guard (computed 64 (binary "arith.ori" 7 8)).isSome
#guard (computed 64 (binary "arith.xori" 7 8)).isSome
#guard (computed 64 (binary "arith.addi" 8 8)).isNone -- Actual coordinate two is OOB.
#guard (computed 64 (binary "arith.subi" 7 8)).isNone -- Actual coordinate is negative.
#guard (computed 64 (binary "arith.addi" 9 8)).isNone -- An annotation supplies no value fact.
#guard (computed 64 (mkOp "arith.addi" [v 5 .index] [7, 8]
  [("unreviewed.meaning", .unit)])).isNone

#guard (proposeAlias (learned 64 (initialOps ++ [binary "arith.addi" 7 8,
  mkOp "unknown.write" [v 5 .index] []]))).isNone
#guard (proposeAlias (learned 64 (initialOps ++ [mkOp "unknown.write" [v 8 .index] [],
  binary "arith.addi" 7 8]))).isNone
#guard (proposeAlias (learned 64 (initialOps ++ [binary "arith.addi" 7 8,
  mkOp "unknown.write" [v 8 .index] []]))).isSome -- Final binding survives an old input write.
#guard (proposeAlias (learned 64 [ctor 3 [0, 1], number 7 4294967296, number 8 1,
  binary "arith.addi" 7 8])).isNone
#guard (proposeAlias (learned 32 [ctor 3 [0, 1], number 7 4294967296, number 8 1,
  binary "arith.addi" 7 8])).isSome

end MerlinVerify.Tests.FromElementsIndex
