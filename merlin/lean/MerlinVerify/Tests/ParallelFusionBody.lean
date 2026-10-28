/- Exact pointwise primitive composition retains poison and ignores only checked-unused outs. -/
import MerlinVerify.Check.ParallelFusionBody
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.ParallelFusionBody

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Tests.Semantics
open MerlinVerify.Rules.ParallelFusionScalar

def primitive (kind : Kind) (w input result : Nat) : Op :=
  mkOp kind.name [v result (.int w)] [input, input]

def fused (p c : Kind) (w : Nat := 13) : Region :=
  .mk [.mk [v 3 (.int w), v 4 (.int w)]
    [primitive p w 3 9, primitive c w 9 10, mkOp "linalg.yield" [] [10]]]

def scalar (w : Nat) (x : Int) : RVal := .scalar (.some (.bits w (BitVec.ofInt w x)))
def run (p c : Kind) (w : Nat) (x old : RVal) :=
  readInts (evalRegion opSem ctx (fun _ _ => .ub) [] (fused p c w) [x, old])

#guard run .add .mul 13 (scalar 13 3) (scalar 13 99) == some [[some 36]]
#guard run .add .mul 13 (scalar 13 3) (.scalar .poison) == some [[some 36]]
#guard run .add .mul 13 (.scalar .poison) (scalar 13 99) == some [[none]]
#guard run .add .mul 13 (.scalar .poison) (.scalar .poison) == some [[none]]
#guard run .add .mul 13 (scalar 13 4095) (scalar 13 99) == some [[some 4]]
#guard run .band .mul 13 (scalar 13 9) (scalar 13 99) == some [[some 81]]
#guard run .bxor .bor 13 (scalar 13 9) (scalar 13 99) == some [[some 0]]
#guard run .bor .add 13 (scalar 13 9) (scalar 13 99) == some [[some 18]]
#guard run .mul .band 13 (scalar 13 9) (scalar 13 99) == some [[some 81]]
#guard run .add .mul 1 (scalar 1 1) (scalar 1 0) == some [[some 0]]
#guard run .add .mul 65 (scalar 65 4294967297) (scalar 65 0) == some [[some 34359738372]]
#guard isStuck (evalRegion opSem ctx (fun _ _ => .ub) [] (fused .add .mul) [])
#guard isStuck (evalRegion opSem ctx (fun _ _ => .ub) [] (fused .add .mul) [scalar 13 3])
#guard kind? "arith.addf" == none
#guard kind? "arith.divsi" == none
#guard kind? "arith.addi" == some .add

end MerlinVerify.Tests.ParallelFusionBody
