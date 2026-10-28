/- Pointwise provenance, repeated operands, precise invalidation and linear fact construction. -/
import MerlinVerify.Check.FromElementsFacts
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.FromElementsFacts

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics

def ctor (result : Nat) (inputs : List Nat) : Op :=
  mkOp "tensor.from_elements" [v result (tensorTy [inputs.length] i32)] inputs
def facts := Check.FromElementsFacts.learn (ctor 3 [0, 1, 0])

#guard facts.map (fun f => (f.scalar, f.coordinate)) == [(0, 0), (1, 1), (0, 2)]
#guard (Check.FromElementsFacts.learn (ctor 3 [])).isEmpty
#guard (Check.FromElementsFacts.learn (ctor 3 [0])).length == 1
#guard (Check.FromElementsFacts.learn (ctor 0 [0, 1])).map (fun f => (f.scalar, f.coordinate)) == [(1, 1)]
#guard (Check.FromElementsFacts.advance facts (mkOp "unknown.write" [v 0 i32] [])).map
  (fun f => f.coordinate) == [1]
#guard (Check.FromElementsFacts.advance facts (mkOp "unknown.write" [v 1 i32] [])).map
  (fun f => f.coordinate) == [0, 2]
#guard (Check.FromElementsFacts.advance facts (mkOp "unknown.write" [v 3 (tensorTy [3] i32)] [])).isEmpty
#guard (Check.FromElementsFacts.advance facts (mkOp "unknown.write" [v 99 i32] [])).length == 3
#guard (Check.FromElementsFacts.learn (mkOp "tensor.from_elements" [v 3 (tensorTy [] i32)] [0])).isEmpty
#guard (Check.FromElementsFacts.learn (mkOp "tensor.from_elements" [v 3 (tensorTy [9] i32)] [0])).isEmpty
#guard (Check.FromElementsFacts.learn (mkOp "tensor.from_elements" [v 3 (tensorTy [1] i32)] [0]
  [("unknown.meaning", .unit)])).isEmpty
#guard (Check.FromElementsFacts.learn (ctor 4096 (List.replicate 1024 0))).length == 1024

end MerlinVerify.Tests.FromElementsFacts
