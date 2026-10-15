/- Positive and adversarial tests for closed-expression CSE and script synthesis. -/
import MerlinVerify.Check.Module
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.Cse

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics

def constant (id : Nat) (n : Int) (ty : Ty := i32) : Op :=
  mkOp "arith.constant" [v id ty] [] [("value", .int n ty)]

def src : Module := { ops := [func "f" [] [constant 0 7, constant 1 7, ret [0, 1]]] }
def target (n : Int := 7) : Module := { ops := [func "f" [] [constant 9 n, ret [9, 9]]] }

#guard moduleCheck src (target 7)
#guard !moduleCheck src (target 8)
#guard !moduleCheck { ops := [func "f" [] [constant 0 7, constant 1 8, ret [0, 1]]] } (target 7)
#guard !moduleCheck src { ops := [func "f" [] [constant 9 7 i8, ret [9, 9]]] }
#guard !moduleCheck { ops := [func "f" [v 2 i32] [constant 0 7, constant 1 7, ret [0, 1]]] }
  { ops := [func "f" [v 9 i32] [constant 9 7, ret [9, 9]]] }

-- DCE alone cannot remove the second constant: both source results are returned.
#guard !c1Check { src, tgt := target 7, fn := "f", script := [.same, .site Rules.Dce.rule 1 0, .same] }
#guard c1Check { src, tgt := target 7, fn := "f", script := [.site Rules.Cse.rule 2 1, .same] }

-- Both source results remain available regardless of their return order.
#guard moduleCheck { ops := [func "f" [] [constant 0 7, constant 1 7, ret [1, 0]]] } (target 7)

-- Calls and operand-dependent computations are outside this closed-expression rule.
#guard (Rules.Cse.check [] [mkOp "func.call" [v 0 i32] [] [] [("callee", .symbolRef "dev" [])],
  mkOp "func.call" [v 1 i32] [] [] [("callee", .symbolRef "dev" [])]]
  [mkOp "func.call" [v 2 i32] [] [] [("callee", .symbolRef "dev" [])]]).isNone
#guard (Rules.Cse.check [(0, 10), (1, 11)]
  [mkOp "arith.addi" [v 2 i32] [0, 1], mkOp "arith.addi" [v 3 i32] [0, 1]]
  [mkOp "arith.addi" [v 12 i32] [10, 11]]).isNone

end MerlinVerify.Tests.Cse
