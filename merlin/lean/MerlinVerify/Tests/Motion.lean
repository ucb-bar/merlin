/- Closed-expression motion: checked bindings, unchanged meaning, and conservative refusals. -/
import MerlinVerify.Rules.Motion
import MerlinVerify.Tests.Cse

namespace MerlinVerify.Tests.Motion

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics MerlinVerify.Tests.Cse

def src : Module := { ops := [func "f" [] [constant 0 7, constant 1 8, ret [0, 1]]] }
def tgt : Module := { ops := [func "f" [] [constant 10 8, constant 11 7, ret [11, 10]]] }

#guard c1Check { src, tgt, fn := "f", script := [.site Rules.Motion.rule 2 2, .same] }
-- Both bindings escape; changing which result escapes must fail after the motion site.
def wrongUses : Module := { ops := [func "f" [] [constant 10 8, constant 11 7, ret [10, 11]]] }
#guard !c1Check { src, tgt := wrongUses, fn := "f", script := [.site Rules.Motion.rule 2 2, .same] }
#guard (Rules.Motion.check [] [constant 0 7, constant 1 8]
  [constant 10 8, constant 11 9]).isNone
#guard (Rules.Motion.check [] [constant 0 7, constant 1 8]
  [constant 10 8 i8, constant 11 7]).isNone
-- Rebinding a source id makes source binding order observable, even for closed expressions.
#guard (Rules.Motion.check [] [constant 0 7, constant 0 8]
  [constant 10 8, constant 11 7]).isNone
#guard (Rules.Motion.check [] [constant 0 7, constant 1 8]
  [constant 10 8, constant 10 7]).isNone
#guard (Rules.Motion.check [(2, 10)] [constant 0 7, constant 1 8]
  [constant 10 8, constant 11 7]).isNone
#guard (Rules.Motion.check [] [constant 0 7, constant 1 8]
  [constant 10 8, constant 11 7, constant 12 9]).isNone
-- Even independent operand-taking expressions need a separate motion proof.
#guard (Rules.Motion.check [(2, 12)]
  [constant 0 7, mkOp "arith.addi" [v 1 i32] [2, 2]]
  [mkOp "arith.addi" [v 10 i32] [12, 12], constant 11 7]).isNone
-- Calls, effects, and region closures are excluded, including zero-argument calls.
#guard (Rules.Motion.check [] [constant 0 7,
  mkOp "func.call" [v 1 i32] [] [] [("callee", .symbolRef "dev" [])]]
  [mkOp "func.call" [v 10 i32] [] [] [("callee", .symbolRef "dev" [])], constant 11 7]).isNone
#guard (Rules.Motion.check [] [constant 0 7, mkOp "cf.assert" [] []]
  [mkOp "cf.assert" [] [], constant 11 7]).isNone
#guard (Rules.Motion.check [] [constant 0 7,
  mkOp "linalg.generic" [v 1 i32] [] [] [] [.mk []]]
  [mkOp "linalg.generic" [v 10 i32] [] [] [] [.mk []], constant 11 7]).isNone

end MerlinVerify.Tests.Motion
