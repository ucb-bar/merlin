/- Independent operand-taking motion: actual SSA dependencies, escaping values and failure paths. -/
import MerlinVerify.Rules.MotionIndependent
import MerlinVerify.Tests.Cse

namespace MerlinVerify.Tests.MotionIndependent

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics MerlinVerify.Tests.Cse

def cast (result input : Nat) : Op := mkOp "arith.extsi" [v result i32] [input]
def rho : Rho := [(0, 10)]
def sourceOps : List Op := [cast 1 0, constant 2 22]
def targetOps : List Op := [constant 11 22, cast 12 10]

#guard (Rules.MotionIndependent.check rho sourceOps targetOps).isSome
#guard (Rules.MotionIndependent.check rho [mkOp "arith.addi" [v 1 i32] [0, 0],
  mkOp "arith.subi" [v 2 i32] [0, 0]] [mkOp "arith.subi" [v 11 i32] [10, 10],
  mkOp "arith.addi" [v 12 i32] [10, 10]]).isSome
#guard (Rules.MotionIndependent.check [] [constant 1 7, constant 2 8]
  [constant 11 8, constant 12 7]).isSome

def src : Module := { ops := [func "f" [v 0 i8] (sourceOps ++ [ret [1, 2]])] }
def tgt : Module := { ops := [func "f" [v 10 i8] (targetOps ++ [ret [12, 11]])] }
#guard c1Check { src, tgt, fn := "f", script := [.site Rules.MotionIndependent.rule 2 2, .same] }
def wrongUses : Module := { ops := [func "f" [v 10 i8] (targetOps ++ [ret [11, 12]])] }
#guard !c1Check { src, tgt := wrongUses, fn := "f", script := [.site Rules.MotionIndependent.rule 2 2, .same] }
-- The old closed rule retains its original narrower API.
#guard (Rules.Motion.check rho sourceOps targetOps).isNone
-- Neither an ordinary producer dependency nor an anti-dependency through an old binding may move.
#guard (Rules.MotionIndependent.check rho [cast 1 0, mkOp "arith.addi" [v 2 i32] [1, 1]]
  [mkOp "arith.addi" [v 11 i32] [12, 12], cast 12 10]).isNone
#guard (Rules.MotionIndependent.check [(2, 20)]
  [mkOp "arith.addi" [v 1 i32] [2, 2], constant 2 22]
  [constant 11 22, mkOp "arith.addi" [v 12 i32] [11, 11]]).isNone
-- Rebinding either own operand or a sibling result is conservatively refused on raw ASTs.
#guard (Rules.MotionIndependent.check [(1, 10)] [cast 1 1, constant 2 22]
  [constant 11 22, cast 12 10]).isNone
#guard (Rules.MotionIndependent.check [(2, 10)] [constant 1 22, cast 2 2]
  [cast 11 10, constant 12 22]).isNone
#guard (Rules.MotionIndependent.check rho [cast 1 0, constant 1 22] targetOps).isNone
-- Actual target operands, constants, types and fresh result IDs must match in target order.
#guard (Rules.MotionIndependent.check rho sourceOps [constant 11 23, cast 12 10]).isNone
#guard (Rules.MotionIndependent.check rho sourceOps [constant 11 22 i8, cast 12 10]).isNone
#guard (Rules.MotionIndependent.check [(0, 10), (3, 20)] sourceOps [constant 11 22, cast 12 20]).isNone
#guard (Rules.MotionIndependent.check rho sourceOps [constant 11 22, cast 11 10]).isNone
#guard (Rules.MotionIndependent.check rho sourceOps [constant 10 22, cast 12 10]).isNone
#guard (Rules.MotionIndependent.check rho sourceOps [constant 11 22,
  mkOp "arith.extsi" [v 12 i32] [10] [("unreviewed", .unit)]]).isNone
#guard (Rules.MotionIndependent.check rho sourceOps [constant 11 22, cast 12 10, constant 13 9]).isNone
-- Calls, regions, effectful operations, terminators and multiple results remain excluded.
#guard (Rules.MotionIndependent.check rho
  [mkOp "func.call" [v 1 i32] [0] [] [("callee", .symbolRef "dev" [])], constant 2 22]
  [constant 11 22, mkOp "func.call" [v 12 i32] [10] [] [("callee", .symbolRef "dev" [])]]).isNone
#guard (Rules.MotionIndependent.check rho [mkOp "linalg.generic" [v 1 i32] [0] [] [] [.mk []], constant 2 22]
  [constant 11 22, mkOp "linalg.generic" [v 12 i32] [10] [] [] [.mk []]]).isNone
#guard (Rules.MotionIndependent.check rho [mkOp "cf.assert" [v 1 i32] [0], constant 2 22]
  [constant 11 22, mkOp "cf.assert" [v 12 i32] [10]]).isNone
#guard (Rules.MotionIndependent.check rho [ret [0], constant 2 22] [constant 11 22, ret [10]]).isNone
#guard (Rules.MotionIndependent.check rho [mkOp "arith.extsi" [v 1 i32, v 3 i32] [0], constant 2 22]
  [constant 11 22, mkOp "arith.extsi" [v 12 i32, v 13 i32] [10]]).isNone
-- Newest bindings win even with unrelated stale/shadowed incoming names. Both results escape.
def sourceEnv : Env := [(0, .scalar (.some (.bits 8 (BitVec.ofInt 8 (-3))))),
  (0, .scalar (.some (.bits 8 99)))]
def targetEnv : Env := [(10, .scalar (.some (.bits 8 (BitVec.ofInt 8 (-3))))),
  (10, .scalar (.some (.bits 8 88)))]
#guard readInts (evalOps Sem.Ops.opSem ctx (fun _ _ => .stuck "no calls") sourceEnv
  (sourceOps ++ [ret [1, 2]])) == some [[some (-3)], [some 22]]
#guard readInts (evalOps Sem.Ops.opSem ctx (fun _ _ => .stuck "no calls") targetEnv
  (targetOps ++ [ret [12, 11]])) == some [[some (-3)], [some 22]]
-- Accepted conditional sites can have missing SSA values or UB. Strict source checks are separate.
#guard (Rules.MotionIndependent.check rho [mkOp "arith.divsi" [v 1 i32] [0, 0], constant 2 22]
  [constant 11 22, mkOp "arith.divsi" [v 12 i32] [10, 10]]).isSome
#guard isUb (evalOps Sem.Ops.opSem ctx (fun _ _ => .stuck "no calls")
  [(0, .scalar (.some (.bits 32 0)))]
  [mkOp "arith.divsi" [v 1 i32] [0, 0], constant 2 22, ret [1, 2]])
#guard isStuck (evalOps Sem.Ops.opSem ctx (fun _ _ => .stuck "no calls") [] (sourceOps ++ [ret [1, 2]]))
#guard readInts (evalOps Sem.Ops.opSem ctx (fun _ _ => .stuck "no calls") [(0, .scalar .poison)]
  (sourceOps ++ [ret [1, 2]])) == some [[none], [some 22]]

end MerlinVerify.Tests.MotionIndependent
