/-
Build-time tests of the C1 checker with the DCE rule: it must accept correct boundaries (including
SSA renaming) and reject wrong ones. Rejection tests are as important as acceptance tests: a
checker that rejects everything would pass the soundness theorem vacuously (design §1.2 rule 7).
-/
import MerlinVerify.Rules.Dce
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.C1

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics

def i32v (id : Nat) : Value := v id i32
def nsw : List (String × Attr) := [("overflowFlags", .enum "arith" "overflow" "nsw")]

/-- f(a, b) = a + b, with a dead `a * b`. -/
def src : Module := { ops := [func "f" [i32v 0, i32v 1] [
  mkOp "arith.addi" [i32v 2] [0, 1],
  mkOp "arith.muli" [i32v 3] [0, 1],
  ret [2]]] }

/-- The same function after DCE, with every SSA id renumbered. -/
def tgtDce : Module := { ops := [func "f" [i32v 10, i32v 11] [
  mkOp "arith.addi" [i32v 12] [10, 11],
  ret [12]]] }

def dceScript : List Step := [.same, .site Rules.Dce.rule 1 0, .same]

-- Accepted: correct DCE under renaming.
#guard c1Check { src, tgt := tgtDce, fn := "f", script := dceScript }

-- Accepted: identity boundary (alpha-equivalent copy).
#guard c1Check { src, tgt := src, fn := "f", script := [.same, .same, .same] }

-- Rejected: the certificate's script does not describe the change.
#guard !c1Check { src, tgt := tgtDce, fn := "f", script := [.same, .same, .same] }
#guard !c1Check { src, tgt := tgtDce, fn := "f", script := [.site Rules.Dce.rule 1 0, .same, .same] }

/-- g(a, b) = (a + b) * a: the add is used. -/
def srcUsed : Module := { ops := [func "f" [i32v 0, i32v 1] [
  mkOp "arith.addi" [i32v 2] [0, 1],
  mkOp "arith.muli" [i32v 3] [2, 0],
  ret [3]]] }

/-- Removing the used add (the target reads an argument in its place). -/
def tgtDropUsed : Module := { ops := [func "f" [i32v 10, i32v 11] [
  mkOp "arith.muli" [i32v 13] [11, 10],
  ret [13]]] }

-- Rejected: DCE of a used value.
#guard !c1Check { src := srcUsed, tgt := tgtDropUsed, fn := "f",
                  script := [.site Rules.Dce.rule 1 0, .same, .same] }

/-- Same shape as `src`'s live part, but the add gained `nsw`. -/
def tgtFlag : Module := { ops := [func "f" [i32v 10, i32v 11] [
  mkOp "arith.addi" [i32v 12] [10, 11] nsw,
  ret [12]]] }

-- Rejected: an attribute changed outside any site.
#guard !c1Check { src, tgt := tgtFlag, fn := "f", script := dceScript }

/-- f(a, b) = a - b versus b - a. -/
def srcSub : Module := { ops := [func "f" [i32v 0, i32v 1] [mkOp "arith.subi" [i32v 2] [0, 1], ret [2]]] }
def tgtSwap : Module := { ops := [func "f" [i32v 10, i32v 11] [mkOp "arith.subi" [i32v 12] [11, 10], ret [12]]] }

-- Rejected: operands swapped.
#guard !c1Check { src := srcSub, tgt := tgtSwap, fn := "f", script := [.same, .same] }

/-- A call to `@dev`, declared in both modules: accepted when unchanged. -/
def decl (name : String) : Op :=
  mkOp "func.func" [] [] [] [("sym_name", .str name), ("sym_visibility", .str "private")] [.mk []]
def callOp (res : Value) (args : List ValueId) : Op :=
  mkOp "func.call" [res] args [] [("callee", .symbolRef "dev" [])]
def srcCall : Module := { ops := [decl "dev", func "f" [i32v 0] [callOp (i32v 1) [0], ret [1]]] }
def tgtCall : Module := { ops := [decl "dev", func "f" [i32v 7] [callOp (i32v 8) [7], ret [8]]] }
def tgtCallUndeclared : Module := { ops := [func "f" [i32v 7] [callOp (i32v 8) [7], ret [8]]] }

#guard c1Check { src := srcCall, tgt := tgtCall, fn := "f", script := [.same, .same] }
-- Rejected: the target calls a symbol it does not declare.
#guard !c1Check { src := srcCall, tgt := tgtCallUndeclared, fn := "f", script := [.same, .same] }

/-! ### DCE erases only ops that compute nothing but their results -/

/-- f(a) = a, with a dead call to the external `@dev` (its result unused). -/
def srcDeadCall : Module := { ops := [decl "dev", func "f" [i32v 0] [callOp (i32v 1) [0], ret [0]]] }
def tgtNoCall : Module := { ops := [decl "dev", func "f" [i32v 7] [ret [7]]] }

-- Rejected: the call may have effects the model does not see.
#guard !c1Check { src := srcDeadCall, tgt := tgtNoCall, fn := "f",
                  script := [.site Rules.Dce.rule 1 0, .same] }

def memrefTy : Ty := .memref [some 4] i32 none none

/-- f(m, a) = a, with a store into `m` and an op of an unknown dialect, both without results. -/
def srcEffects (eff : Op) : Module := { ops := [func "f" [v 0 memrefTy, i32v 1] [eff, ret [1]]] }
def tgtNoEffect : Module := { ops := [func "f" [v 10 memrefTy, i32v 11] [ret [11]]] }

#guard !c1Check { src := srcEffects (mkOp "memref.store" [] [1, 0]), tgt := tgtNoEffect, fn := "f",
                  script := [.site Rules.Dce.rule 1 0, .same] }
#guard !c1Check { src := srcEffects (mkOp "review.effect" [] [1]), tgt := tgtNoEffect, fn := "f",
                  script := [.site Rules.Dce.rule 1 0, .same] }
-- even an arith op is not erased when it produces no result
#guard !c1Check { src := srcEffects (mkOp "arith.addi" [] [1, 1]), tgt := tgtNoEffect, fn := "f",
                  script := [.site Rules.Dce.rule 1 0, .same] }

/-! ### Regions outside the single-block fragment never match -/

#guard !alphaRegion [] (.mk [.mk [] [mkOp "review.a" [] []], .mk [] []])
  (.mk [.mk [] [mkOp "review.b" [] []], .mk [] []])
#guard !alphaRegion [] (.mk [.mk [] [], .mk [] []]) (.mk [.mk [] [], .mk [] []])
#guard alphaRegion [] (.mk []) (.mk [])

end MerlinVerify.Tests.C1
