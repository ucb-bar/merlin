/- Exact index bits, closed bounds, poison, shadow invalidation, escaping tensors and captures. -/
import MerlinVerify.Check.FromElementsSelect
import MerlinVerify.Check.DefinedFromElements
import MerlinVerify.Tests.FromElementsFacts

namespace MerlinVerify.Tests.FromElementsSelect

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics
open MerlinVerify.Tests.FromElementsFacts

def Γ : TEnv := [(0, .scalar i32), (1, .scalar i32)]
def index (n : Int) : Op := mkOp "arith.constant" [v 5 .index] [] [("value", .int n .index)]
def load : Op := mkOp "tensor.extract" [v 6 i32] [3, 5]
def program (n : Int := 1) : List Op := [ctor 3 [0, 1], index n, load, ret [6]]
def postΓ : TEnv := [(5, .scalar .index), (3, .tensor [2] i32)] ++ Γ
def facts (n : Int := 1) := Check.FromElementsSelect.advance 64
  (Check.FromElementsSelect.advance 64 ⟨[], []⟩ (ctor 3 [0, 1])) (index n)
def result (ops : List Op) := (Check.FromElementsSelect.base 64).run 0 Γ [] ops [.scalar i32]
def proposeAlias (f : Check.FromElementsSelect.Facts) (tail : List Op := [ret [6]]) :=
  Check.FromElementsSelect.eliminate postΓ f load tail

#guard (result (program)).map (List.map Op.name) == some ["func.return"]
#guard (result (program)).map (fun ops => ops.getLast!.operands) == some [1]
#guard (result (program 0)).map (fun ops => ops.getLast!.operands) == some [0]
#guard (result (program (-1))).map (List.map Op.name) ==
  some ["tensor.from_elements", "arith.constant", "tensor.extract", "func.return"]
#guard (result (program 2)).map (List.map Op.name) ==
  some ["tensor.from_elements", "arith.constant", "tensor.extract", "func.return"]
#guard (proposeAlias (facts)).isSome
#guard (proposeAlias (facts (-1))).isNone
#guard (proposeAlias (facts 2)).isNone

#guard (proposeAlias (Check.FromElementsSelect.advance 64 (facts) (mkOp "unknown.write" [v 1 i32] []))).isNone
#guard (proposeAlias (Check.FromElementsSelect.advance 64 (facts) (mkOp "unknown.write" [v 0 i32] []))).isSome
#guard (proposeAlias (Check.FromElementsSelect.advance 64 (facts) (mkOp "unknown.write" [v 3 (tensorTy [2] i32)] []))).isNone
#guard (proposeAlias (Check.FromElementsSelect.advance 64 (facts) (mkOp "unknown.write" [v 5 .index] []))).isNone

#guard (proposeAlias (facts) [mkOp "arith.constant" [v 1 i32] [] [("value", .int 11 i32)], ret [6]]).isNone
#guard (proposeAlias (facts) [mkOp "arith.constant" [v 6 i32] [] [("value", .int 11 i32)], ret [6]]).isNone
#guard (proposeAlias (facts) [mkOp "unknown.capture" [] [] [] [] [.mk [.mk [v 1 i32] [ret [6]]]]]).isNone
#guard (proposeAlias (facts) [mkOp "unknown.capture" [] [] [] [] [.mk [.mk [v 6 i32] [ret [6]]]]]).isNone
#guard (proposeAlias (facts) [mkOp "unknown.capture" [] [] [] [] [.mk [.mk [v 99 i32] [ret [6]]]]]).isSome

#guard ((Check.FromElementsSelect.base 64).run 0 Γ []
  [ctor 3 [0, 1], index 1, load, ret [6, 3]] [.scalar i32, .tensor [2] i32]).map
  (List.map Op.name) == some ["tensor.from_elements", "func.return"]
#guard !(Check.FromElementsSelect.drop? 64 Γ (ctor 3 [0, 1])
  [mkOp "unknown.capture" [] [] [] [] [.mk [.mk [] [ret [3]]]]])
#guard !(Check.FromElementsSelect.drop? 64 ((3, .tensor [2] i32) :: Γ) (ctor 3 [0, 1]) [ret [1]])
#guard ((Check.FromElementsSelect.base 32).run 0 Γ [] (program 4294967297) [.scalar i32]).map
  (fun ops => ops.getLast!.operands) == some [1]
#guard (result (program 4294967297)).map (List.map Op.name) ==
  some ["tensor.from_elements", "arith.constant", "tensor.extract", "func.return"]

def isPoison : Outcome (List RVal) → Bool
  | .ok [.scalar .poison] => true
  | _ => false
#guard isPoison (evalOps Sem.Ops.opSem ctx (fun _ _ => .stuck "no calls")
  [(0, .scalar (.some (.bits 32 7))), (1, .scalar .poison)] (program))
#guard isPoison (evalOps Sem.Ops.opSem ctx (fun _ _ => .stuck "no calls")
  [(0, .scalar (.some (.bits 32 7))), (1, .scalar .poison)] [ret [1]])
#guard isUb (evalOps Sem.Ops.opSem ctx (fun _ _ => .stuck "no calls")
  [(0, .scalar (.some (.bits 32 7))), (1, .scalar .poison)] (program (-1)))

#guard (Definedness.FromElements.rule (ctor 3 []) []).isSome
#guard (Definedness.FromElements.rule (ctor 3 [0]) [⟨.scalar i32, true⟩]).isSome
#guard (Definedness.FromElements.rule (ctor 3 [0, 1])
  [⟨.scalar i32, true⟩, ⟨.scalar i32, true⟩]).isSome
#guard (Definedness.FromElements.rule (ctor 3 [0, 1])
  [⟨.scalar i32, true⟩, ⟨.scalar i32, false⟩]).isNone
#guard (Definedness.FromElements.rule (ctor 3 [0, 1])
  [⟨.scalar (.int 64), true⟩, ⟨.scalar i32, true⟩]).isNone
#guard (Definedness.FromElements.rule (ctor 3 [0, 1]) [⟨.scalar i32, true⟩]).isNone

end MerlinVerify.Tests.FromElementsSelect
