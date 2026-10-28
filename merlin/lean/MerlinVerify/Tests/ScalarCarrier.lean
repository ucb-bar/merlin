/- Actual producer facts, poison, stale writes, escaping tensors and alias scope boundaries. -/
import MerlinVerify.Check.ScalarCarrier
import MerlinVerify.Check.DefinedCarrier
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.ScalarCarrier

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics

def Γ : TEnv := [(0, .scalar i32)]
def ctor : Op := mkOp "tensor.from_elements" [v 3 (tensorTy [] i32)] [0]
def postΓ : TEnv := (3, .tensor [] i32) :: Γ
def extract : Op := mkOp "tensor.extract" [v 4 i32] [3]
def program : List Op := [ctor, extract, ret [4]]
def facts := Check.ScalarCarrier.advance Γ [] ctor
def result (ops : List Op) := (Check.ScalarCarrier.base 64).run 0 Γ [] ops [.scalar i32]

#guard facts == [(⟨3, 0, [], i32⟩ : UniformFacts.Fact)]
#guard (result program).map (List.map Op.name) == some ["func.return"]
#guard (result program).map (fun ops => ops.getLast!.operands) == some [0]
#guard (result [ctor, extract, ret [4, 3]]).isNone -- mismatched declared yields, not a proved program
#guard ((Check.ScalarCarrier.base 64).run 0 Γ [] [ctor, extract, ret [4, 3]]
  [.scalar i32, .tensor [] i32]).map (List.map Op.name) == some ["tensor.splat", "func.return"]

#guard (Check.ScalarCarrier.advance Γ facts (mkOp "unknown.write" [v 0 i32] [])).isEmpty
#guard (Check.ScalarCarrier.advance Γ facts (mkOp "unknown.write" [v 3 (tensorTy [] i32)] [])).isEmpty
#guard (Check.ScalarCarrier.advance Γ facts (mkOp "unknown.write" [v 99 i32] [])) == facts
#guard (Check.ScalarCarrier.advance [] [] ctor).isEmpty
#guard (Check.ScalarCarrier.advance [(0, .scalar (.int 64))] [] ctor).isEmpty
#guard (Check.ScalarCarrier.eliminate postΓ [] extract [ret [4]]).isNone

-- Rebinding either the payload or alias name cannot capture the proposed substitution.
#guard (Check.ScalarCarrier.eliminate postΓ facts extract
  [mkOp "arith.constant" [v 0 i32] [] [("value", .int 11 i32)], ret [4]]).isNone
#guard (Check.ScalarCarrier.eliminate postΓ facts extract
  [mkOp "arith.constant" [v 4 i32] [] [("value", .int 11 i32)], ret [4]]).isNone
#guard (Check.ScalarCarrier.eliminate postΓ facts extract
  [mkOp "unknown.capture" [] [] [] [] [.mk [.mk [v 4 i32] [ret [4]]]]]).isNone
#guard (Check.ScalarCarrier.eliminate postΓ facts extract
  [mkOp "unknown.capture" [] [] [] [] [.mk [.mk [v 0 i32] [ret [4]]]]]).isNone
#guard (Check.ScalarCarrier.eliminate postΓ facts extract
  [mkOp "unknown.capture" [] [] [] [] [.mk [.mk [v 9 i32] [ret [4]]]]]).isSome

def poisonResult : Outcome (List RVal) → Bool
  | .ok [.scalar .poison] => true
  | _ => false
#guard poisonResult (evalOps Sem.Ops.opSem ctx (fun _ _ => .stuck "no calls") [(0, .scalar .poison)] program)
#guard poisonResult (evalOps Sem.Ops.opSem ctx (fun _ _ => .stuck "no calls") [(0, .scalar .poison)] [ret [0]])

-- A positive-rank coordinate remains observable. Exact normalization must preserve its UB.
def positive : List Op := [mkOp "tensor.from_elements" [v 3 (tensorTy [1] i32)] [0],
  mkOp "tensor.extract" [v 4 i32] [3, 1], ret [4]]
def indexedΓ : TEnv := (1, .scalar .index) :: Γ
#guard ((Check.ScalarCarrier.base 64).run 0 indexedΓ [] positive [.scalar i32]).map
  (List.map Op.name) == some ["tensor.from_elements", "tensor.extract", "func.return"]
#guard isUb (evalOps Sem.Ops.opSem ctx (fun _ _ => .stuck "no calls")
  [(0, .scalar (.some (.bits 32 7))), (1, .scalar (.some (.bits 64 1)))] positive)

#guard (Definedness.Carrier.rule ctor [⟨.scalar i32, true⟩]).isSome
#guard (Definedness.Carrier.rule ctor [⟨.scalar i32, false⟩]).isNone
#guard (Definedness.Carrier.rule ctor [⟨.scalar (.int 64), true⟩]).isNone
#guard (Definedness.Carrier.rule (mkOp "tensor.from_elements" [v 3 (tensorTy [1] i32)] [0])
  [⟨.scalar i32, true⟩]).isNone

end MerlinVerify.Tests.ScalarCarrier
