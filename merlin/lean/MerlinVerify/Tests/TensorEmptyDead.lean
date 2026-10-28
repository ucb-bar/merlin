/- Liveness includes nested captures, shadowed identifiers, and every remaining use. -/
import MerlinVerify.Check.TensorEmptyDead
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.TensorEmptyDead

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics
open MerlinVerify.Check.TensorEmptyDead

def ty (shape : List Nat) : Ty := .tensor (shape.map some) (.int 13) none
def empty (shape : List Nat) : Op := .mk "tensor.empty" [⟨1, ty shape⟩] [] [] [] [] []
def Γ : TEnv := [(0, .scalar (.int 13))]
def ret (ids : List Nat) : Op := .mk "func.return" [] ids [] [] [] []
def nested (args : List Value) (uses : List Nat) : Op :=
  .mk "test.region" [] [] [] [] [.mk [.mk args [ret uses]]] []

#guard [[], [0], [1], [2, 0, 3], [2, 3]].all fun sh => total (empty sh)
#guard [[], [0], [1], [2, 0, 3], [2, 3]].all fun sh => drop? Γ (empty sh) [ret [0]]
#guard (normalize 64 Γ [empty [], ret [0]]).map (fun ss => ss.map (·.name)) == some ["func.return"]
#guard (normalize 32 Γ [empty [2, 0, 3], ret [0]]).map List.length == some 1

-- Direct uses and captures preserve the produced poison tensor; shadowed region arguments do not.
#guard !drop? Γ (empty []) [ret [1]]
#guard !drop? Γ (empty []) [nested [⟨2, .int 13⟩] [1], ret [0]]
#guard drop? Γ (empty []) [nested [⟨1, .int 13⟩] [1], ret [0]]
#guard !drop? Γ (empty []) [nested [⟨2, .int 13⟩] [1], ret [1]]
#guard !drop? ((1, .tensor [] (.int 13)) :: Γ) (empty []) [ret [0]]
#guard (normalize 64 Γ [empty [], ret [1]]).map List.length == some 2

-- A later binding may shadow an earlier unused one; the whole continuation is checked.
def replacement : Op := .mk "arith.constant" [⟨1, .int 13⟩] [] [("value", .int 7 (.int 13))] [] [] []
#guard drop? Γ (empty []) [replacement, ret [1]]
#guard !drop? Γ (empty []) [nested [] [1], replacement, ret [1]]

-- Dynamic sizes, encodings, nondefault headers and malformed operations are retained/refused.
#guard !total (.mk "tensor.empty" [⟨1, .tensor [none] (.int 13) none⟩] [0] [] [] [] [])
#guard !total (.mk "tensor.empty" [⟨1, .tensor [some 2] (.int 13) (some (.str "layout"))⟩] [] [] [] [] [])
#guard !total (.mk "tensor.empty" [⟨1, ty []⟩] [0] [] [] [] [])
#guard !total (.mk "tensor.empty" [] [] [] [] [] [])
#guard !total (.mk "tensor.empty" [⟨1, ty []⟩, ⟨2, ty []⟩] [] [] [] [] [])
#guard !total (.mk "tensor.empty" [⟨1, .int 13⟩] [] [] [] [] [])
#guard !total (.mk "tensor.empty" [⟨1, ty []⟩] [] [("unknown", .unit)] [] [] [])
#guard !total (.mk "tensor.empty" [⟨1, ty []⟩] [] [] [("unknown", .unit)] [] [])
#guard !total (.mk "tensor.empty" [⟨1, ty []⟩] [] [] [] [.mk []] [])
#guard !total (.mk "tensor.empty" [⟨1, ty []⟩] [] [] [] [] [4])
#guard !total (.mk "tensor.splat" [⟨1, ty []⟩] [] [] [] [] [])

-- Exact normalization preserves poison, successful returns and a failing continuation alike.
def run (ss : List Op) := evalOps Sem.Ops.opSem ctx (fun _ _ => .stuck "no calls")
  [(0, .scalar .poison)] ss
#guard readInts (run [empty [], ret [0]]) == readInts (run [ret [0]])
#guard readInts (run [empty [], ret [1]]) == some [[none]]
#guard isStuck (run [empty [], ret [9]])
#guard (normalize 64 Γ [empty [], ret [9]]).map List.length == some 2

end MerlinVerify.Tests.TensorEmptyDead
