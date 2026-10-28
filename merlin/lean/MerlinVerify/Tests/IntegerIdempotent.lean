/- Runtime type, poison, escaping-use and capture-scope checks for exact idempotent aliases. -/
import MerlinVerify.Rules.IntegerIdempotent
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.IntegerIdempotent

open MerlinVerify MerlinVerify.Check MerlinVerify.Rules.IntegerIdempotent

def Γ : TEnv := [(0, .scalar (.int 13)), (1, .scalar (.int 13))]
def binary (name : String) (x y r : Nat) : Op := .mk name [⟨r, .int 13⟩] [x,y] [] [] [] []
def ret (ids : List Nat) : Op := .mk "func.return" [] ids [] [] [] []
def source (name : String) : List Op := [binary name 0 0 2, binary "arith.addi" 2 1 3, ret [3,1]]

#guard names.all fun name => ((normalize 64 Γ (source name)).bind (·[0]?)).map (·.operands) == some [0,1]
#guard names.all fun name => (normalize 64 Γ (source name)).map List.length == some 2
#guard ((normalize 64 Γ [binary "arith.andi" 0 0 2, ret [2,2,1]]).bind (·[0]?)).map (·.operands) == some [0,0,1]
#guard (eliminate Γ (binary "arith.andi" 0 1 2) [ret [2]]).isNone
#guard (eliminate [] (binary "arith.andi" 0 0 2) [ret [2]]).isNone
#guard (eliminate [(0,.scalar (.int 8))] (binary "arith.andi" 0 0 2) [ret [2]]).isNone
#guard (eliminate [(0,.tensor [3] (.int 13))] (binary "arith.andi" 0 0 2) [ret [2]]).isNone
#guard (eliminate [(0,.scalar .index)] (binary "arith.andi" 0 0 2) [ret [2]]).isNone

def flagged : Op := .mk "arith.andi" [⟨2,.int 13⟩] [0,0]
  [("overflowFlags",.enum "arith" "overflow" "nsw")] [] [] []
#guard (eliminate Γ flagged [ret [2]]).isNone

def nested (arg : Nat) : Op := .mk "test.region" [⟨4,.int 13⟩] [1] [] []
  [.mk [.mk [⟨arg,.int 13⟩] [ret [2]]]] []
#guard (eliminate Γ (binary "arith.ori" 0 0 2) [nested 3,ret [4]]).isSome
-- A textual replacement that captures a shadowing region argument is rejected by alphaOps.
#guard (eliminate Γ (binary "arith.ori" 0 0 2) [nested 2,ret [4]]).isNone
-- Rebinding the aliased root in the continuation must not change its new meaning.
#guard (eliminate Γ (binary "arith.ori" 0 0 2) [binary "arith.addi" 1 1 0,ret [2]]).isNone

def run (ops : List Op) (value : Val) : Option (List (List (Option Int))) :=
  Semantics.readInts (evalOps Sem.Ops.opSem Semantics.ctx (fun _ _ => .stuck "no calls")
    [(0,.scalar value),(1,.scalar (.some (.bits 13 7)))] ops)
#guard names.all fun name => run (source name) .poison == some [[none],[some 7]]
#guard names.all fun name => run ((normalize 64 Γ (source name)).getD []) .poison == some [[none],[some 7]]
#guard names.all fun name => run ((normalize 64 Γ (source name)).getD []) (.some (.bits 13 19)) == some [[some 26],[some 7]]

end MerlinVerify.Tests.IntegerIdempotent
