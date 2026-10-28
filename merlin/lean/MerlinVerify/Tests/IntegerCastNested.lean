/- Dead cast cleanup preserves poison, captures and escaping intermediate bindings. -/
import MerlinVerify.Rules.IntegerCastNested
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.IntegerCastNested

open MerlinVerify MerlinVerify.Check MerlinVerify.Rules

def Γ : TEnv := [(0,.scalar (.int 1))]
def cast (k : IntegerCast.Kind) (x r _a b : Nat) : Op := .mk k.name [⟨r,.int b⟩] [x] [] [] [] []
def ret (xs : List Nat) : Op := .mk "func.return" [] xs [] [] [] []
def source : List Op := [cast .zero 0 1 1 8, cast .sign 1 2 8 32, ret [2]]
def normalized : Option (List Op) := IntegerCastNested.localNormalizer.run 64 0 Γ source [] [.scalar (.int 32)]

#guard normalized.map List.length == some 2
#guard (normalized.bind (·[0]?)).map (·.name) == some "arith.extui"
#guard (normalized.bind (·[0]?)).map (·.operands) == some [0]
#guard IntegerCastDead.drop? Γ (cast .zero 0 1 1 8) [ret [0]]
#guard !IntegerCastDead.drop? Γ (cast .zero 0 1 1 8) [ret [1]]
#guard !IntegerCastDead.drop? [(0,.scalar (.int 1)),(1,.scalar (.int 8))] (cast .zero 0 1 1 8) [ret [0]]
#guard !IntegerCastDead.drop? Γ (cast .zero 0 0 1 8) [ret []]
#guard !IntegerCastDead.total Γ (cast .trunc 0 1 1 8)
#guard !IntegerCastDead.total [(0,.tensor [3] (.int 1))] (cast .zero 0 1 1 8)

def escaped : List Op := [cast .zero 0 1 1 8, cast .sign 1 2 8 32, ret [2,1]]
#guard (IntegerCastNested.localNormalizer.run 64 0 Γ escaped [] [.scalar (.int 32),.scalar (.int 8)]).map List.length == some 3

def capture : Op := .mk "test.region" [⟨4,.int 8⟩] [0] [] []
  [.mk [.mk [⟨3,.int 1⟩] [ret [1]]]] []
#guard !IntegerCastDead.drop? Γ (cast .zero 0 1 1 8) [capture,ret [4]]

def nonneg : Op := .mk "arith.extui" [⟨1,.int 8⟩] [0] [("nonNeg",.unit)] [] [] []
#guard !IntegerCastDead.total Γ nonneg
#guard !IntegerCastDead.drop? Γ nonneg [ret [0]]

def run (ops : List Op) (value : Val) : Option (List (List (Option Int))) :=
  Semantics.readInts (evalOps Sem.Ops.opSem Semantics.ctx (fun _ _ => .stuck "no calls") [(0,.scalar value)] ops)
#guard run source .poison == some [[none]]
#guard run (normalized.getD []) .poison == some [[none]]
#guard run (normalized.getD []) (.some (.bits 1 1)) == some [[some 1]]

end MerlinVerify.Tests.IntegerCastNested
