/- Runtime-domain and provenance regression checks for exact scalar cast normalization. -/
import MerlinVerify.Rules.IntegerCast
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.IntegerCast

open MerlinVerify MerlinVerify.Check MerlinVerify.Rules.IntegerCast

def cast (k : Kind) (x r _a b : Nat) : Op :=
  .mk k.name [⟨r, .int b⟩] [x] [] [] [] []

def ret (ids : List Nat) : Op := .mk "func.return" [] ids [] [] [] []

def Γ : TEnv := [(0, .scalar (.int 1))]

def normalized (first second : Kind) : Option (List Op) :=
  normalize 64 Γ [] [cast first 0 1 1 8, cast second 1 2 8 32, ret [2]]

#guard ((normalized .zero .zero).bind (·[1]?)).map (·.operands) == some [0]
#guard ((normalized .sign .sign).bind (·[1]?)).map (·.operands) == some [0]
#guard ((normalized .zero .sign).bind (·[1]?)).map (·.name) == some "arith.extui"
#guard ((normalized .sign .zero).bind (·[1]?)).map (·.operands) == some [1]

#guard ((normalize 32 [(0, .scalar (.int 64))] []
  [cast .trunc 0 1 64 13, cast .trunc 1 2 13 1, ret [2]]).bind (·[1]?)).map (·.operands) == some [0]

-- Runtime scalar typing is required. A tensor declaration cannot establish scalar origins.
#guard (read [(0, .tensor [3] (.int 1))] (cast .zero 0 1 1 8)).isNone
#guard (read Γ (cast .zero 0 1 1 1)).isNone
#guard (read Γ (cast .trunc 0 1 1 8)).isNone
#guard (learn Γ (cast .zero 0 0 1 8)).isNone

-- Rebinding either dependency invalidates an old origin fact; a distinct unrelated write does not.
def fact : Fact := ⟨0, ⟨1, .int 8⟩, 1, 8, .zero⟩
#guard (BindingFacts.forget model [0] [fact]).isEmpty
#guard (BindingFacts.forget model [1] [fact]).isEmpty
#guard (BindingFacts.forget model [3] [fact]).length == 1

-- Chain learning follows the proved root, not merely the immediately preceding syntax.
#guard ((normalize 64 Γ [] [cast .zero 0 1 1 8, cast .sign 1 2 8 13,
  cast .zero 2 3 13 32, ret [3]]).bind (·[2]?)).map (·.operands) == some [0]

example : Kind.sign.apply 32 (Kind.zero.apply 8 .poison) = Kind.zero.apply 32 .poison := rfl
example : Kind.trunc.apply 1 (Kind.trunc.apply 13 .poison) = Kind.trunc.apply 1 .poison := rfl

-- The actual interpreter also preserves poison. Native execution never receives poison inputs.
def original : List Op := [cast .zero 0 1 1 8, cast .sign 1 2 8 32, ret [2]]
def run (ops : List Op) (value : Val) : Option (List (List (Option Int))) :=
  Semantics.readInts (evalOps Sem.Ops.opSem Semantics.ctx (fun _ _ => .stuck "no calls")
    [(0, .scalar value)] ops)
#guard run original .poison == some [[none]]
#guard run ((normalize 64 Γ [] original).getD []) .poison == some [[none]]

-- This deliberately mistyped runtime input distinguishes the programs. It demonstrates why a
-- universal C1 cast rule would be unsound, and why the checked runtime domain is essential.
#guard run original (.some (.bits 8 1)) == some [[none]]
#guard run ((normalize 64 Γ [] original).getD []) (.some (.bits 8 1)) == some [[some 1]]

def literal (id width : Nat) : Op := .mk "arith.constant" [⟨id, .int width⟩] []
  [("value", .int 0 (.int width))] [] [] []
#guard ((normalize 64 Γ [] [cast .zero 0 1 1 8, literal 0 1,
  cast .zero 1 2 8 32, ret [2]]).bind (·[2]?)).map (·.operands) == some [1]
#guard ((normalize 64 Γ [] [cast .zero 0 1 1 8, literal 1 8,
  cast .zero 1 2 8 32, ret [2]]).bind (·[2]?)).map (·.operands) == some [1]

end MerlinVerify.Tests.IntegerCast
