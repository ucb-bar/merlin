/- Captured uniform facts are masked by region arguments and by actual source rebinding.
These raw AST probes exercise shadowing cases that valid MLIR normally forbids. -/
import MerlinVerify.Check.SplatNested
import MerlinVerify.Tests.SplatC1

namespace MerlinVerify.Tests.SplatNested

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics
open MerlinVerify.Check.SplatNested

def environment : TEnv := [(0, .scalar i32), (1, .tensor [2] i32), (3, .tensor [2] i32)]
def facts : List UniformFacts.Fact := [⟨3, 0, [2], i32⟩]
def signature : RegionSig := ⟨[.scalar i32], [.tensor [2] i32], 0⟩

def body (arg : ValueId) (leading : List Op := []) : Region :=
  .mk [.mk [v arg i32] (leading ++ [SplatC1.generic 3 false, ret [4]])]

def operands (r : Option Region) : Option (List ValueId) :=
  r.bind fun r => match r with
    | .mk [.mk _ ops] => (ops.find? fun op => op.name == "linalg.generic").map (·.operands)
    | _ => none

#guard operands (region (base 64) environment facts (body 9) signature) == some [0, 1]
#guard operands (region (base 64) environment facts (body 0) signature) == some [3, 1]
#guard operands (region (base 64) environment [] (body 9) signature) == some [3, 1]
#guard (region (base 64) environment facts (body 3) signature).isNone

-- Even a fresh region argument does not preserve a fact across an actual scalar overwrite.
#guard operands (region (base 64) environment facts
  (body 9 [mkOp "arith.constant" [v 0 i32] [] [("value", .int 11 i32)]]) signature) == some [3, 1]

-- Replacing the tensor binding learns its new payload; the old captured value is never reused.
#guard operands (region (base 64) environment facts
  (body 9 [SplatC1.splat 3 9 SplatC1.shaped]) signature) == some [9, 1]

-- A fact with an unbound dependency is not a substitute for an actual scalar binding.
#guard operands (region (base 64) environment [⟨3, 8, [2], i32⟩] (body 9) signature) == some [3, 1]

-- Region signatures and source scopes are independently recomputed before normalization.
#guard (region (base 64) environment facts (body 9) ⟨[.scalar (.float .f32)], [.tensor [2] i32], 0⟩).isNone
#guard (region (base 64) environment facts (body 9) ⟨[.scalar i32], [.scalar i32], 0⟩).isNone
#guard (region (base 64) environment facts (.mk []) signature).isNone

end MerlinVerify.Tests.SplatNested
