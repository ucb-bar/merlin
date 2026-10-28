/- Projection keeps existing definedness obligations, while per-binding evidence is shadowed
alongside its type and initialization status. No external certificate populates entry facts. -/
import MerlinVerify.Check.DefinedRefinements
import MerlinVerify.Check.DefinedRules
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.DefinedRefinements

open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness.Refined
open MerlinVerify.Tests.Semantics

def trivialEvidence : Model Unit := ⟨fun _ _ => True, (), fun _ => trivial⟩
def projected : Book trivialEvidence Sem.Ops.opSem 64 := Book.lift trivialEvidence (Definedness.leafBook 64)

def scalar : Module := { ops := [func "f" []
  [mkOp "arith.constant" [v 0 i32] [] [("value", .int 7 i32)], ret [0]]] }
#guard check projected scalar "f" [] [.scalar i32]
#guard !check projected scalar "f" [] [.scalar i8]

def dirty : Module := { ops := [func "f" [] [mkOp "tensor.empty" [v 0 (tensorTy [2] i32)] [], ret [0]]] }
#guard !check projected dirty "f" [] [.tensor [2] i32]

-- Evidence belongs to the current value, not to a formerly used SSA name.
def old : FEnv (Option Nat) := [(0, ⟨⟨.scalar .index, true⟩, some 0⟩)]
def shadowed : FEnv (Option Nat) := (0, ⟨⟨.scalar .index, true⟩, none⟩) :: old
#guard (old.lookup 0).map (·.info) == some (some 0)
#guard (shadowed.lookup 0).map (·.info) == some none
#guard (baseEnv shadowed).lookup 0 == some ⟨.scalar .index, true⟩

end MerlinVerify.Tests.DefinedRefinements
