import Lean.Elab.Tactic.Omega

namespace DiagnosticReprOmegaListNat

structure Payload where
  value : List Nat
  deriving Repr

def witness : Payload := ⟨[7, 11]⟩

theorem witness_value : witness.value = [7, 11] := rfl

end DiagnosticReprOmegaListNat
