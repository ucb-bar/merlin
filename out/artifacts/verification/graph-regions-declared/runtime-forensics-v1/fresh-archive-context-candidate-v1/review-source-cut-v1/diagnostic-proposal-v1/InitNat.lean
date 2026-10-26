import Init

namespace DiagnosticReprInitNat

structure Payload where
  value : Nat
  deriving Repr

def witness : Payload := ⟨7⟩

theorem witness_value : witness.value = 7 := rfl

end DiagnosticReprInitNat
