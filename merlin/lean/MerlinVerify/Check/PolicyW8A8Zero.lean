/- Matrix i32 specialization of the shared literal-initializer checker. -/
import MerlinVerify.Rules.W8A8Zero
import MerlinVerify.Check.PolicyW8A8Fill

namespace MerlinVerify.Check.PolicyW8A8Zero

open MerlinVerify Rules.W8A8Zero

def Correct (m n : Nat) (actual : List Op) (destination : Nat) : Prop :=
  ∀ ctx call env, evalOps Sem.Ops.opSem ctx call env (actual ++ [returned destination]) =
    .ok [.tensor (Tensor.full [m, n] (.int 32) zero)]

def check (m n : Nat) (actual : List Op) (destination : Nat) : Bool :=
  PolicyW8A8Fill.check [m, n] (.int 32) (.int 0 (.int 32)) actual destination

/-- Acceptance checks the closed actual initializer and its selected SSA destination. -/
theorem check_sound {m n : Nat} {actual : List Op} {destination : Nat}
    (h : check m n actual destination = true) : Correct m n actual destination := by
  intro ctx call env
  exact PolicyW8A8Fill.check_sound h ctx call env (.bits 32 0#32) rfl

/-- The zero value follows from executing the checked producer. -/
theorem check_binding_sound {m n : Nat} {actual : List Op} {destination : Nat}
    (h : check m n actual destination = true) (ctx : Ctx) (call : CallSem) (env env' : Env)
    (executed : execOps Sem.Ops.opSem ctx call env actual = .ok env') :
    env'.get? destination = some (.tensor (Tensor.full [m, n] (.int 32) zero)) :=
  PolicyW8A8Fill.check_binding_sound h ctx call env env' (.bits 32 0#32) rfl executed

#print axioms check_sound
#print axioms check_binding_sound

end MerlinVerify.Check.PolicyW8A8Zero
