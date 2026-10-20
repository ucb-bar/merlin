/- Closed initializer recognition with an actual post-execution SSA binding theorem. -/
import MerlinVerify.Rules.W8A8Zero

namespace MerlinVerify.Check.PolicyW8A8Zero

open MerlinVerify Rules.W8A8Zero

def Correct (m n : Nat) (actual : List Op) (destination : Nat) : Prop :=
  ∀ ctx call env, evalOps Sem.Ops.opSem ctx call env (actual ++ [returned destination]) =
    .ok [.tensor (Tensor.full [m, n] (.int 32) zero)]

def check (m n : Nat) (actual : List Op) (destination : Nat) : Bool :=
  actual.all (fun op => !op.isTerminator) &&
    alphaOps [] (operations m n ++ [returned 2]) (actual ++ [returned destination])

/-- Acceptance checks every initializer operation and its selected destination, including fresh
SSA names. Closed matching leaves the surrounding environment arbitrary. -/
theorem check_sound {m n : Nat} {actual : List Op} {destination : Nat}
    (h : check m n actual destination = true) : Correct m n actual destination := by
  have hh := h
  simp only [check, Bool.and_eq_true] at hh
  intro ctx call env
  rw [← alphaOps_sound call call [] (operations m n ++ [returned 2])
    (actual ++ [returned destination]) ctx env env hh.2
    (by intro a b hab; simp [Rho.find, List.lookup] at hab) (fun _ _ => rfl)]
  exact operations_sound ctx call env m n

/-- The checked destination really is a zero tensor after executing the actual initializer.
Neither its name nor a certificate-supplied tensor value can establish this fact. -/
theorem check_binding_sound {m n : Nat} {actual : List Op} {destination : Nat}
    (h : check m n actual destination = true) (ctx : Ctx) (call : CallSem) (env env' : Env)
    (executed : execOps Sem.Ops.opSem ctx call env actual = .ok env') :
    env'.get? destination = some (.tensor (Tensor.full [m, n] (.int 32) zero)) := by
  have hh := h
  simp only [check, Bool.and_eq_true, List.all_eq_true, Bool.not_eq_true'] at hh
  have result := check_sound h ctx call env
  rw [evalOps_append _ _ _ _ _ _ hh.1, executed, Outcome.bind_ok] at result
  have hr : evalOps Sem.Ops.opSem ctx call env' [returned destination] =
      env'.getAll [destination] := by
    simp [evalOps, returned, Op.isTerminator, Op.name, terminatorNames]
  rw [hr] at result
  cases found : env'.get? destination with
  | none => simp [Env.getAll, Outcome.mapM, Outcome.ofOption, found] at result
  | some value =>
    have hv : value = .tensor (Tensor.full [m, n] (.int 32) zero) := by
      simpa [Env.getAll, Outcome.mapM, Outcome.ofOption, found] using result
    simpa [found] using congrArg some hv

#print axioms check_sound
#print axioms check_binding_sound

end MerlinVerify.Check.PolicyW8A8Zero
