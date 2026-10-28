/- Closed literal initializer recognition. Its post-execution binding is proved from syntax,
including the fill body and selected destination, for every static rank. -/
import MerlinVerify.Rules.W8A8Fill

namespace MerlinVerify.Check.PolicyW8A8Fill

open MerlinVerify Rules.W8A8Fill

def Correct (shape : List Nat) (elt : Ty) (literal : Attr) (actual : List Op)
    (destination : Nat) : Prop :=
  ∀ ctx call env value, scalarOfAttr ctx.indexBits literal = some value →
    evalOps Sem.Ops.opSem ctx call env (actual ++ [returned destination]) =
      .ok [.tensor (Tensor.full shape elt (.some value))]

def check (shape : List Nat) (elt : Ty) (literal : Attr) (actual : List Op)
    (destination : Nat) : Bool :=
  actual.all (fun op => !op.isTerminator) &&
    alphaOps [] (operations shape elt literal ++ [returned 2]) (actual ++ [returned destination])

theorem check_sound {shape : List Nat} {elt : Ty} {literal : Attr} {actual : List Op}
    {destination : Nat} (h : check shape elt literal actual destination = true) :
    Correct shape elt literal actual destination := by
  have hh := h
  simp only [check, Bool.and_eq_true] at hh
  intro ctx call env value decoded
  rw [← alphaOps_sound call call [] (operations shape elt literal ++ [returned 2])
    (actual ++ [returned destination]) ctx env env hh.2
    (by intro a b hab; simp [Rho.find, List.lookup] at hab) (fun _ _ => rfl)]
  exact operations_sound ctx call env shape elt literal value decoded

theorem check_binding_sound {shape : List Nat} {elt : Ty} {literal : Attr} {actual : List Op}
    {destination : Nat} (h : check shape elt literal actual destination = true)
    (ctx : Ctx) (call : CallSem) (env env' : Env) (value : Scalar)
    (decoded : scalarOfAttr ctx.indexBits literal = some value)
    (executed : execOps Sem.Ops.opSem ctx call env actual = .ok env') :
    env'.get? destination = some (.tensor (Tensor.full shape elt (.some value))) := by
  have hh := h
  simp only [check, Bool.and_eq_true, List.all_eq_true, Bool.not_eq_true'] at hh
  have result := check_sound h ctx call env value decoded
  rw [evalOps_append _ _ _ _ _ _ hh.1, executed, Outcome.bind_ok] at result
  have hr : evalOps Sem.Ops.opSem ctx call env' [returned destination] =
      env'.getAll [destination] := by
    simp [evalOps, returned, Op.isTerminator, Op.name, terminatorNames]
  rw [hr] at result
  cases found : env'.get? destination with
  | none => simp [Env.getAll, Outcome.mapM, Outcome.ofOption, found] at result
  | some output =>
    have hv : output = .tensor (Tensor.full shape elt (.some value)) := by
      simpa [Env.getAll, Outcome.mapM, Outcome.ofOption, found] using result
    simpa [found] using congrArg some hv

#print axioms check_sound
#print axioms check_binding_sound

end MerlinVerify.Check.PolicyW8A8Fill
