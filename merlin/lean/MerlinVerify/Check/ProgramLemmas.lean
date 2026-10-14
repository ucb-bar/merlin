/-
Lemmas about the frozen evaluator (`Sem/Program.lean`), generic over `OpSem`.

* `execOps` runs a straight-line segment of non-terminator ops and returns the environment;
  `evalOps_append` splits block evaluation at any terminator-free prefix.
-/
import MerlinVerify.Sem.Program

namespace MerlinVerify

/-- Run non-terminator ops in order, returning the final environment. -/
def execOps (sem : OpSem) (ctx : Ctx) (call : CallSem) : Env → List Op → Outcome Env
  | env, [] => .ok env
  | env, op :: rest =>
    if op.isTerminator then .stuck "terminator inside a straight-line segment" else
    (env.getAll op.operands).bind fun args =>
    (sem ctx call op args (evalRegions sem call env op.regions)).bind fun results =>
    (env.bindAll op.results results).bind fun env' =>
    execOps sem ctx call env' rest

theorem evalOps_cons_nonterm (sem : OpSem) (ctx : Ctx) (call : CallSem) (env : Env) (op : Op)
    (rest : List Op) (h : op.isTerminator = false) :
    evalOps sem ctx call env (op :: rest) =
      (env.getAll op.operands).bind fun args =>
      (sem ctx call op args (evalRegions sem call env op.regions)).bind fun results =>
      (env.bindAll op.results results).bind fun env' =>
      evalOps sem ctx call env' rest := by
  cases op with
  | mk name results operands attrs props regions succ =>
    rw [evalOps.eq_2]
    rw [ite_eq_right (by rw [h]; decide)]
    rfl

/-- Block evaluation splits at a terminator-free prefix. -/
theorem evalOps_append (sem : OpSem) (ctx : Ctx) (call : CallSem) :
    ∀ (env : Env) (pre rest : List Op), (∀ op ∈ pre, op.isTerminator = false) →
      evalOps sem ctx call env (pre ++ rest) =
        (execOps sem ctx call env pre).bind fun env' => evalOps sem ctx call env' rest
  | env, [], rest, _ => by simp [execOps, Outcome.bind]
  | env, op :: pre, rest, h => by
    have hop : op.isTerminator = false := h op (List.mem_cons_self ..)
    have hpre : ∀ o ∈ pre, o.isTerminator = false := fun o ho => h o (List.mem_cons_of_mem _ ho)
    rw [List.cons_append, evalOps_cons_nonterm sem ctx call env op _ hop]
    simp only [execOps, hop, Bool.false_eq_true, ↓reduceIte]
    cases (env.getAll op.operands) with
    | ub => rfl
    | stuck r => rfl
    | ok args =>
      simp only [Outcome.bind]
      cases (sem ctx call op args (evalRegions sem call env op.regions)) with
      | ub => rfl
      | stuck r => rfl
      | ok results =>
        simp only
        cases (env.bindAll op.results results) with
        | ub => rfl
        | stuck r => rfl
        | ok env' => exact evalOps_append sem ctx call env' pre rest hpre

end MerlinVerify
