/- A reusable initialized-execution prefix walker for any proved contextual rulebook.
Every actual SSA write shadows the previous fact; unsupported operations refuse the prefix. -/
import MerlinVerify.Check.DefinedContext
import MerlinVerify.Check.Step

namespace MerlinVerify.Check.Definedness.Contextual

open MerlinVerify

variable {Info : Type} {M : Refined.Model Info} {sem : OpSem} {ib : Nat} {allowed : Conditions}

def inferPrefix (R : Book M sem ib allowed) (Γ : Refined.FEnv Info) :
    List Op → Option (Refined.FEnv Info)
  | [] => some Γ
  | op :: rest => do
    if op.isTerminator = false ∧ op.successors = [] then
      let inputs ← op.operands.mapM (fun id => Γ.lookup id)
      let outputs ← R.rule Γ op inputs
      if outputs.length == op.results.length then
        inferPrefix R ((op.results.map (·.id)).zip outputs ++ Γ) rest
      else none
    else none

theorem inferPrefix_sound (R : Book M sem ib allowed) : ∀ Γ ops final ctx call env,
    inferPrefix R Γ ops = some final → Refined.EnvHolds M ib Γ env →
    ctx.indexBits = ib → allowed ctx call →
    ∃ actual, execOps sem ctx call env ops = .ok actual ∧ Refined.EnvHolds M ib final actual
  | Γ, [], final, _ctx, _call, env, checked, valid, _width, _permitted => by
    cases checked
    exact ⟨env, rfl, valid⟩
  | Γ, op :: rest, final, ctx, call, env, checked, valid, width, permitted => by
    simp only [inferPrefix] at checked
    split at checked
    · next header =>
      cases fetched : op.operands.mapM (fun id => Γ.lookup id) with
      | none => simp [fetched] at checked
      | some inputs =>
        simp only [fetched, bind, Option.bind] at checked
        cases selected : R.rule Γ op inputs with
        | none => simp [selected] at checked
        | some outputs =>
          simp only [selected] at checked
          split at checked
          · next arity =>
            obtain ⟨args, bound, holds⟩ := valid.getAll op.operands inputs fetched
            obtain ⟨values, executed, results⟩ := R.sound Γ op inputs outputs selected ctx call env args
              width permitted valid holds
            have count : op.results.length = values.length :=
              (eq_of_beq arity).symm.trans (List.Forall₂.length_eq results).symm
            obtain ⟨actual, remaining, finalHolds⟩ := inferPrefix_sound R _ rest final ctx call _ checked
              (valid.extend _ values outputs results) width permitted
            refine ⟨actual, ?_, finalHolds⟩
            rw [execOps_cons_step sem ctx call env op rest header.1, stepOp, bound, Outcome.bind_ok,
              executed, Outcome.bind_ok, bindAll_ok count, Outcome.bind_ok]
            exact remaining
          · cases checked
    · cases checked

#print axioms inferPrefix_sound

end MerlinVerify.Check.Definedness.Contextual
