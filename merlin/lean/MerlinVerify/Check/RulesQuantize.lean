/- Typed source support for the full-range signed-byte per-tensor subsection. -/
import MerlinVerify.Rules.QuantizeSource

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops

def quantizeRule : OpRule := fun _ op argTys resTys =>
  match Quantize.config? op with
  | some c =>
    if (argTys = [.tensor c.shape (.float .f32), .tensor [] (.float .f32), .tensor [] (.int 32)] ∨
        argTys = [.tensor c.shape (.float .f32), .tensor [] (.float .f32), .tensor [] (.int 64)]) ∧
        resTys = [.tensor c.shape (.int 8)] then some [] else none
  | none => none

theorem quantizeRule_sound (ib : Nat) : RuleSound ib quantizeRule := by
  intro d op argTys resTys sigs modelled accepted
  simp only [quantizeRule] at accepted
  split at accepted
  · next c configured =>
    split at accepted
    · next typed =>
      obtain ⟨arguments, rfl⟩ := typed
      rcases arguments with rfl | rfl
      all_goals
        cases accepted
        intro ctx call args regions _ _ ha hr
        cases hr
        cases ha with
        | cons hx hrest =>
          cases hrest with
          | cons hs hrest =>
            cases hrest with
            | cons hz hnil =>
              cases hnil
              rename_i xv sv zv
              cases xv with
              | scalar v => exact (hx : False).elim
              | tensor x =>
                cases sv with
                | scalar v => exact (hs : False).elim
                | tensor s =>
                  cases zv with
                  | scalar v => exact (hz : False).elim
                  | tensor z =>
                    rw [Rules.QuantizeSource.source_sem ctx call _ modelled configured]
                    first
                    | have valid := Rules.QuantizeSource.validInputs_of_typing hx hs hz (Or.inl rfl)
                      simp only [Quantize.run, valid, ↓reduceIte]
                      exact Safe.of_ok (.cons (Rules.QuantizeSource.result_typed ib 32 c ctx.M x s z) .nil)
                    | have valid := Rules.QuantizeSource.validInputs_of_typing hx hs hz (Or.inr rfl)
                      have wrongWidth : Quantize.validInputs 32 c x s z = false := by
                        simp [Quantize.validInputs, Dequant.tensorTyped, hz.2.1]
                      simp only [Quantize.run, wrongWidth, valid, Bool.false_eq_true, ↓reduceIte]
                      exact Safe.of_ok (.cons (Rules.QuantizeSource.result_typed ib 64 c ctx.M x s z) .nil)
    · cases accepted
  · cases accepted

end MerlinVerify.Check
