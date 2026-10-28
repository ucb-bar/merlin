/- Source support/nonvacuity for the proposed signed-i8 dequantization contract: per-channel with
an i32 zero point, per-tensor with an i32 or i64 zero point. -/
import MerlinVerify.Rules.DequantSite

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops

/-- For the three supported scalar types, typing does not depend on the index width. -/
theorem dequant_conforms {ib : Nat} {ty : Ty} {v : Val}
    (ht : ty = .int 8 ∨ ty = .int 32 ∨ ty = .float .f32) (hv : Val.HasTy ib ty v) :
    v.conforms 0 ty = true := by
  rcases ht with rfl | rfl | rfl <;> cases v with
  | poison => rfl
  | some s => cases s <;> simpa [Val.HasTy, Val.conforms, Scalar.conforms] using hv

theorem dequant_conforms_i64 {ib : Nat} {v : Val} (hv : Val.HasTy ib (.int 64) v) :
    v.conforms 0 (.int 64) = true := by
  cases v with
  | poison => rfl
  | some s => cases s <;> simpa [Val.HasTy, Val.conforms, Scalar.conforms] using hv

theorem dequant_tensorTyped_i64 {ib : Nat} {t : Tensor} {shape : List Nat}
    (hv : RVal.HasTy ib (.tensor t) (.tensor shape (.int 64))) : Dequant.tensorTyped t (.int 64) = true := by
  obtain ⟨hs, he, hg⟩ := hv
  simp only [Dequant.tensorTyped, he, beq_self_eq_true, Bool.true_and, List.all_eq_true]
  intro p hp
  exact dequant_conforms_i64 (hg p (hs ▸ (mem_allPoints _ _).1 hp))

/-- A zero point of another integer type is never the i32 contract's. -/
theorem validInputs_of_i64 {c : Dequant.Config} {q s z : Tensor} (he : z.elt = .int 64) :
    Dequant.validInputs c q s z = false := by
  have : Dequant.tensorTyped z (.int 32) = false := by
    simp [Dequant.tensorTyped, he]
  simp [Dequant.validInputs, this]

theorem dequant_tensorTyped {ib : Nat} {t : Tensor} {shape : List Nat} {ty : Ty}
    (ht : ty = .int 8 ∨ ty = .int 32 ∨ ty = .float .f32)
    (hv : RVal.HasTy ib (.tensor t) (.tensor shape ty)) : Dequant.tensorTyped t ty = true := by
  obtain ⟨hs, he, hg⟩ := hv
  simp only [Dequant.tensorTyped, he, beq_self_eq_true, Bool.true_and, List.all_eq_true]
  intro p hp
  exact dequant_conforms ht (hg p (hs ▸ (mem_allPoints _ _).1 hp))

theorem dequant_result_typed (ib : Nat) (c : Dequant.Config) (M : FloatModel) (q s z : Tensor) :
    RVal.HasTy ib (.tensor (Dequant.result c M q s z)) (.tensor c.shape (.float .f32)) := by
  refine ⟨rfl, rfl, fun p hp => ?_⟩
  simp only [Dequant.result, Spec.Dequant.tensor, Tensor.canon, hp, ↓reduceIte]
  unfold Spec.Dequant.scalarValue
  split <;> simp [Val.HasTy, Scalar.conforms]

/-- This is a source typing rule, separate from the C1 transformation rule. -/
def dequantRule : OpRule := fun _ op argTys resTys =>
  match Dequant.config? op with
  | some c =>
    if dialectOf op.name = "quant_ext" ∧
        (argTys = [.tensor c.shape (.int 8), .tensor (Dequant.paramShape c) (.float .f32),
          .tensor (Dequant.paramShape c) (.int 32)] ∨
         (c.channel = false ∧ argTys = [.tensor c.shape (.int 8), .tensor [] (.float .f32),
          .tensor [] (.int 64)])) ∧
        resTys = [.tensor c.shape (.float .f32)] then some [] else none
  | none => none

theorem dequantRule_sound (ib : Nat) : RuleSound ib dequantRule := by
  intro d op argTys resTys sigs hm h
  simp only [dequantRule] at h
  split at h
  · next c hc =>
    split at h
    · next ht =>
      obtain ⟨hd, hargs, rfl⟩ := ht
      rcases hargs with rfl | ⟨hch, rfl⟩
      swap
      · cases h
        intro ctx call args regions _ _ ha hr
        cases hr
        cases ha with
        | cons hq hrest =>
          cases hrest with
          | cons hs hrest =>
            cases hrest with
            | cons hz hnil =>
              cases hnil
              rename_i qv sv zv
              cases qv with
              | scalar v => exact (hq : False).elim
              | tensor q =>
                cases sv with
                | scalar v => exact (hs : False).elim
                | tensor s =>
                  cases zv with
                  | scalar v => exact (hz : False).elim
                  | tensor z =>
                    have hv : Dequant.validPerTensorAt 64 c q s z = true := by
                      simp only [Dequant.validPerTensorAt, hch, Bool.not_false, Bool.true_and,
                        Bool.and_eq_true, decide_eq_true_eq]
                      exact ⟨⟨⟨⟨hq.1, hs.1, hz.1⟩, dequant_tensorTyped (Or.inl rfl) hq⟩,
                        dequant_tensorTyped (Or.inr (Or.inr rfl)) hs⟩, dequant_tensorTyped_i64 hz⟩
                    rw [Rules.DequantSite.source_sem ctx call op c _ hm hd hc]
                    simp only [Dequant.run, validInputs_of_i64 hz.2.1, hv, Bool.false_eq_true,
                      ↓reduceIte]
                    exact Safe.of_ok (.cons
                      (Rules.DequantPerTensor.resultPerTensorAt_typed ib 64 c ctx.M q s z) .nil)
      cases h
      intro ctx call args regions _ _ ha hr
      cases hr
      cases ha with
      | cons hq hrest =>
        cases hrest with
        | cons hs hrest =>
          cases hrest with
          | cons hz hnil =>
            cases hnil
            rename_i qv sv zv
            cases qv with
            | scalar v => exact (hq : False).elim
            | tensor q =>
              cases sv with
              | scalar v => exact (hs : False).elim
              | tensor s =>
                cases zv with
                | scalar v => exact (hz : False).elim
                | tensor z =>
                  have hv : Dequant.validInputs c q s z = true := by
                    simp only [Dequant.validInputs, Bool.and_eq_true, decide_eq_true_eq]
                    exact ⟨⟨⟨⟨hq.1, hs.1, hz.1⟩,
                      dequant_tensorTyped (Or.inl rfl) hq⟩,
                      dequant_tensorTyped (Or.inr (Or.inr rfl)) hs⟩,
                      dequant_tensorTyped (Or.inr (Or.inl rfl)) hz⟩
                  rw [Rules.DequantSite.source_sem ctx call op c _ hm hd hc]
                  simp only [Dequant.run, hv, ↓reduceIte]
                  exact Safe.of_ok (.cons (dequant_result_typed ib c ctx.M q s z) .nil)
    · cases h
  · cases h

end MerlinVerify.Check
