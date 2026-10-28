/- Runtime typing for the independent per-tensor quantization source contract. -/
import MerlinVerify.Sem.Ops.Quantize
import MerlinVerify.Check.RulesDequant

namespace MerlinVerify.Rules.QuantizeSource

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

@[simp] theorem config_blind (op : Op) : Quantize.config? op.blind = Quantize.config? op := by
  cases op with
  | mk name results operands attrs props regions successors =>
    cases results with
    | nil => rfl
    | cons r rest => cases rest <;> rfl

theorem config_name {op : Op} {c : Quantize.Config} (h : Quantize.config? op = some c) :
    op.name = "quant_ext.quantize_per_tensor" := by
  by_cases hn : op.name = "quant_ext.quantize_per_tensor"
  · exact hn
  · simp [Quantize.config?, hn] at h

theorem source_sem {op : Op} {c : Quantize.Config} (ctx : Ctx) (call : CallSem) (args : List RVal)
    (modelled : unmodelledAttr? op.blind = none) (configured : Quantize.config? op = some c) :
    opSem ctx call op args [] = Quantize.run c ctx.M args := by
  have name := config_name configured
  have dialect : dialectOf "quant_ext.quantize_per_tensor" = "quant_ext" := by cbv
  rw [opSem_modelled modelled]
  simp [dispatch, dialect, name, Quantize.sem, config_blind, configured]

theorem result_typed (ib width : Nat) (c : Quantize.Config) (M : FloatModel)
    (input scale zero : Tensor) :
    RVal.HasTy ib (.tensor (Quantize.result width c M input scale zero)) (.tensor c.shape (.int 8)) := by
  refine ⟨rfl, rfl, fun point hp => ?_⟩
  simp only [Quantize.result, Spec.Quantize.tensor, Tensor.canon, hp, ↓reduceIte]
  unfold Spec.Quantize.scalarValue
  split
  · split <;> simp [Val.HasTy, Scalar.conforms]
  · simp [Val.HasTy]

theorem validInputs_of_typing {ib width : Nat} {c : Quantize.Config} {input scale zero : Tensor}
    (data : RVal.HasTy ib (.tensor input) (.tensor c.shape (.float .f32)))
    (scaling : RVal.HasTy ib (.tensor scale) (.tensor [] (.float .f32)))
    (offset : RVal.HasTy ib (.tensor zero) (.tensor [] (.int width)))
    (widthOK : width = 32 ∨ width = 64) :
    Quantize.validInputs width c input scale zero = true := by
  have hz : Dequant.tensorTyped zero (.int width) = true := by
    rcases widthOK with rfl | rfl
    · exact dequant_tensorTyped (Or.inr (Or.inl rfl)) offset
    · exact dequant_tensorTyped_i64 offset
  simp only [Quantize.validInputs, Bool.and_eq_true, decide_eq_true_eq]
  exact ⟨⟨⟨⟨data.1, scaling.1, offset.1⟩,
    dequant_tensorTyped (Or.inr (Or.inr rfl)) data⟩,
    dequant_tensorTyped (Or.inr (Or.inr rfl)) scaling⟩, hz⟩

#print axioms result_typed
#print axioms validInputs_of_typing

end MerlinVerify.Rules.QuantizeSource
