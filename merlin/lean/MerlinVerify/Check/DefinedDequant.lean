/- Defined execution of the locked dequantization source contract on initialized tensor inputs.
This strengthens source support without adding float laws or changing the source semantics. -/
import MerlinVerify.Check.DefinedLoops
import MerlinVerify.Check.RulesDequant

namespace MerlinVerify.Check.Definedness

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open MerlinVerify.Rules.DequantSource

theorem dequant_bits_defined {w : Nat} {v : Val}
    (ht : v.conforms 0 (.int w) = true) (hd : v ≠ .poison) :
    ∃ x, Dequant.bits w v = some x := by
  have hr := bits_roundtrip ht
  cases hb : Dequant.bits w v with
  | none => simp only [hb, Rules.DequantScalar.intValue] at hr; exact (hd hr.symm).elim
  | some x => exact ⟨x, rfl⟩

theorem dequant_float_defined {v : Val}
    (ht : v.conforms 0 (.float .f32) = true) (hd : v ≠ .poison) :
    ∃ x, Dequant.float32 v = some x := by
  have hr := float_roundtrip ht
  cases hb : Dequant.float32 v with
  | none => simp only [hb, Rules.DequantScalar.floatValue] at hr; exact (hd hr.symm).elim
  | some x => exact ⟨x, rfl⟩

/-- Every observable result is nonpoison on typed, initialized inputs and a valid parameter axis.
This quantifies over every float model; it establishes definedness, not IEEE accuracy. -/
theorem dequant_result_defined {c : Dequant.Config} {q s z : Tensor} (M : FloatModel)
    (ha : c.channel = false ∨ c.axis < c.shape.length)
    (hv : Dequant.validInputs c q s z = true)
    (hq : (RVal.tensor q).Defined) (hs : (RVal.tensor s).Defined)
    (hz : (RVal.tensor z).Defined) : (RVal.tensor (Dequant.result c M q s z)).Defined := by
  simp only [Dequant.validInputs, Bool.and_eq_true, decide_eq_true_eq] at hv
  obtain ⟨⟨⟨⟨hqshape, hsshape, hzshape⟩, hqt⟩, hst⟩, hzt⟩ := hv
  intro p hp
  change Idx.InBounds c.shape p at hp
  have hi : Idx.InBounds (Dequant.paramShape c) (Spec.Dequant.parameterIndex c.channel c.axis p) :=
    parameterIndex_bounds ha hp
  have hqp : Idx.InBounds q.shape p := hqshape ▸ hp
  have hsp : Idx.InBounds s.shape (Spec.Dequant.parameterIndex c.channel c.axis p) := hsshape ▸ hi
  have hzp : Idx.InBounds z.shape (Spec.Dequant.parameterIndex c.channel c.axis p) := hzshape ▸ hi
  obtain ⟨a, hqa⟩ := dequant_bits_defined (tensorTyped_get hqt hqp) (hq p hqp)
  obtain ⟨b, hzb⟩ := dequant_bits_defined (tensorTyped_get hzt hzp) (hz _ hzp)
  obtain ⟨scale, hsc⟩ := dequant_float_defined (tensorTyped_get hst hsp) (hs _ hsp)
  simp [Dequant.result, Spec.Dequant.tensor, Tensor.canon, hp,
    Spec.Dequant.scalarValue, hqa, hzb, hsc]

/-- Per-tensor at any zero-point width: initialized inputs give a nonpoison result. -/
theorem resultPerTensorAt_defined {w : Nat} {c : Dequant.Config} {q s z : Tensor} (M : FloatModel)
    (hv : Dequant.validPerTensorAt w c q s z = true)
    (hq : (RVal.tensor q).Defined) (hs : (RVal.tensor s).Defined)
    (hz : (RVal.tensor z).Defined) : (RVal.tensor (Dequant.resultPerTensorAt w c M q s z)).Defined := by
  simp only [Dequant.validPerTensorAt, Bool.and_eq_true, decide_eq_true_eq] at hv
  obtain ⟨⟨⟨⟨_, hqshape, hsshape, hzshape⟩, hqt⟩, hst⟩, hzt⟩ := hv
  intro p hp
  change Idx.InBounds c.shape p at hp
  have hempty : Idx.InBounds [] ([] : Idx) := (Idx.inBounds_nil_iff []).2 rfl
  have hqp : Idx.InBounds q.shape p := hqshape ▸ hp
  have hsp : Idx.InBounds s.shape [] := hsshape ▸ hempty
  have hzp : Idx.InBounds z.shape [] := hzshape ▸ hempty
  obtain ⟨a, hqa⟩ := dequant_bits_defined (tensorTyped_get hqt hqp) (hq p hqp)
  obtain ⟨b, hzb⟩ := dequant_bits_defined (tensorTyped_get hzt hzp) (hz _ hzp)
  obtain ⟨scale, hsc⟩ := dequant_float_defined (tensorTyped_get hst hsp) (hs _ hsp)
  simp [Dequant.resultPerTensorAt, Spec.Dequant.perTensorAtTensor, Tensor.canon, hp,
    Spec.Dequant.perTensorValueAt, hqa, hzb, hsc]

/-- The source contract's exact tensor domain, with initialization required for all arithmetic
inputs. The parameter-axis check is explicit rather than inferred from a successful rewrite. -/
def dequantDefinedRule (op : Op) (fs : List Fact) : Option (List Fact) := do
  let c ← Dequant.config? op
  if dialectOf op.name = "quant_ext" ∧
      (c.channel = false ∨ c.axis < c.shape.length) ∧
      (fs = clean [.tensor c.shape (.int 8), .tensor (Dequant.paramShape c) (.float .f32),
        .tensor (Dequant.paramShape c) (.int 32)] ∨
       (c.channel = false ∧ fs = clean [.tensor c.shape (.int 8), .tensor [] (.float .f32),
        .tensor [] (.int 64)])) then
    some (clean [.tensor c.shape (.float .f32)])
  else none

theorem dequantDefinedRule_sound (ib : Nat) : RuleSound ib dequantDefinedRule := by
  intro op fs rs hm h ctx call args _ hargs
  unfold dequantDefinedRule at h
  cases hc : Dequant.config? op with
  | none => simp [hc] at h
  | some c =>
    simp only [hc, bind, Option.bind] at h
    split at h
    · next hchecks =>
      obtain ⟨hd, haxis, hfs⟩ := hchecks
      rcases hfs with rfl | ⟨hch, rfl⟩
      swap
      · cases h
        obtain _ | ⟨hq, hrest⟩ := hargs
        obtain _ | ⟨hs, hrest⟩ := hrest
        obtain _ | ⟨hz, hnil⟩ := hrest
        cases hnil
        rename_i qv sv zv
        cases qv with
        | scalar _ => cases hq.1
        | tensor q =>
          cases sv with
          | scalar _ => cases hs.1
          | tensor s =>
            cases zv with
            | scalar _ => cases hz.1
            | tensor z =>
              have hv : Dequant.validPerTensorAt 64 c q s z = true := by
                simp only [Dequant.validPerTensorAt, hch, Bool.not_false, Bool.true_and,
                  Bool.and_eq_true, decide_eq_true_eq]
                exact ⟨⟨⟨⟨hq.1.1, hs.1.1, hz.1.1⟩, dequant_tensorTyped (Or.inl rfl) hq.1⟩,
                  dequant_tensorTyped (Or.inr (Or.inr rfl)) hs.1⟩, dequant_tensorTyped_i64 hz.1⟩
              refine ⟨[.tensor (Dequant.resultPerTensorAt 64 c ctx.M q s z)], ?_,
                List.Forall₂.cons ⟨Rules.DequantPerTensor.resultPerTensorAt_typed ib 64 c ctx.M q s z,
                  fun _ => resultPerTensorAt_defined ctx.M hv (hq.2 rfl) (hs.2 rfl) (hz.2 rfl)⟩ .nil⟩
              rw [Rules.DequantSite.source_sem ctx call op c _ hm hd hc]
              simp only [Dequant.run, validInputs_of_i64 hz.1.2.1, hv, Bool.false_eq_true, ↓reduceIte]
      cases h
      obtain _ | ⟨hq, hrest⟩ := hargs
      obtain _ | ⟨hs, hrest⟩ := hrest
      obtain _ | ⟨hz, hnil⟩ := hrest
      cases hnil
      rename_i qv sv zv
      cases qv with
      | scalar _ => cases hq.1
      | tensor q =>
        cases sv with
        | scalar _ => cases hs.1
        | tensor s =>
          cases zv with
          | scalar _ => cases hz.1
          | tensor z =>
            have hv : Dequant.validInputs c q s z = true := by
              simp only [Dequant.validInputs, Bool.and_eq_true, decide_eq_true_eq]
              exact ⟨⟨⟨⟨hq.1.1, hs.1.1, hz.1.1⟩,
                dequant_tensorTyped (Or.inl rfl) hq.1⟩,
                dequant_tensorTyped (Or.inr (Or.inr rfl)) hs.1⟩,
                dequant_tensorTyped (Or.inr (Or.inl rfl)) hz.1⟩
            refine ⟨[.tensor (Dequant.result c ctx.M q s z)], ?_,
              List.Forall₂.cons ⟨dequant_result_typed ib c ctx.M q s z,
                fun _ => dequant_result_defined ctx.M haxis hv (hq.2 rfl) (hs.2 rfl) (hz.2 rfl)⟩ .nil⟩
            rw [Rules.DequantSite.source_sem ctx call op c _ hm hd hc]
            simp only [Dequant.run, hv, ↓reduceIte]
    · cases h

def dequantBook (ib : Nat) : Book opSem ib where
  rule _ op fs := if unmodelledAttr? op.blind = none ∧ op.regions = [] then
    dequantDefinedRule op fs else none
  sound Γ op fs rs h ctx call env args hib _ ha := by
    split at h
    · next hchecks =>
      obtain ⟨hm, hregions⟩ := hchecks
      simpa only [hregions, evalRegions] using
        dequantDefinedRule_sound ib op fs rs hm h ctx call args hib ha
    · cases h

end MerlinVerify.Check.Definedness
