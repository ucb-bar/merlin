/- Conditional definedness of the existing, proposed signed-i8 quantizer.

No IEEE laws are added. The domain names the precise final conversion whose
failure produces poison. An executable countermodel explains why initialized,
well-typed inputs alone cannot establish definedness for every FloatModel.
This is an additive theorem, not source-specification or numerical approval. -/
import MerlinVerify.Rules.QuantizeSource
import MerlinVerify.Rules.QuantizeSite
import MerlinVerify.Check.DefinedDequant

namespace MerlinVerify.Rules.QuantizeDefinedness

open MerlinVerify MerlinVerify.Sem.Ops

/-- Exact scalar definedness, without appealing to clamp names or float laws. -/
theorem scalar_defined_iff (w : Nat) (M : FloatModel) (x scale : FBits .f32)
    (zero : BitVec w) (lower upper : FBits .f32) :
    Spec.Quantize.scalarValue w M (some x) (some scale) (some zero) lower upper ≠ .poison ↔
      ∃ q, Spec.Quantize.withInverse w M x (Spec.Quantize.inverse M scale) zero lower upper = some q := by
  unfold Spec.Quantize.scalarValue
  cases h : Spec.Quantize.withInverse w M x (Spec.Quantize.inverse M scale) zero lower upper <;> simp [h]

/-- The actual multiply, round, shift, lower-clamp, upper-clamp sequence before
the signed conversion. Bounds and arithmetic keys are the existing source's. -/
def conversionInput (w : Nat) (M : FloatModel) (x scale : FBits .f32)
    (zero : BitVec w) : FBits .f32 :=
  let scaled := M.op (Spec.Dequant.defaultKey "arith.mulf") .f32
    [x, Spec.Quantize.inverse M scale]
  let rounded := M.op (Spec.Dequant.defaultKey "math.roundeven") .f32 [scaled]
  let shifted := M.op (Spec.Dequant.defaultKey "arith.addf") .f32
    [rounded, M.ofInt (Spec.Dequant.defaultKey "arith.sitofp") w .f32 zero]
  let low := M.op (Spec.Dequant.defaultKey "arith.maximumf") .f32 [shifted, 0xc3000000#32]
  M.op (Spec.Dequant.defaultKey "arith.minimumf") .f32 [low, 0x42fe0000#32]

/-- Every observed activation and both scalar parameters have actual values,
and the exact final signed-i8 conversion succeeds. This condition is model
dependent; positivity of scale alone supplies no laws for abstract operations. -/
def Domain (w : Nat) (M : FloatModel) (shape : List Nat)
    (input scale zero : Tensor) : Prop :=
  ∃ s z, Dequant.float32 (scale.get []) = some s ∧
    Dequant.bits w (zero.get []) = some z ∧
    ∀ p, Idx.InBounds shape p → ∃ x q,
      Dequant.float32 (input.get p) = some x ∧
      M.toInt (Spec.Dequant.defaultKey "arith.fptosi") .f32 8
        (conversionInput w M x s z) = some q

/-- A concrete, independently dischargeable input domain gives a nonpoison
quantized tensor, including rank-zero and empty tensors. -/
theorem result_defined {w : Nat} {c : Quantize.Config} {M : FloatModel}
    {input scale zero : Tensor} (h : Domain w M c.shape input scale zero) :
    (RVal.tensor (Quantize.result w c M input scale zero)).Defined := by
  obtain ⟨s, z, hs, hz, data⟩ := h
  intro p hp
  change Idx.InBounds c.shape p at hp
  obtain ⟨x, q, hx, hq⟩ := data p hp
  have converted : Spec.Quantize.withInverse w M x (Spec.Quantize.inverse M s) z
      0xc3000000#32 0x42fe0000#32 = some q := hq
  simp [Quantize.result, Spec.Quantize.tensor, Tensor.canon, hp,
    Spec.Quantize.scalarValue, hx, hs, hz, converted]

/-- Source execution succeeds and has defined results on the explicit domain.
Both admitted zero-point widths use the actual dispatch order of `run`. -/
theorem run_defined {w : Nat} {c : Quantize.Config} {M : FloatModel}
    {input scale zero : Tensor} (width : w = 32 ∨ w = 64)
    (valid : Quantize.validInputs w c input scale zero = true)
    (h : Domain w M c.shape input scale zero) :
    ∃ results, Quantize.run c M [.tensor input, .tensor scale, .tensor zero] = .ok results ∧
      ResultsDefined results := by
  rcases width with rfl | rfl
  · refine ⟨[.tensor (Quantize.result 32 c M input scale zero)], ?_, ?_⟩
    · simp [Quantize.run, valid]
    · intro v hv
      simp only [List.mem_singleton] at hv
      subst v
      exact result_defined h
  · by_cases h32 : Quantize.validInputs 32 c input scale zero = true
    · have htype32 : zero.elt = .int 32 := by
        simp only [Quantize.validInputs, Bool.and_eq_true] at h32
        have typed := h32.2
        simp only [Dequant.tensorTyped, Bool.and_eq_true, beq_iff_eq] at typed
        exact typed.1
      have htype64 : zero.elt = .int 64 := by
        simp only [Quantize.validInputs, Bool.and_eq_true] at valid
        have typed := valid.2
        simp only [Dequant.tensorTyped, Bool.and_eq_true, beq_iff_eq] at typed
        exact typed.1
      simp [htype32] at htype64
    · refine ⟨[.tensor (Quantize.result 64 c M input scale zero)], ?_, ?_⟩
      · simp [Quantize.run, h32, valid]
      · intro v hv
        simp only [List.mem_singleton] at hv
        subst v
        exact result_defined h

/-- Transfer the explicit domain to the actual six-operation lowering boundary.
All target temporaries, captured clamps, arithmetic and SSA correspondences are
still checked by the existing site theorem. This is not a whole-model theorem. -/
theorem target_result_defined {ρ : Check.Rho} {s : QuantizeSite.Site}
    (checked : QuantizeSite.Checks ρ s) (ctx : Ctx) (call : CallSem) (envS envT : Env)
    (corresponds : Check.Corr ρ envS envT) (w : Nat) (data scale zero : Tensor)
    (fetch : envS.getAll s.src.operands = .ok [.tensor data, .tensor scale, .tensor zero])
    (valid : Quantize.validInputs w s.c data scale zero = true)
    (domain : Domain w ctx.M s.c.shape data scale zero) :
    ∃ envT', execOps opSem ctx call envT (QuantizeSite.targets s) = .ok envT' ∧
      ∃ result, envT'.get? s.rg.id = some result ∧ result.Defined := by
  obtain ⟨envT', executed, matched⟩ :=
    QuantizeSite.target_sound checked ctx call envS envT corresponds w data scale zero fetch valid
  refine ⟨envT', executed, .tensor (Quantize.result w s.c ctx.M data scale zero), ?_,
    result_defined domain⟩
  have pair := matched s.rs.id s.rg.id (by simp [Check.Rho.find, List.lookup])
  simpa [Env.get?, List.lookup] using pair.symm

/-- A legal interpretation of the frozen model: all integer conversions fail.
It uses no axioms about actual hardware or framework arithmetic. -/
def failingConversionModel : FloatModel where
  op := fun _ k _ => 0#k.width
  cmp := fun _ _ _ _ => false
  convert := fun _ _ dst _ => 0#dst.width
  toInt := fun _ _ _ _ => none
  ofInt := fun _ _ k _ => 0#k.width

/-- Initialized scalar arguments do not suffice for unconditional quantizer
definedness in the existing abstract model. Preserve this negative witness. -/
theorem initialized_counterexample :
    let input := Tensor.full [] (.float .f32) (.some (.flt .f32 0#32))
    let scale := Tensor.full [] (.float .f32) (.some (.flt .f32 0x3f800000#32))
    let zero := Tensor.full [] (.int 32) (.some (.bits 32 0#32))
    Quantize.validInputs 32 ⟨[]⟩ input scale zero = true ∧
    (RVal.tensor input).Defined ∧ (RVal.tensor scale).Defined ∧ (RVal.tensor zero).Defined ∧
    ¬ (RVal.tensor (Quantize.result 32 ⟨[]⟩ failingConversionModel input scale zero)).Defined := by
  refine ⟨by cbv, ?_, ?_, ?_, ?_⟩
  · intro p hp; change Idx.InBounds [] p at hp; simp [Tensor.full, Tensor.canon, hp]
  · intro p hp; change Idx.InBounds [] p at hp; simp [Tensor.full, Tensor.canon, hp]
  · intro p hp; change Idx.InBounds [] p at hp; simp [Tensor.full, Tensor.canon, hp]
  · intro h
    have hp : Idx.InBounds [] ([] : Idx) := (Idx.inBounds_nil_iff []).2 rfl
    have := h [] hp
    simp [Quantize.result, Spec.Quantize.tensor, Tensor.full, Tensor.canon, hp,
      Spec.Quantize.scalarValue, Dequant.float32, Dequant.bits,
      Spec.Quantize.withInverse, failingConversionModel] at this

#print axioms result_defined
#print axioms scalar_defined_iff
#print axioms run_defined
#print axioms target_result_defined
#print axioms initialized_counterexample

end MerlinVerify.Rules.QuantizeDefinedness
