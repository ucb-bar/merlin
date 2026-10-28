/- Actual runtime tensors connected to the independently specified quantizer result. -/
import MerlinVerify.Rules.QuantizeGeometry

namespace MerlinVerify.Rules.QuantizeLowering

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg
open QuantizeTensor QuantizeRuntime W8A8TensorOperands DequantScalar

def inputTypes (shape : List Nat) : List VTy :=
  [.tensor shape (.float .f32), .tensor [] (.float .f32), .tensor [] (.int 32), .tensor shape (.int 8)]

theorem inverse_typed (M : FloatModel) (scale : Tensor) :
    Dequant.tensorTyped (inverseTensor M scale) (.float .f32) = true := by
  have conforming (v : Option (FBits .f32)) : (floatValue v).conforms 0 (.float .f32) = true := by
    cases v <;> rfl
  have hn : Idx.InBounds [] ([] : Idx) := (Idx.inBounds_nil_iff []).2 rfl
  simp [Dequant.tensorTyped, inverseTensor, Tensor.canon, allPoints, hn, conforming]

theorem inverse_decode (M : FloatModel) (scale : Tensor) :
    Dequant.float32 ((inverseTensor M scale).get []) =
      (Dequant.float32 (scale.get [])).map (Spec.Quantize.inverse M) := by
  have hn : Idx.InBounds [] ([] : Idx) := (Idx.inBounds_nil_iff []).2 rfl
  simp only [inverseTensor, Tensor.canon, hn, ↓reduceIte]
  cases h : Dequant.float32 (scale.get []) <;> rfl

theorem inverse_from_valid {w : Nat} {c : Quantize.Config} {data scale zero : Tensor}
    {actual : Region} (valid : Quantize.validInputs w c data scale zero = true)
    (accepted : alphaRegion [] inverseRegion actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) :
    genericSem ctx [identityMap 0, identityMap 0] [] [.tensor scale]
      [.tensor (Tensor.full [] (.float .f32) .poison)]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (inverseTensor ctx.M scale)] := by
  simp only [Quantize.validInputs, Bool.and_eq_true, decide_eq_true_eq] at valid
  obtain ⟨⟨⟨⟨_, hs, _⟩, _⟩, typed⟩, _⟩ := valid
  have geometry : StructuredOk [identityMap 0, identityMap 0] 0
      [.tensor [] (.float .f32), .tensor [] (.float .f32)] [] :=
    structuredRanges?_spec (by decide)
  apply inverse_actual accepted ctx call env scale hs typed _ [] rfl
  apply QuantizeGeometry.retype geometry
  simp [input, Input.type, VTy.shape, hs]

theorem data_from_valid {w wireWidth lowerId upperId : Nat} {c : Quantize.Config}
    {data scale zero : Tensor} {actual : Region} {maps : List AffineMap} {iters : List IterKind}
    (valid : Quantize.validInputs w c data scale zero = true)
    (accepted : alphaRegion [(100, lowerId), (101, upperId)] (region wireWidth) actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env)
    (known : Corr [(100, lowerId), (101, upperId)] (captures 0xc3000000#32 0x42fe0000#32) env)
    (layout : maps = [identityMap c.shape.length, broadcast c.shape.length,
      broadcast c.shape.length, identityMap c.shape.length])
    (geometry : StructuredOk maps iters.length (inputTypes c.shape) c.shape) :
    genericSem ctx maps iters [.tensor data, .tensor (inverseTensor ctx.M scale), .tensor zero]
      [.tensor (Tensor.full c.shape (.int 8) .poison)]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Quantize.result w c ctx.M data scale zero)] := by
  have valid' := valid
  simp only [Quantize.validInputs, Bool.and_eq_true, decide_eq_true_eq] at valid'
  obtain ⟨⟨⟨⟨hd, hs, hz⟩, dt⟩, _⟩, zt⟩ := valid'
  have transported : StructuredOk maps iters.length
      [(input data).type, (input (inverseTensor ctx.M scale)).type, (input zero).type,
        .tensor data.shape (.int 8)] data.shape := by
    rw [hd]
    apply QuantizeGeometry.retype geometry
    simp [inputTypes, input, Input.type, VTy.shape, inverseTensor, Tensor.canon, hd, hz]
  have run := data_actual w accepted ctx call env 0xc3000000#32 0x42fe0000#32 known
    data (inverseTensor ctx.M scale) zero dt rfl (inverse_typed ctx.M scale) hz zt maps iters
    (by simpa [hd] using layout) transported
  simp only [hd, inverse_decode] at run
  have resultEq :
      Tensor.canon c.shape (.int 8) (fun point => QuantizeScalar.bodyValue w ctx.M
        (Dequant.float32 (data.get point))
        ((Dequant.float32 (scale.get [])).map (Spec.Quantize.inverse ctx.M))
        (Dequant.bits w (zero.get [])) 0xc3000000#32 0x42fe0000#32) =
      Quantize.result w c ctx.M data scale zero := by
    unfold Quantize.result Spec.Quantize.tensor
    congr 1
    funext point
    exact QuantizeScalar.compose w ctx.M (Dequant.float32 (data.get point))
      (Dequant.float32 (scale.get [])) (Dequant.bits w (zero.get [])) 0xc3000000#32 0x42fe0000#32
  exact run.trans (congrArg (fun t => Outcome.ok [RVal.tensor t]) resultEq)

#print axioms inverse_from_valid
#print axioms data_from_valid

end MerlinVerify.Rules.QuantizeLowering
