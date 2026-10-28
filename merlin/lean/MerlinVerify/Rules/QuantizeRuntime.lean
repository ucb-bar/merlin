import MerlinVerify.Rules.QuantizeSource
import MerlinVerify.Rules.QuantizeTensor

namespace MerlinVerify.Rules.QuantizeRuntime

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg
open QuantizeTensor W8A8TensorOperands DequantScalar DequantSource

def input (t : Tensor) : Input := ⟨t.shape, t.elt, t.get⟩

@[simp] theorem input_value (t : Tensor) : (input t).value = .tensor t := by
  cases t; rfl

def inverseTensor (M : FloatModel) (s : Tensor) : Tensor :=
  Tensor.canon [] (.float .f32) fun _ =>
    floatValue ((Dequant.float32 (s.get [])).map (Spec.Quantize.inverse M))

theorem inverse_actual {actual : Region}
    (accepted : alphaRegion [] inverseRegion actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (scale : Tensor)
    (scalar : scale.shape = []) (typed : Dequant.tensorTyped scale (.float .f32) = true)
    (maps : List AffineMap) (iters : List IterKind)
    (layout : maps = [identityMap 0, identityMap 0])
    (geometry : StructuredOk maps iters.length
      [(input scale).type, .tensor [] (.float .f32)] []) :
    genericSem ctx maps iters [.tensor scale] [.tensor (Tensor.full [] (.float .f32) .poison)]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (inverseTensor ctx.M scale)] := by
  change genericSem ctx maps iters [(input scale).value]
    [.tensor (Tensor.full [] (.float .f32) .poison)] _ = _
  apply W8A8Pointwise.generic_sound ctx call env [input scale]
    (Tensor.full [] (.float .f32) .poison) maps iters actual geometry
    (by simp [layout, Tensor.full, Tensor.canon]) _
  intro point hp old
  have pointNil : point = [] := (Idx.inBounds_nil_iff point).1 hp
  subst point
  have decoded := float_roundtrip (tensorTyped_get (p := []) typed
    (by simpa [scalar, Tensor.full, Tensor.canon] using hp))
  have body := inverse_region_sound accepted { ctx with loopIndex := [] } call env
    (Dequant.float32 (scale.get [])) old
  rw [decoded] at body
  simp [values, layout, Input.at, input, scalar, coordinate, identityMap, evalMap, toIdx]
  exact body

theorem data_actual {wireWidth lowerId upperId : Nat} {actual : Region} (w : Nat)
    (accepted : alphaRegion [(100, lowerId), (101, upperId)] (region wireWidth) actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (lower upper : FBits .f32)
    (known : Corr [(100, lowerId), (101, upperId)] (captures lower upper) env)
    (data inverse zero : Tensor)
    (dataTyped : Dequant.tensorTyped data (.float .f32) = true)
    (inverseShape : inverse.shape = []) (inverseTyped : Dequant.tensorTyped inverse (.float .f32) = true)
    (zeroShape : zero.shape = []) (zeroTyped : Dequant.tensorTyped zero (.int w) = true)
    (maps : List AffineMap) (iters : List IterKind)
    (layout : maps = [identityMap data.shape.length, broadcast data.shape.length,
      broadcast data.shape.length, identityMap data.shape.length])
    (geometry : StructuredOk maps iters.length
      [(input data).type, (input inverse).type, (input zero).type, .tensor data.shape (.int 8)] data.shape) :
    genericSem ctx maps iters [.tensor data, .tensor inverse, .tensor zero]
      [.tensor (Tensor.full data.shape (.int 8) .poison)]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Tensor.canon data.shape (.int 8) fun point =>
        QuantizeScalar.bodyValue w ctx.M (Dequant.float32 (data.get point))
          (Dequant.float32 (inverse.get [])) (Dequant.bits w (zero.get [])) lower upper)] := by
  change genericSem ctx maps iters [(input data).value, (input inverse).value, (input zero).value]
    [.tensor (Tensor.full data.shape (.int 8) .poison)] _ = _
  apply W8A8Pointwise.generic_sound ctx call env [input data, input inverse, input zero]
    (Tensor.full data.shape (.int 8) .poison) maps iters actual geometry
    (by simp [layout, Tensor.full, Tensor.canon]) _
  intro point hp old
  have nilBounds : Idx.InBounds [] ([] : Idx) := (Idx.inBounds_nil_iff []).2 rfl
  have hp' : Idx.InBounds data.shape point := hp
  have xval := float_roundtrip (tensorTyped_get dataTyped hp')
  have ival := float_roundtrip (tensorTyped_get (p := []) inverseTyped (by simpa [inverseShape] using nilBounds))
  have zval := bits_roundtrip (tensorTyped_get (p := []) zeroTyped (by simpa [zeroShape] using nilBounds))
  have body := region_sound_at w accepted { ctx with loopIndex := point } call env lower upper known
    (Dequant.float32 (data.get point)) (Dequant.float32 (inverse.get [])) (Dequant.bits w (zero.get [])) old
  rw [xval, ival, zval] at body
  simp [values, layout, Input.at, input, inverseShape, zeroShape,
    coordinate_identity hp', coordinate_broadcast]
  exact body

#print axioms inverse_actual
#print axioms data_actual

end MerlinVerify.Rules.QuantizeRuntime
