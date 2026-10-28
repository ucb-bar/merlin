/- Exact tensor results for checked softmax scalar maps. Actual maps select the
reads; only in-bounds input cells must be initialized, and destination poison is allowed. -/
import MerlinVerify.Check.PolicyW8A8AttentionMapBody
import MerlinVerify.Rules.W8A8SourcePointwise

namespace MerlinVerify.Rules.W8A8AttentionMapTensor

open MerlinVerify Sem.Ops.Linalg W8A8TensorOperands W8A8AttentionMapBody

theorem binary_sound {kind : BinaryKind} {actual : Region}
    (accepted : Check.PolicyW8A8AttentionMapBody.checkBinary kind actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8RoundedReal.model fallback)
    (leftShape rightShape shape : List Nat) (leftMap rightMap : AffineMap) (iters : List IterKind)
    (geometry : Check.StructuredOk [leftMap, rightMap, identityMap shape.length] iters.length
      [.tensor leftShape (.float .f32), .tensor rightShape (.float .f32), .tensor shape (.float .f32)] shape)
    (leftValues rightValues initial : Idx → Val) (left right : Idx → FBits .f32)
    (leftInitialized : ∀ cell, Idx.InBounds leftShape cell → leftValues cell = W8A8OrderedBody.value (left cell))
    (rightInitialized : ∀ cell, Idx.InBounds rightShape cell → rightValues cell = W8A8OrderedBody.value (right cell)) :
    genericSem ctx [leftMap, rightMap, identityMap shape.length] iters
      [.tensor ⟨leftShape, .float .f32, leftValues⟩, .tensor ⟨rightShape, .float .f32, rightValues⟩]
      [.tensor ⟨shape, .float .f32, initial⟩]
      (fun c => evalRegion W8A8SourceSemantics.opSem c call env actual) =
      .ok [.tensor (Tensor.canon shape (.float .f32) fun point => W8A8OrderedBody.value
        (binary kind (left (coordinate leftShape leftMap point)) (right (coordinate rightShape rightMap point))))] := by
  apply W8A8SourcePointwise.generic_sound W8A8SourceSemantics.opSem ctx call env
    [⟨leftShape, .float .f32, leftValues⟩, ⟨rightShape, .float .f32, rightValues⟩]
    ⟨shape, .float .f32, initial⟩ [leftMap, rightMap, identityMap shape.length] iters actual
    geometry (by rfl) _
  intro point bounds old
  have mappings := geometry.inBounds
  cases mappings with
  | cons leftMapping remaining =>
    cases remaining with
    | cons rightMapping _ =>
      have leftValue := leftInitialized _ (W8A8SourcePointwise.coordinate_inBounds leftMapping bounds)
      have rightValue := rightInitialized _ (W8A8SourcePointwise.coordinate_inBounds rightMapping bounds)
      have executed := Check.PolicyW8A8AttentionMapBody.checkBinary_sound accepted { ctx with loopIndex := point }
        call env fallback interpretation (left (coordinate leftShape leftMap point))
        (right (coordinate rightShape rightMap point)) old
      simpa [values, Input.at, leftValue, rightValue] using executed

theorem exponential_sound {actual : Region}
    (accepted : Check.PolicyW8A8AttentionMapBody.checkExponential actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8RoundedReal.model fallback)
    (inputShape shape : List Nat) (inputMap : AffineMap) (iters : List IterKind)
    (geometry : Check.StructuredOk [inputMap, identityMap shape.length] iters.length
      [.tensor inputShape (.float .f32), .tensor shape (.float .f32)] shape)
    (inputValues initial : Idx → Val) (input : Idx → FBits .f32)
    (initialized : ∀ cell, Idx.InBounds inputShape cell → inputValues cell = W8A8OrderedBody.value (input cell)) :
    genericSem ctx [inputMap, identityMap shape.length] iters
      [.tensor ⟨inputShape, .float .f32, inputValues⟩] [.tensor ⟨shape, .float .f32, initial⟩]
      (fun c => evalRegion W8A8SourceSemantics.opSem c call env actual) =
      .ok [.tensor (Tensor.canon shape (.float .f32) fun point =>
        W8A8OrderedBody.value (W8A8RoundedReal.exponential (input (coordinate inputShape inputMap point))))] := by
  apply W8A8SourcePointwise.generic_sound W8A8SourceSemantics.opSem ctx call env
    [⟨inputShape, .float .f32, inputValues⟩] ⟨shape, .float .f32, initial⟩
    [inputMap, identityMap shape.length] iters actual geometry (by rfl) _
  intro point bounds old
  have mappings := geometry.inBounds
  cases mappings with
  | cons inputMapping _ =>
    have inputValue := initialized _ (W8A8SourcePointwise.coordinate_inBounds inputMapping bounds)
    have executed := Check.PolicyW8A8AttentionMapBody.checkExponential_sound accepted { ctx with loopIndex := point }
      call env fallback interpretation (input (coordinate inputShape inputMap point)) old
    simpa [values, Input.at, inputValue] using executed

end MerlinVerify.Rules.W8A8AttentionMapTensor
