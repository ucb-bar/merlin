/- Exact subtract/divide tensors for the checked broadcast of one shared row tensor.
Both reshape steps and the broadcast coordinate are part of the value proof. -/
import MerlinVerify.Rules.W8A8RowBroadcast
import MerlinVerify.Rules.W8A8AttentionMapTensor

namespace MerlinVerify.Rules.W8A8AttentionPointwiseRows

open MerlinVerify Sem.Ops.Linalg W8A8AttentionMapBody

def rowTensor (shape : List Nat) (words : Idx → FBits .f32) : Tensor :=
  Tensor.canon shape (.float .f32) (fun cell => W8A8OrderedBody.value (words cell))

def expanded (shape : List Nat) (words : Idx → FBits .f32) : Tensor :=
  ReshapeIndex.reshape
    (ReshapeIndex.reshape (rowTensor shape words) [shapeSize shape] (.float .f32))
    (shape ++ [1]) (.float .f32)

theorem expanded_initialized (shape : List Nat) (words : Idx → FBits .f32)
    (point : Idx) (bounds : Idx.InBounds (shape ++ [1]) point) :
    (expanded shape words).get point = W8A8OrderedBody.value (words (point.take shape.length)) := by
  have rowBounds := W8A8RowBroadcast.leading_inBounds bounds
  have actual := W8A8RowReshape.collapsed_then_unit_get (rowTensor shape words) [shapeSize shape]
    (point.take shape.length) (by change shapeSize [shapeSize shape] = shapeSize shape; simp [shapeSize]) rowBounds
  change (expanded shape words).get (point.take shape.length ++ [0]) = _ at actual
  rw [← W8A8RowBroadcast.unit_point bounds] at actual
  simpa only [rowTensor, Tensor.canon, rowBounds, ↓reduceIte] using actual

theorem binary_sound {kind : BinaryKind} {actual : Region}
    (accepted : Check.PolicyW8A8AttentionMapBody.checkBinary kind actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8RoundedReal.model fallback)
    (shape : List Nat) (width : Nat) (iters : List IterKind)
    (geometry : Check.StructuredOk
      [identityMap (shape ++ [width]).length, W8A8AttentionTemplate.broadcastMap shape.length,
        identityMap (shape ++ [width]).length] iters.length
      [.tensor (shape ++ [width]) (.float .f32), .tensor (shape ++ [1]) (.float .f32),
        .tensor (shape ++ [width]) (.float .f32)] (shape ++ [width]))
    (inputValues initial : Idx → Val) (input rowWords : Idx → FBits .f32)
    (initialized : ∀ point, Idx.InBounds (shape ++ [width]) point →
      inputValues point = W8A8OrderedBody.value (input point)) :
    genericSem ctx
      [identityMap (shape ++ [width]).length, W8A8AttentionTemplate.broadcastMap shape.length,
        identityMap (shape ++ [width]).length] iters
      [.tensor ⟨shape ++ [width], .float .f32, inputValues⟩, .tensor (expanded shape rowWords)]
      [.tensor ⟨shape ++ [width], .float .f32, initial⟩]
      (fun c => evalRegion W8A8SourceSemantics.opSem c call env actual) =
      .ok [.tensor (Tensor.canon (shape ++ [width]) (.float .f32) fun point =>
        W8A8OrderedBody.value (binary kind (input point) (rowWords (point.take shape.length))))] := by
  let right : Idx → FBits .f32 := fun point => rowWords (point.take shape.length)
  have executed := W8A8AttentionMapTensor.binary_sound accepted ctx call env fallback interpretation
    (shape ++ [width]) (shape ++ [1]) (shape ++ [width])
    (identityMap (shape ++ [width]).length) (W8A8AttentionTemplate.broadcastMap shape.length) iters
    geometry inputValues (expanded shape rowWords).get initial input right initialized
    (expanded_initialized shape rowWords)
  have results :
      Tensor.canon (shape ++ [width]) (.float .f32) (fun point => W8A8OrderedBody.value
        (binary kind (input (W8A8TensorOperands.coordinate (shape ++ [width])
          (identityMap (shape ++ [width]).length) point))
          (right (W8A8TensorOperands.coordinate (shape ++ [1])
            (W8A8AttentionTemplate.broadcastMap shape.length) point)))) =
      Tensor.canon (shape ++ [width]) (.float .f32) (fun point =>
        W8A8OrderedBody.value (binary kind (input point) (rowWords (point.take shape.length)))) := by
    apply Tensor.canon_ext
    intro point bounds
    rw [W8A8TensorOperands.coordinate_identity bounds, W8A8RowBroadcast.coordinate bounds]
    have lengths := (W8A8RowBroadcast.leading_inBounds bounds).1
    have takeUnit : (point.take shape.length ++ [0]).take shape.length = point.take shape.length := by
      calc
        _ = (point.take shape.length ++ [0]).take (point.take shape.length).length :=
          congrArg (fun length => (point.take shape.length ++ [0]).take length) lengths.symm
        _ = _ := List.take_left
    simp only [right, takeUnit]
  rw [results] at executed
  exact executed

end MerlinVerify.Rules.W8A8AttentionPointwiseRows
