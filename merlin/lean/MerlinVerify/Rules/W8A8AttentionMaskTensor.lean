/- Exact actual mask inversion and score-selection tensors, through checked bodies
and maps. Destinations may contain poison; only the actual input reads are required. -/
import MerlinVerify.Check.PolicyW8A8AttentionMaskBody
import MerlinVerify.Rules.W8A8SourcePointwise
import MerlinVerify.Rules.W8A8AttentionMaskRows

namespace MerlinVerify.Rules.W8A8AttentionMaskTensor

open MerlinVerify Sem.Ops.Linalg W8A8TensorOperands W8A8AttentionMaskBody

def scalarMap (rank : Nat) : AffineMap := ⟨rank, 0, []⟩

theorem select_sound {actual : Region}
    (accepted : Check.PolicyW8A8AttentionMaskBody.check .select actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (maskShape shape : List Nat) (maskMap : AffineMap)
    (iters : List IterKind)
    (geometry : Check.StructuredOk
      [maskMap, scalarMap shape.length, identityMap shape.length, identityMap shape.length] iters.length
      [.tensor maskShape (.int 1), .tensor [] (.float .f32),
        .tensor shape (.float .f32), .tensor shape (.float .f32)] shape)
    (maskValues scoreValues initial : Idx → Val) (masked : Idx → Bool) (scores : Idx → FBits .f32)
    (maskInitialized : ∀ cell, Idx.InBounds maskShape cell → maskValues cell = boolean (masked cell))
    (scoresInitialized : ∀ point, Idx.InBounds shape point → scoreValues point = W8A8OrderedBody.value (scores point)) :
    genericSem ctx [maskMap, scalarMap shape.length, identityMap shape.length, identityMap shape.length] iters
      [.tensor ⟨maskShape, .int 1, maskValues⟩,
        .tensor (Tensor.full [] (.float .f32) (W8A8OrderedBody.value W8A8MaskedMaximum.negativeInfinity)),
        .tensor ⟨shape, .float .f32, scoreValues⟩] [.tensor ⟨shape, .float .f32, initial⟩]
      (fun c => evalRegion W8A8SourceSemantics.opSem c call env actual) =
      .ok [.tensor (Tensor.canon shape (.float .f32) fun point => W8A8OrderedBody.value
        (if masked (coordinate maskShape maskMap point) then W8A8MaskedMaximum.negativeInfinity else scores point))] := by
  apply W8A8SourcePointwise.generic_sound W8A8SourceSemantics.opSem ctx call env
    [⟨maskShape, .int 1, maskValues⟩,
      ⟨[], .float .f32, (Tensor.full [] (.float .f32) (W8A8OrderedBody.value W8A8MaskedMaximum.negativeInfinity)).get⟩,
      ⟨shape, .float .f32, scoreValues⟩]
    ⟨shape, .float .f32, initial⟩ _ iters actual geometry (by rfl) _
  intro point bounds old
  have mappings := geometry.inBounds
  cases mappings with
  | cons maskMapping remaining =>
    cases remaining with
    | cons _ remaining =>
      cases remaining with
      | cons _ _ =>
        have maskValue := maskInitialized _ (W8A8SourcePointwise.coordinate_inBounds maskMapping bounds)
        have scoreValue := scoresInitialized point bounds
        have same := Check.PolicyW8A8AttentionMaskBody.check_sound accepted { ctx with loopIndex := point }
          call env [.scalar (boolean (masked (coordinate maskShape maskMap point))),
            .scalar (W8A8OrderedBody.value W8A8MaskedMaximum.negativeInfinity),
            .scalar (W8A8OrderedBody.value (scores point)), .scalar old]
        have scalar := W8A8AttentionMaskBody.select_sound { ctx with loopIndex := point } call env
          (masked (coordinate maskShape maskMap point)) W8A8MaskedMaximum.negativeInfinity (scores point) old
        have emptyPoint : coordinate [] (scalarMap shape.length) point = [] := by rfl
        simpa [values, Input.at, emptyPoint, Tensor.full, Tensor.canon, show Idx.InBounds [] [] by decide,
          maskValue, scoreValue, coordinate_identity bounds] using same.trans scalar

theorem inversion_sound {actual : Region}
    (accepted : Check.PolicyW8A8AttentionMaskBody.check .inversion actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (shape : List Nat) (iters : List IterKind)
    (geometry : Check.StructuredOk [identityMap shape.length, identityMap shape.length] iters.length
      [.tensor shape (.int 1), .tensor shape (.int 1)] shape)
    (inputValues initial : Idx → Val) (input : Idx → Bool)
    (initialized : ∀ point, Idx.InBounds shape point → inputValues point = boolean (input point)) :
    genericSem ctx [identityMap shape.length, identityMap shape.length] iters
      [.tensor ⟨shape, .int 1, inputValues⟩] [.tensor ⟨shape, .int 1, initial⟩]
      (fun c => evalRegion W8A8SourceSemantics.opSem c call env actual) =
      .ok [.tensor (Tensor.canon shape (.int 1) fun point => boolean (!(input point)))] := by
  apply W8A8SourcePointwise.generic_sound W8A8SourceSemantics.opSem ctx call env
    [⟨shape, .int 1, inputValues⟩] ⟨shape, .int 1, initial⟩ _ iters actual geometry (by rfl) _
  intro point bounds old
  have same := Check.PolicyW8A8AttentionMaskBody.check_sound accepted { ctx with loopIndex := point }
    call env [.scalar (boolean (input point)), .scalar old]
  have scalar := W8A8AttentionMaskBody.inversion_sound { ctx with loopIndex := point } call env (input point) old
  simpa [values, Input.at, coordinate_identity bounds, initialized point bounds] using same.trans scalar

end MerlinVerify.Rules.W8A8AttentionMaskTensor
