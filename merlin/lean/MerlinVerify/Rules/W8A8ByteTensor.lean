/- Nonpoison output of an actual checked quantization tensor stage under the explicit byte
model and pointwise non-NaN intermediate condition. No whole-program domain is inferred. -/
import MerlinVerify.Rules.W8A8ByteModel
import MerlinVerify.Rules.W8A8FloatTensor
import MerlinVerify.Sem.Defined

namespace MerlinVerify.Rules.W8A8ByteTensor

open MerlinVerify MerlinVerify.Sem.Ops.Linalg W8A8TensorOperands

def result (base : FloatModel) (shape parameters : List Nat) (map : AffineMap)
    (get inverse : Idx → FBits .f32) : Tensor :=
  Tensor.canon shape (.int 8) fun point => W8A8Scalar.quantize (W8A8ByteModel.extend base)
    (some (get point)) (some (inverse (coordinate parameters map point))) 0x42fe0000#32 0xc2fe0000#32

def Domain (base : FloatModel) (shape parameters : List Nat) (map : AffineMap)
    (get inverse : Idx → FBits .f32) : Prop :=
  ∀ point, Idx.InBounds shape point → W8A8ByteFloat.isNaN
    (W8A8ByteModel.rounded base (get point) (inverse (coordinate parameters map point))) = false

theorem result_defined {base : FloatModel} {shape parameters : List Nat} {map : AffineMap}
    {get inverse : Idx → FBits .f32} (nonNaN : Domain base shape parameters map get inverse) :
    (RVal.tensor (result base shape parameters map get inverse)).Defined := by
  intro point hp
  change Idx.InBounds shape point at hp
  change (Tensor.canon shape (.int 8) _).get point ≠ .poison
  simp only [Tensor.canon, hp, ↓reduceIte]
  exact W8A8ByteModel.quantize_defined base _ _ (nonNaN point hp)

/-- The region-body premise is supplied by the existing proved body checker. Geometry checks
the actual indexing maps. The result is initialized independently of the unused destination. -/
theorem generic_result (ctx : Ctx) (call : CallSem) (env : Env)
    (shape parameters : List Nat) (map : AffineMap) (maps : List AffineMap)
    (iters : List IterKind) (actual : Region)
    (identity : maps = [identityMap shape.length, map, identityMap shape.length])
    (geometry : Check.StructuredOk maps iters.length
      [.tensor shape (.float .f32), .tensor parameters (.float .f32), .tensor shape (.int 8)] shape)
    (body : ∀ c : Ctx, ∀ x inv : Option (FBits .f32), ∀ old : Val,
      evalRegion Sem.Ops.opSem c call env actual
        [.scalar (W8A8Scalar.floatValue x), .scalar (W8A8Scalar.floatValue inv), .scalar old] =
        .ok [.scalar (W8A8Scalar.quantize c.M x inv 0x42fe0000#32 0xc2fe0000#32)])
    (get inverse : Idx → FBits .f32) (initial : Idx → Val)
    (nonNaN : Domain ctx.M shape parameters map get inverse) :
    ∃ output, genericSem { ctx with M := W8A8ByteModel.extend ctx.M } maps iters
      [(W8A8FloatTensor.floatInput shape (fun p => some (get p))).value,
       (W8A8FloatTensor.floatInput parameters (fun p => some (inverse p))).value]
      [.tensor ⟨shape, .int 8, initial⟩] (fun c => evalRegion Sem.Ops.opSem c call env actual) =
        .ok [.tensor output] ∧ (RVal.tensor output).Defined := by
  refine ⟨result ctx.M shape parameters map get inverse, ?_, result_defined nonNaN⟩
  exact W8A8FloatTensor.quantize_sound { ctx with M := W8A8ByteModel.extend ctx.M } call env
    shape parameters map maps iters actual identity geometry 0x42fe0000#32 0xc2fe0000#32 body
      (fun p => some (get p)) (fun p => some (inverse p)) initial

#print axioms result_defined
#print axioms generic_result

end MerlinVerify.Rules.W8A8ByteTensor
