/- Ordered floating-point W8A8 stage contracts lifted to complete tensor overwrites. These
remain implementation components; the independently reviewed policy transformer is separate. -/
import MerlinVerify.Rules.W8A8Pointwise
import MerlinVerify.Rules.W8A8Scalar

namespace MerlinVerify.Rules.W8A8FloatTensor

open MerlinVerify MerlinVerify.Sem.Ops.Linalg W8A8TensorOperands W8A8Scalar

def floatInput (shape : List Nat) (get : Idx → Option (FBits .f32)) : Input :=
  ⟨shape, .float .f32, fun p => floatValue (get p)⟩

def intInput (shape : List Nat) (get : Idx → Option (BitVec 32)) : Input :=
  ⟨shape, .int 32, fun p => intValue (get p)⟩

theorem floatInput_typed (ib : Nat) (shape : List Nat) (get : Idx → Option (FBits .f32)) :
    Check.RVal.HasTy ib (floatInput shape get).value (.tensor shape (.float .f32)) := by
  refine ⟨rfl, rfl, ?_⟩
  intro cell _
  cases h : get cell <;> simp [floatInput, floatValue, Check.Val.HasTy, Scalar.conforms, h]

/-- Unary scale/reciprocal loops have an identity-indexed input and output. The old destination
may contain arbitrary values, including poison; no numerical identity is needed. -/
theorem unary_sound (ctx : Ctx) (call : CallSem) (env : Env) (shape : List Nat)
    (maps : List AffineMap) (iters : List IterKind) (actual : Region)
    (hm : maps = [identityMap shape.length, identityMap shape.length])
    (geometry : Check.StructuredOk maps iters.length
      [.tensor shape (.float .f32), .tensor shape (.float .f32)] shape)
    (contract : FloatModel → Option (FBits .f32) → Val)
    (body : ∀ c : Ctx, ∀ x : Option (FBits .f32), ∀ old : Val,
      evalRegion Sem.Ops.opSem c call env actual [.scalar (floatValue x), .scalar old] =
        .ok [.scalar (contract c.M x)])
    (get : Idx → Option (FBits .f32)) (initial : Idx → Val) :
    genericSem ctx maps iters [(floatInput shape get).value]
      [.tensor ⟨shape, .float .f32, initial⟩]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Tensor.canon shape (.float .f32) (fun p => contract ctx.M (get p)))] := by
  apply W8A8Pointwise.generic_sound ctx call env [floatInput shape get]
    ⟨shape, .float .f32, initial⟩ maps iters actual
    geometry (by simp [hm]) _
  intro point hp old
  simpa [values, hm, Input.at, floatInput, coordinate_identity hp] using
    body { ctx with loopIndex := point } (get point) old

/-- Quantize uses the data value at the output coordinate and the reciprocal at its checked
parameter coordinate. The exact multiply/round/clamp/convert order is inherited unchanged. -/
theorem quantize_sound (ctx : Ctx) (call : CallSem) (env : Env)
    (shape paramShape : List Nat) (paramMap : AffineMap) (maps : List AffineMap)
    (iters : List IterKind) (actual : Region)
    (hm : maps = [identityMap shape.length, paramMap, identityMap shape.length])
    (geometry : Check.StructuredOk maps iters.length
      [.tensor shape (.float .f32), .tensor paramShape (.float .f32), .tensor shape (.int 8)] shape)
    (upper lower : FBits .f32)
    (body : ∀ c : Ctx, ∀ x inv : Option (FBits .f32), ∀ old : Val,
      evalRegion Sem.Ops.opSem c call env actual
        [.scalar (floatValue x), .scalar (floatValue inv), .scalar old] =
        .ok [.scalar (W8A8Scalar.quantize c.M x inv upper lower)])
    (get reciprocal : Idx → Option (FBits .f32)) (initial : Idx → Val) :
    genericSem ctx maps iters [(floatInput shape get).value, (floatInput paramShape reciprocal).value]
      [.tensor ⟨shape, .int 8, initial⟩]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Tensor.canon shape (.int 8) (fun point => W8A8Scalar.quantize ctx.M
        (get point) (reciprocal (coordinate paramShape paramMap point)) upper lower))] := by
  apply W8A8Pointwise.generic_sound ctx call env
    [floatInput shape get, floatInput paramShape reciprocal] ⟨shape, .int 8, initial⟩ maps iters actual
    geometry (by simp [hm]) _
  intro point hp old
  simpa [values, hm, Input.at, floatInput, coordinate_identity hp] using
    body { ctx with loopIndex := point } (get point)
      (reciprocal (coordinate paramShape paramMap point)) old

/-- Requantization reads each scale through its own checked projection. The single i32-to-f32
conversion and left-to-right scale multiplications are preserved for zero, one or two scales. -/
theorem requantize_sound (ctx : Ctx) (call : CallSem) (env : Env)
    (count : Nat) (hc : count ≤ 2) (shape leftShape rightShape : List Nat)
    (leftMap rightMap : AffineMap) (maps : List AffineMap) (iters : List IterKind) (actual : Region)
    (hm : maps = identityMap shape.length ::
      (([leftMap, rightMap].take count) ++ [identityMap shape.length]))
    (geometry : Check.StructuredOk maps iters.length
      ([.tensor shape (.int 32)] ++
        ([.tensor leftShape (.float .f32), .tensor rightShape (.float .f32)].take count) ++
        [.tensor shape (.float .f32)]) shape)
    (body : ∀ c : Ctx, ∀ acc : Option (BitVec 32), ∀ left right : Option (FBits .f32), ∀ old : Val,
      evalRegion Sem.Ops.opSem c call env actual
        ([.scalar (intValue acc)] ++
          ([.scalar (floatValue left), .scalar (floatValue right)].take count) ++ [.scalar old]) =
        .ok [.scalar (W8A8Scalar.requantize c.M acc ([left, right].take count))])
    (get : Idx → Option (BitVec 32)) (left right : Idx → Option (FBits .f32)) (initial : Idx → Val) :
    genericSem ctx maps iters
      ((intInput shape get :: ([floatInput leftShape left, floatInput rightShape right].take count)).map Input.value)
      [.tensor ⟨shape, .float .f32, initial⟩]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Tensor.canon shape (.float .f32) (fun point => W8A8Scalar.requantize ctx.M
        (get point) ([left (coordinate leftShape leftMap point),
          right (coordinate rightShape rightMap point)].take count)))] := by
  have casesCount : count = 0 ∨ count = 1 ∨ count = 2 := by omega
  rcases casesCount with rfl | rfl | rfl
  all_goals
    apply W8A8Pointwise.generic_sound ctx call env _ ⟨shape, .float .f32, initial⟩
      maps iters actual (by simpa [Input.type, intInput, floatInput] using geometry) (by simp [hm]) _
  all_goals intro point hp old
  all_goals
    simpa [values, hm, Input.at, intInput, floatInput, coordinate_identity hp] using
      body { ctx with loopIndex := point } (get point)
        (left (coordinate leftShape leftMap point)) (right (coordinate rightShape rightMap point)) old

#print axioms unary_sound
#print axioms quantize_sound
#print axioms requantize_sound

end MerlinVerify.Rules.W8A8FloatTensor
