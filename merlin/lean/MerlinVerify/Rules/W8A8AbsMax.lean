/- W8A8 abs/max tensor reduction. The original lexicographic trace is retained; no
commutativity, associativity, IEEE behavior or zero identity is assumed for floating point. -/
import MerlinVerify.Rules.W8A8TensorTrace
import MerlinVerify.Rules.W8A8FloatTensor
import MerlinVerify.Check.PolicyW8A8Scalar

namespace MerlinVerify.Rules.W8A8AbsMax

open MerlinVerify MerlinVerify.Sem.Ops.Linalg W8A8Scalar W8A8TensorOperands W8A8FloatTensor

def step (M : FloatModel) (get : Idx → Option (FBits .f32)) (point : Idx)
    (acc : Option (FBits .f32)) : Option (FBits .f32) := do
  let value ← get point
  let old ← acc
  pure (M.op (Spec.Dequant.defaultKey "arith.maximumf") .f32
    [M.op (Spec.Dequant.defaultKey "math.absf") .f32 [value], old])

theorem step_value (M : FloatModel) (get : Idx → Option (FBits .f32)) (point : Idx)
    (acc : Option (FBits .f32)) :
    floatValue (step M get point acc) = W8A8Scalar.absMax M (get point) acc := by
  cases h : get point <;> cases acc <;> simp [step, W8A8Scalar.absMax, h]

/-- Every access and repeated write follows the actual checked reduction geometry. Initial
values are explicit; proving the actual floating zero producer remains a separate site step. -/
theorem generic_sound {ctx : Ctx} {iters : List IterKind} {shape outShape : List Nat}
    {outMap : AffineMap} {actual : Region}
    (geometry : Check.StructuredOk [identityMap shape.length, outMap] iters.length
      [.tensor shape (.float .f32), .tensor outShape (.float .f32)] shape)
    (accepted : Check.PolicyW8A8Scalar.checkAbsMax actual = true)
    (call : CallSem) (env : Env) (get : Idx → Option (FBits .f32))
    (initial : Idx → Option (FBits .f32)) :
    genericSem ctx [identityMap shape.length, outMap] iters [(floatInput shape get).value]
      [.tensor (W8A8TensorTrace.pack outShape (.float .f32) floatValue initial)]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Tensor.canon outShape (.float .f32) (fun cell => floatValue
        (W8A8TensorTrace.accumulate (coordinate outShape outMap) (step ctx.M get)
          initial (allPoints shape) cell)))] := by
  have hbounds := geometry.inBounds
  cases hbounds with
  | cons _ tail =>
    cases tail with
    | cons ho _ =>
      simp only [Check.VTy.shape] at ho
      apply W8A8TensorTrace.generic_sound (ib := ctx.indexBits)
        (inTys := [.tensor shape (.float .f32)]) geometry
        (.cons (floatInput_typed ctx.indexBits shape get) .nil) floatValue initial
        (floatInput_typed ctx.indexBits outShape initial) (by rfl) _
        (coordinate outShape outMap) (step ctx.M get)
        (fun p hp => coordinate_sound ho hp)
      intro point hp state
      have input : readAt (floatInput shape get).value (identityMap shape.length) point =
          .ok (floatValue (get point)) := readAt_identity hp
      have output : readAt
          (.tensor (W8A8TensorTrace.pack outShape (.float .f32) floatValue state)) outMap point =
          .ok (floatValue (state (coordinate outShape outMap point))) := by
        simp only [readAt, W8A8TensorTrace.pack]
        rw [coordinate_sound ho hp]
      have body := Check.PolicyW8A8Scalar.checkAbsMax_sound accepted
        { ctx with loopIndex := point } call env (get point) (state (coordinate outShape outMap point))
      simp [bodyAt, input, output, body, step_value]

#print axioms generic_sound

end MerlinVerify.Rules.W8A8AbsMax
