import MerlinVerify.Rules.QuantizeScalar
import MerlinVerify.Rules.W8A8FloatTensor

namespace MerlinVerify.Rules.QuantizeTensor

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops.Linalg
open QuantizeScalar W8A8TensorOperands
open MerlinVerify.Rules.DequantScalar

def inverseRegion : Region :=
  .mk [.mk [⟨0, .float .f32⟩, ⟨1, .float .f32⟩] inverseBody]

def region (w : Nat) : Region :=
  .mk [.mk [⟨0, .float .f32⟩, ⟨1, .float .f32⟩, ⟨2, .int w⟩, ⟨3, .int 8⟩] body]

def captures (lower upper : FBits .f32) : Env :=
  [(100, .scalar (.some (.flt .f32 lower))), (101, .scalar (.some (.flt .f32 upper)))]

theorem inverse_region_sound {actual : Region} (accepted : alphaRegion [] inverseRegion actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (scale : Option (FBits .f32)) (old : Val) :
    evalRegion Sem.Ops.opSem ctx call env actual [.scalar (floatValue scale), .scalar old] =
      .ok [.scalar (floatValue (scale.map (Spec.Quantize.inverse ctx.M)))] := by
  have he := alphaRegion_sound call call [] inverseRegion actual ctx [] env accepted
    (fun _ _ h => by simp [Rho.find] at h)
    (fun _ _ => rfl)
  rw [← he]
  simpa [inverseRegion, evalRegion, evalBlocks, evalBlock, Env.bindAll] using
    inverse_sound ctx call scale old

theorem region_sound {w lowerId upperId : Nat} {actual : Region}
    (accepted : alphaRegion [(100, lowerId), (101, upperId)] (region w) actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (lower upper : FBits .f32)
    (known : Corr [(100, lowerId), (101, upperId)] (captures lower upper) env)
    (x inv : Option (FBits .f32)) (zero : Option (BitVec w)) (old : Val) :
    evalRegion Sem.Ops.opSem ctx call env actual
      [.scalar (floatValue x), .scalar (floatValue inv), .scalar (intValue w zero), .scalar old] =
      .ok [.scalar (bodyValue w ctx.M x inv zero lower upper)] := by
  have he := alphaRegion_sound call call _ (region w) actual ctx (captures lower upper) env
    accepted known (fun _ _ => rfl)
  rw [← he]
  simpa [region, captures, evalRegion, evalBlocks, evalBlock, Env.bindAll] using
    body_sound w ctx call x inv zero lower upper old

/-- Integer conversion reads the executed value's width. The block annotation is checked
separately by MLIR; it does not supply a runtime-width premise to this scalar theorem. -/
theorem region_sound_at {wireWidth lowerId upperId : Nat} {actual : Region} (w : Nat)
    (accepted : alphaRegion [(100, lowerId), (101, upperId)] (region wireWidth) actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (lower upper : FBits .f32)
    (known : Corr [(100, lowerId), (101, upperId)] (captures lower upper) env)
    (x inv : Option (FBits .f32)) (zero : Option (BitVec w)) (old : Val) :
    evalRegion Sem.Ops.opSem ctx call env actual
      [.scalar (floatValue x), .scalar (floatValue inv), .scalar (intValue w zero), .scalar old] =
      .ok [.scalar (bodyValue w ctx.M x inv zero lower upper)] := by
  have he := alphaRegion_sound call call _ (region wireWidth) actual ctx (captures lower upper) env
    accepted known (fun _ _ => rfl)
  rw [← he]
  simpa [region, captures, evalRegion, evalBlocks, evalBlock, Env.bindAll] using
    body_sound w ctx call x inv zero lower upper old

theorem inverse_generic_sound {actual : Region} (accepted : alphaRegion [] inverseRegion actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (shape : List Nat)
    (maps : List AffineMap) (iters : List IterKind)
    (layout : maps = [identityMap shape.length, identityMap shape.length])
    (geometry : StructuredOk maps iters.length
      [.tensor shape (.float .f32), .tensor shape (.float .f32)] shape)
    (scale : Idx → Option (FBits .f32)) (old : Idx → Val) :
    genericSem ctx maps iters [(W8A8FloatTensor.floatInput shape scale).value]
      [.tensor ⟨shape, .float .f32, old⟩]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Tensor.canon shape (.float .f32)
        (fun p => floatValue ((scale p).map (Spec.Quantize.inverse ctx.M))))] := by
  apply W8A8FloatTensor.unary_sound ctx call env shape maps iters actual layout geometry
    (fun M s => floatValue (s.map (Spec.Quantize.inverse M))) _ scale old
  intro c s initial
  exact inverse_region_sound accepted c call env s initial

def intInput (w : Nat) (shape : List Nat) (get : Idx → Option (BitVec w)) : Input :=
  ⟨shape, .int w, fun p => intValue w (get p)⟩

def broadcast (rank : Nat) : AffineMap := ⟨rank, 0, []⟩

theorem coordinate_broadcast (rank : Nat) (point : Idx) :
    coordinate [] (broadcast rank) point = [] := by
  simp [coordinate, broadcast, evalMap, toIdx]

theorem generic_sound {w lowerId upperId : Nat} {actual : Region}
    (accepted : alphaRegion [(100, lowerId), (101, upperId)] (region w) actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (lower upper : FBits .f32)
    (known : Corr [(100, lowerId), (101, upperId)] (captures lower upper) env)
    (shape : List Nat) (maps : List AffineMap) (iters : List IterKind)
    (layout : maps = [identityMap shape.length, broadcast shape.length,
      broadcast shape.length, identityMap shape.length])
    (geometry : StructuredOk maps iters.length
      [.tensor shape (.float .f32), .tensor [] (.float .f32), .tensor [] (.int w),
       .tensor shape (.int 8)] shape)
    (input : Idx → Option (FBits .f32)) (inv : Option (FBits .f32))
    (zero : Option (BitVec w)) (old : Idx → Val) :
    genericSem ctx maps iters
      [(W8A8FloatTensor.floatInput shape input).value,
       (W8A8FloatTensor.floatInput [] (fun _ => inv)).value,
       (intInput w [] (fun _ => zero)).value]
      [.tensor ⟨shape, .int 8, old⟩]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Tensor.canon shape (.int 8)
        (fun p => bodyValue w ctx.M (input p) inv zero lower upper))] := by
  apply W8A8Pointwise.generic_sound ctx call env
    [W8A8FloatTensor.floatInput shape input, W8A8FloatTensor.floatInput [] (fun _ => inv),
      intInput w [] (fun _ => zero)] ⟨shape, .int 8, old⟩ maps iters actual
    geometry (by simp [layout]) _
  intro point hp initial
  simp [values, layout, Input.at, intInput, W8A8FloatTensor.floatInput,
    coordinate_identity hp]
  exact region_sound accepted { ctx with loopIndex := point } call env lower upper known
    (input point) inv zero initial

#print axioms inverse_region_sound
#print axioms region_sound
#print axioms inverse_generic_sound
#print axioms generic_sound

end MerlinVerify.Rules.QuantizeTensor
