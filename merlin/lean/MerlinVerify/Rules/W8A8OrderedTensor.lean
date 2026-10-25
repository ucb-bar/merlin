/- Actual floating contraction execution, including repeated writes and arbitrary unobserved
runtime cells. The checked body and independently checked coordinates determine the trace. -/
import MerlinVerify.Check.PolicyW8A8OrderedBody
import MerlinVerify.Rules.W8A8OrderedCells

namespace MerlinVerify.Rules.W8A8OrderedTensor

open MerlinVerify Sem.Ops.Linalg W8A8OrderedCells

theorem bodyAt_sound {ib : Nat} {actual : Region}
    (accepted : Check.PolicyW8A8OrderedBody.check actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8ByteModel.extend (W8A8IEEEPrimitives.model fallback))
    (left right : RVal) (leftMap rightMap outMap : AffineMap) (shape : List Nat)
    (a b : Idx → FBits .f32) (coordinate : Idx → Idx) (point : Idx)
    (leftRead : readAt left leftMap point = .ok (W8A8OrderedBody.value (a point)))
    (rightRead : readAt right rightMap point = .ok (W8A8OrderedBody.value (b point)))
    (outputIndex : (evalMap outMap point >>= toIdx shape) = some (coordinate point))
    (outputBounds : Idx.InBounds shape (coordinate point))
    (state : Idx → Val) (typed : typedState ib shape state) :
    bodyAt ctx [left, right] [leftMap, rightMap] [outMap]
      [W8A8TensorTrace.pack shape (.float .f32) id state]
      (fun c => evalRegion W8A8SourceSemantics.opSem c call env actual) point =
      .ok [step a b point (state (coordinate point))] := by
  have outputRead : readAt (.tensor (W8A8TensorTrace.pack shape (.float .f32) id state)) outMap point =
      .ok (state (coordinate point)) := by
    simp only [readAt, W8A8TensorTrace.pack, outputIndex]
    rfl
  have executed := Check.PolicyW8A8OrderedBody.check_sound accepted { ctx with loopIndex := point }
    call env fallback interpretation (a point) (b point) (Sem.Ops.Dequant.float32 (state (coordinate point)))
  rw [decode_typed (typed _ outputBounds)] at executed
  simp [bodyAt, leftRead, rightRead, outputRead, executed, step, Check.PolicyW8A8OrderedBody.result]

theorem generic_sound {ib : Nat} {ctx : Ctx} {actual : Region} {iters : List IterKind}
    {leftShape rightShape shape ranges : List Nat} {left right : RVal}
    {leftMap rightMap outMap : AffineMap}
    (geometry : Check.StructuredOk [leftMap, rightMap, outMap] iters.length
      [.tensor leftShape (.float .f32), .tensor rightShape (.float .f32), .tensor shape (.float .f32)] ranges)
    (typedLeft : Check.RVal.HasTy ib left (.tensor leftShape (.float .f32)))
    (typedRight : Check.RVal.HasTy ib right (.tensor rightShape (.float .f32)))
    (accepted : Check.PolicyW8A8OrderedBody.check actual = true)
    (call : CallSem) (env : Env) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8ByteModel.extend (W8A8IEEEPrimitives.model fallback))
    (a b : Idx → FBits .f32) (initial : Idx → Val) (coordinate : Idx → Idx)
    (initialTyped : typedState ib shape initial)
    (leftRead : ∀ point, Idx.InBounds ranges point →
      readAt left leftMap point = .ok (W8A8OrderedBody.value (a point)))
    (rightRead : ∀ point, Idx.InBounds ranges point →
      readAt right rightMap point = .ok (W8A8OrderedBody.value (b point)))
    (outputIndex : ∀ point, Idx.InBounds ranges point →
      (evalMap outMap point >>= toIdx shape) = some (coordinate point))
    (outputBounds : ∀ point, Idx.InBounds ranges point → Idx.InBounds shape (coordinate point)) :
    genericSem ctx [leftMap, rightMap, outMap] iters [left, right]
      [.tensor (W8A8TensorTrace.pack shape (.float .f32) id initial)]
      (fun c => evalRegion W8A8SourceSemantics.opSem c call env actual) =
      .ok [.tensor (Tensor.canon shape (.float .f32)
        (W8A8TensorTrace.accumulate coordinate (step a b) initial (allPoints ranges)))] := by
  apply W8A8TypedTensorTrace.generic_sound (inTys :=
    [.tensor leftShape (.float .f32), .tensor rightShape (.float .f32)]) geometry
    (.cons typedLeft (.cons typedRight .nil)) id initial
    (by exact ⟨rfl, rfl, initialTyped⟩) (by rfl) _ coordinate (step a b) (typedState ib shape)
    initialTyped outputIndex
  · intro point _ state typed
    exact update_typed typed coordinate a b point
  · intro point bounds state typed
    exact bodyAt_sound accepted ctx call env fallback interpretation left right leftMap rightMap outMap
      shape a b coordinate point (leftRead point bounds) (rightRead point bounds)
      (outputIndex point bounds) (outputBounds point bounds) state typed

/-- Complete tensor execution and initialized exponent bounds for every actual output cell.
The reduction count comes from checked projection dimensions, not the total tensor grid. -/
theorem projected_bounded {ib : Nat} {ctx : Ctx} {actual : Region} {iters : List IterKind}
    {leftShape rightShape ranges : List Nat} {mask : List Bool} {left right : RVal}
    {leftMap rightMap : AffineMap} {leftCap rightCap : Nat}
    (lengths : mask.length = ranges.length)
    (geometry : Check.StructuredOk [leftMap, rightMap, W8A8ProjectionTrace.outputMap mask] iters.length
      [.tensor leftShape (.float .f32), .tensor rightShape (.float .f32),
        .tensor (W8A8ProjectionTrace.select mask ranges) (.float .f32)] ranges)
    (typedLeft : Check.RVal.HasTy ib left (.tensor leftShape (.float .f32)))
    (typedRight : Check.RVal.HasTy ib right (.tensor rightShape (.float .f32)))
    (accepted : Check.PolicyW8A8OrderedBody.check actual = true)
    (call : CallSem) (env : Env) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8ByteModel.extend (W8A8IEEEPrimitives.model fallback))
    (a b : Idx → FBits .f32) (initial : Idx → Val)
    (initialized : ∀ cell, Idx.InBounds (W8A8ProjectionTrace.select mask ranges) cell →
      initial cell = W8A8OrderedBody.value 0#32)
    (leftRead : ∀ point, Idx.InBounds ranges point →
      readAt left leftMap point = .ok (W8A8OrderedBody.value (a point)))
    (rightRead : ∀ point, Idx.InBounds ranges point →
      readAt right rightMap point = .ok (W8A8OrderedBody.value (b point)))
    (inputs : ∀ point, Idx.InBounds ranges point → W8A8IEEEBounds.ExponentLE leftCap (a point) ∧
      W8A8IEEEBounds.ExponentLE rightCap (b point))
    (safe : W8A8OrderedDot.admissible leftCap rightCap (W8A8ProjectionTrace.select (mask.map not) ranges).prod = true) :
    ∃ result : Tensor,
      genericSem ctx [leftMap, rightMap, W8A8ProjectionTrace.outputMap mask] iters [left, right]
        [.tensor (W8A8TensorTrace.pack (W8A8ProjectionTrace.select mask ranges) (.float .f32) id initial)]
        (fun c => evalRegion W8A8SourceSemantics.opSem c call env actual) = .ok [.tensor result] ∧
      result.shape = W8A8ProjectionTrace.select mask ranges ∧ result.elt = .float .f32 ∧
      ∀ cell, Idx.InBounds (W8A8ProjectionTrace.select mask ranges) cell → ∃ bits,
        result.get cell = W8A8OrderedBody.value bits ∧
        W8A8IEEEBounds.ExponentLE
          (W8A8OrderedDot.cap leftCap rightCap (W8A8ProjectionTrace.select (mask.map not) ranges).prod) bits := by
  let shape := W8A8ProjectionTrace.select mask ranges
  let coordinate := W8A8IntegerTensor.coordinate shape (W8A8ProjectionTrace.outputMap mask)
  let result := Tensor.canon shape (.float .f32)
    (W8A8TensorTrace.accumulate coordinate (step a b) initial (allPoints ranges))
  have initialTyped : typedState ib shape initial := by
    intro cell bounds
    rw [initialized cell bounds]
    simp [W8A8OrderedBody.value, Check.Val.HasTy, Scalar.conforms]
  refine ⟨result, ?_, rfl, rfl, ?_⟩
  · exact generic_sound geometry typedLeft typedRight accepted call env fallback interpretation a b initial coordinate
      initialTyped leftRead rightRead (fun point bounds => (projected_index lengths bounds).1)
      (fun point bounds => (projected_index lengths bounds).2)
  · intro cell bounds
    have count := projected_count lengths bounds a b
    have cellSafe : W8A8OrderedDot.admissible leftCap rightCap
        (pairs coordinate a b (allPoints ranges) cell).length = true := by
      simpa only [coordinate, shape, count] using safe
    obtain ⟨bits, actual, bounded⟩ := bounded_cell coordinate a b (allPoints ranges) initial cell
      (initialized cell bounds) (fun point member => inputs point ((mem_allPoints _ _).1 member)) cellSafe
    refine ⟨bits, ?_, ?_⟩
    · simpa only [result, shape, Tensor.canon, bounds, ↓reduceIte] using actual
    · simpa only [coordinate, shape, count] using bounded

#print axioms bodyAt_sound
#print axioms generic_sound
#print axioms projected_bounded

end MerlinVerify.Rules.W8A8OrderedTensor
