/- Actual structured max/sum execution with typed runtime state. Row membership,
initializer values and numerical bounds are separate obligations to be checked. -/
import MerlinVerify.Rules.W8A8ReductionCells

namespace MerlinVerify.Rules.W8A8ReductionTensor

open MerlinVerify Sem.Ops.Linalg W8A8ReductionBody W8A8ReductionCells

theorem bodyAt_sound {ib : Nat} {actual : Region} {kind : Kind}
    (accepted : Check.PolicyW8A8ReductionBody.check kind actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8ByteModel.extend (W8A8IEEEPrimitives.model fallback))
    (input : RVal) (inputMap outMap : AffineMap) (shape : List Nat)
    (a : Idx → FBits .f32) (coordinate : Idx → Idx) (point : Idx)
    (inputRead : readAt input inputMap point = .ok (W8A8OrderedBody.value (a point)))
    (outputIndex : (evalMap outMap point >>= toIdx shape) = some (coordinate point))
    (outputBounds : Idx.InBounds shape (coordinate point))
    (state : Idx → Val) (typed : W8A8OrderedCells.typedState ib shape state) :
    bodyAt ctx [input] [inputMap] [outMap] [W8A8TensorTrace.pack shape (.float .f32) id state]
      (fun c => evalRegion W8A8SourceSemantics.opSem c call env actual) point =
      .ok [step kind a point (state (coordinate point))] := by
  have outputRead : readAt (.tensor (W8A8TensorTrace.pack shape (.float .f32) id state)) outMap point =
      .ok (state (coordinate point)) := by
    simp only [readAt, W8A8TensorTrace.pack, outputIndex]
    rfl
  have executed := Check.PolicyW8A8ReductionBody.check_sound accepted { ctx with loopIndex := point }
    call env fallback interpretation (a point) (Sem.Ops.Dequant.float32 (state (coordinate point)))
  rw [W8A8OrderedCells.decode_typed (typed _ outputBounds)] at executed
  simp [bodyAt, inputRead, outputRead, executed, step, Check.PolicyW8A8ReductionBody.result]

theorem generic_sound {ib : Nat} {ctx : Ctx} {actual : Region} {kind : Kind} {iters : List IterKind}
    {inputShape shape ranges : List Nat} {input : RVal} {inputMap outMap : AffineMap}
    (geometry : Check.StructuredOk [inputMap, outMap] iters.length
      [.tensor inputShape (.float .f32), .tensor shape (.float .f32)] ranges)
    (typedInput : Check.RVal.HasTy ib input (.tensor inputShape (.float .f32)))
    (accepted : Check.PolicyW8A8ReductionBody.check kind actual = true)
    (call : CallSem) (env : Env) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8ByteModel.extend (W8A8IEEEPrimitives.model fallback))
    (a : Idx → FBits .f32) (initial : Idx → Val) (coordinate : Idx → Idx)
    (initialTyped : W8A8OrderedCells.typedState ib shape initial)
    (inputRead : ∀ point, Idx.InBounds ranges point →
      readAt input inputMap point = .ok (W8A8OrderedBody.value (a point)))
    (outputIndex : ∀ point, Idx.InBounds ranges point →
      (evalMap outMap point >>= toIdx shape) = some (coordinate point))
    (outputBounds : ∀ point, Idx.InBounds ranges point → Idx.InBounds shape (coordinate point)) :
    genericSem ctx [inputMap, outMap] iters [input] [.tensor (W8A8TensorTrace.pack shape (.float .f32) id initial)]
      (fun c => evalRegion W8A8SourceSemantics.opSem c call env actual) =
      .ok [.tensor (Tensor.canon shape (.float .f32)
        (W8A8TensorTrace.accumulate coordinate (step kind a) initial (allPoints ranges)))] := by
  apply W8A8TypedTensorTrace.generic_sound (inTys := [.tensor inputShape (.float .f32)]) geometry
    (.cons typedInput .nil) id initial (by exact ⟨rfl, rfl, initialTyped⟩) (by rfl) _
    coordinate (step kind a) (W8A8OrderedCells.typedState ib shape) initialTyped outputIndex
  · intro point _ state typed
    exact update_typed typed coordinate kind a point
  · intro point bounds state typed
    exact bodyAt_sound accepted ctx call env fallback interpretation input inputMap outMap shape a coordinate point
      (inputRead point bounds) (outputIndex point bounds) (outputBounds point bounds) state typed

/-- Exact initialized result for every projected output cell. The retained list
contains just that cell's reads, in their actual order, and permits empty rows. -/
theorem projected_sound {ib : Nat} {ctx : Ctx} {actual : Region} {kind : Kind} {iters : List IterKind}
    {inputShape ranges : List Nat} {mask : List Bool} {input : RVal} {inputMap : AffineMap}
    (lengths : mask.length = ranges.length)
    (geometry : Check.StructuredOk [inputMap, W8A8ProjectionTrace.outputMap mask] iters.length
      [.tensor inputShape (.float .f32), .tensor (W8A8ProjectionTrace.select mask ranges) (.float .f32)] ranges)
    (typedInput : Check.RVal.HasTy ib input (.tensor inputShape (.float .f32)))
    (accepted : Check.PolicyW8A8ReductionBody.check kind actual = true)
    (call : CallSem) (env : Env) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8ByteModel.extend (W8A8IEEEPrimitives.model fallback))
    (a : Idx → FBits .f32) (initial : Idx → Val) (seeds : Idx → FBits .f32)
    (initialized : ∀ cell, Idx.InBounds (W8A8ProjectionTrace.select mask ranges) cell →
      initial cell = W8A8OrderedBody.value (seeds cell))
    (inputRead : ∀ point, Idx.InBounds ranges point →
      readAt input inputMap point = .ok (W8A8OrderedBody.value (a point))) :
    ∃ result : Tensor,
      genericSem ctx [inputMap, W8A8ProjectionTrace.outputMap mask] iters [input]
        [.tensor (W8A8TensorTrace.pack (W8A8ProjectionTrace.select mask ranges) (.float .f32) id initial)]
        (fun c => evalRegion W8A8SourceSemantics.opSem c call env actual) = .ok [.tensor result] ∧
      result.shape = W8A8ProjectionTrace.select mask ranges ∧ result.elt = .float .f32 ∧
      ∀ cell, Idx.InBounds (W8A8ProjectionTrace.select mask ranges) cell →
        result.get cell = W8A8OrderedBody.value
          (reduce kind (words (W8A8IntegerTensor.coordinate (W8A8ProjectionTrace.select mask ranges)
            (W8A8ProjectionTrace.outputMap mask)) a (allPoints ranges) cell) (seeds cell)) := by
  let shape := W8A8ProjectionTrace.select mask ranges
  let coordinate := W8A8IntegerTensor.coordinate shape (W8A8ProjectionTrace.outputMap mask)
  let result := Tensor.canon shape (.float .f32)
    (W8A8TensorTrace.accumulate coordinate (step kind a) initial (allPoints ranges))
  have initialTyped : W8A8OrderedCells.typedState ib shape initial := by
    intro cell bounds
    rw [initialized cell bounds]
    simp [W8A8OrderedBody.value, Check.Val.HasTy, Scalar.conforms]
  refine ⟨result, ?_, rfl, rfl, ?_⟩
  · exact generic_sound geometry typedInput accepted call env fallback interpretation a initial coordinate
      initialTyped inputRead (fun point bounds => (W8A8OrderedCells.projected_index lengths bounds).1)
      (fun point bounds => (W8A8OrderedCells.projected_index lengths bounds).2)
  · intro cell bounds
    simpa only [result, shape, Tensor.canon, bounds, ↓reduceIte] using
      initialized_cell kind coordinate a (allPoints ranges) initial cell (seeds cell) (initialized cell bounds)

end MerlinVerify.Rules.W8A8ReductionTensor
