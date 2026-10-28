/- Exact per-cell traces for actual max/sum reductions. Filtering preserves loop
order; poison outside the accessed domain is not replaced with invented float bits. -/
import MerlinVerify.Check.PolicyW8A8ReductionBody

namespace MerlinVerify.Rules.W8A8ReductionCells

open MerlinVerify Sem.Ops.Linalg W8A8ReductionBody

def step (kind : Kind) (input : Idx → FBits .f32) (point : Idx) (accumulator : Val) : Val :=
  W8A8OrderedBody.accumulatorValue ((Sem.Ops.Dequant.float32 accumulator).map fun current =>
    combine kind (input point) current)

def words (coordinate : Idx → Idx) (input : Idx → FBits .f32) (points : List Idx) (cell : Idx) :
    List (FBits .f32) :=
  (points.filter (fun point => decide (cell = coordinate point))).map input

def reduce (kind : Kind) (input : List (FBits .f32)) (initial : FBits .f32) : FBits .f32 :=
  input.foldl (fun accumulator item => combine kind item accumulator) initial

theorem step_typed (ib : Nat) (kind : Kind) (input : Idx → FBits .f32) (point : Idx) (accumulator : Val) :
    Check.Val.HasTy ib (.float .f32) (step kind input point accumulator) := by
  cases decoded : Sem.Ops.Dequant.float32 accumulator <;>
    simp [step, decoded, W8A8OrderedBody.accumulatorValue, W8A8OrderedBody.value,
      Check.Val.HasTy, Scalar.conforms]

theorem update_typed {ib : Nat} {shape : List Nat} {state : Idx → Val}
    (typed : W8A8OrderedCells.typedState ib shape state) (coordinate : Idx → Idx)
    (kind : Kind) (input : Idx → FBits .f32) (point : Idx) :
    W8A8OrderedCells.typedState ib shape (W8A8TensorTrace.update coordinate (step kind input) state point) := by
  intro cell bounds
  unfold W8A8TensorTrace.update
  split
  · exact step_typed ib kind input point _
  · exact typed cell bounds

private theorem initialized_fold (kind : Kind) (input : Idx → FBits .f32) (points : List Idx)
    (initial : FBits .f32) :
    points.foldl (fun accumulator point => step kind input point accumulator) (W8A8OrderedBody.value initial) =
      W8A8OrderedBody.value (reduce kind (points.map input) initial) := by
  induction points generalizing initial with
  | nil => rfl
  | cons point rest induction =>
    simpa only [List.foldl_cons, step, W8A8OrderedBody.value, Sem.Ops.Dequant.float32,
      Option.map_some, W8A8OrderedBody.accumulatorValue, List.map_cons, reduce] using
      induction (combine kind (input point) initial)

theorem initialized_cell (kind : Kind) (coordinate : Idx → Idx) (input : Idx → FBits .f32)
    (points : List Idx) (initial : Idx → Val) (cell : Idx) (bits : FBits .f32)
    (initialized : initial cell = W8A8OrderedBody.value bits) :
    W8A8TensorTrace.accumulate coordinate (step kind input) initial points cell =
      W8A8OrderedBody.value (reduce kind (words coordinate input points cell) bits) := by
  rw [W8A8TensorTrace.accumulate_cell, initialized]
  simpa only [words, List.foldl_filter, decide_eq_true_eq] using initialized_fold kind input
    (points.filter (fun point => decide (cell = coordinate point))) bits

theorem projected_count {mask : List Bool} {ranges cell : List Nat}
    (lengths : mask.length = ranges.length) (bounds : Idx.InBounds (W8A8ProjectionTrace.select mask ranges) cell)
    (input : Idx → FBits .f32) :
    (words (W8A8IntegerTensor.coordinate (W8A8ProjectionTrace.select mask ranges)
      (W8A8ProjectionTrace.outputMap mask)) input (allPoints ranges) cell).length =
      (W8A8ProjectionTrace.select (mask.map not) ranges).prod := by
  simpa only [words, W8A8OrderedCells.pairs, List.length_map] using
    W8A8OrderedCells.projected_count lengths bounds input input

end MerlinVerify.Rules.W8A8ReductionCells
