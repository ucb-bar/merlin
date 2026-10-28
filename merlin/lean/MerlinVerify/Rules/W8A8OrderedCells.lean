/- Per-cell ordered floating traces. Unobserved cells may contain arbitrary runtime values;
the initialized selected trace is recovered exactly before applying its numerical bound. -/
import MerlinVerify.Rules.W8A8OrderedBody
import MerlinVerify.Rules.W8A8TypedTensorTrace
import MerlinVerify.Rules.W8A8ProjectionTrace

namespace MerlinVerify.Rules.W8A8OrderedCells

open MerlinVerify Sem.Ops.Linalg

def step (left right : Idx → FBits .f32) (point : Idx) (accumulator : Val) : Val :=
  W8A8OrderedBody.accumulatorValue ((Sem.Ops.Dequant.float32 accumulator).map fun z =>
    W8A8OrderedDot.step z (left point, right point))

def typedState (ib : Nat) (shape : List Nat) (state : Idx → Val) : Prop :=
  ∀ cell, Idx.InBounds shape cell → Check.Val.HasTy ib (.float .f32) (state cell)

theorem decode_typed {ib : Nat} {value : Val} (typed : Check.Val.HasTy ib (.float .f32) value) :
    W8A8OrderedBody.accumulatorValue (Sem.Ops.Dequant.float32 value) = value := by
  cases value with
  | poison => rfl
  | some scalar =>
    cases scalar with
    | bits width bits => simp [Check.Val.HasTy, Scalar.conforms] at typed
    | flt kind bits =>
      cases kind <;> simp_all [Check.Val.HasTy, Scalar.conforms, Sem.Ops.Dequant.float32,
        W8A8OrderedBody.accumulatorValue, W8A8OrderedBody.value]

theorem step_typed (ib : Nat) (left right : Idx → FBits .f32) (point : Idx) (accumulator : Val) :
    Check.Val.HasTy ib (.float .f32) (step left right point accumulator) := by
  cases decoded : Sem.Ops.Dequant.float32 accumulator <;>
    simp [step, decoded, W8A8OrderedBody.accumulatorValue, W8A8OrderedBody.value,
      Check.Val.HasTy, Scalar.conforms]

theorem update_typed {ib : Nat} {shape : List Nat} {state : Idx → Val}
    (typed : typedState ib shape state) (coordinate : Idx → Idx)
    (left right : Idx → FBits .f32) (point : Idx) :
    typedState ib shape (W8A8TensorTrace.update coordinate (step left right) state point) := by
  intro cell bounds
  unfold W8A8TensorTrace.update
  split
  · exact step_typed ib left right point _
  · exact typed cell bounds

theorem step_value (left right : Idx → FBits .f32) (point : Idx) (accumulator : FBits .f32) :
    step left right point (W8A8OrderedBody.value accumulator) =
      W8A8OrderedBody.value (W8A8OrderedDot.step accumulator (left point, right point)) := rfl

def pairs (coordinate : Idx → Idx) (left right : Idx → FBits .f32)
    (points : List Idx) (cell : Idx) : List W8A8OrderedDot.Pair :=
  (points.filter (fun point => decide (cell = coordinate point))).map fun point => (left point, right point)

private theorem initialized_fold (left right : Idx → FBits .f32) (points : List Idx)
    (initial : FBits .f32) :
    points.foldl (fun accumulator point => step left right point accumulator) (W8A8OrderedBody.value initial) =
      W8A8OrderedBody.value (W8A8OrderedDot.evaluate (points.map fun point => (left point, right point)) initial) := by
  induction points generalizing initial with
  | nil => rfl
  | cons point rest ih =>
    simpa only [List.foldl_cons, step_value, List.map_cons, W8A8OrderedDot.evaluate] using
      ih (W8A8OrderedDot.step initial (left point, right point))

theorem initialized_cell (coordinate : Idx → Idx) (left right : Idx → FBits .f32)
    (points : List Idx) (initial : Idx → Val) (cell : Idx) (bits : FBits .f32)
    (initialized : initial cell = W8A8OrderedBody.value bits) :
    W8A8TensorTrace.accumulate coordinate (step left right) initial points cell =
      W8A8OrderedBody.value (W8A8OrderedDot.evaluate (pairs coordinate left right points cell) bits) := by
  rw [W8A8TensorTrace.accumulate_cell, initialized]
  simpa only [pairs, List.foldl_filter, decide_eq_true_eq] using initialized_fold left right
    (points.filter (fun point => decide (cell = coordinate point))) bits

theorem bounded_cell {leftCap rightCap : Nat} (coordinate : Idx → Idx)
    (left right : Idx → FBits .f32) (points : List Idx) (initial : Idx → Val) (cell : Idx)
    (initialized : initial cell = W8A8OrderedBody.value 0#32)
    (inputs : ∀ point ∈ points, W8A8IEEEBounds.ExponentLE leftCap (left point) ∧
      W8A8IEEEBounds.ExponentLE rightCap (right point))
    (safe : W8A8OrderedDot.admissible leftCap rightCap (pairs coordinate left right points cell).length = true) :
    ∃ bits, W8A8TensorTrace.accumulate coordinate (step left right) initial points cell = W8A8OrderedBody.value bits ∧
      W8A8IEEEBounds.ExponentLE (W8A8OrderedDot.cap leftCap rightCap (pairs coordinate left right points cell).length) bits := by
  have selected : ∀ pair ∈ pairs coordinate left right points cell,
      W8A8IEEEBounds.ExponentLE leftCap pair.1 ∧ W8A8IEEEBounds.ExponentLE rightCap pair.2 := by
    intro pair member
    obtain ⟨point, found, same⟩ := List.mem_map.1 member
    subst pair
    exact inputs point (List.mem_filter.1 found).1
  exact ⟨_, initialized_cell coordinate left right points initial cell 0#32 initialized,
    W8A8OrderedDot.evaluate_bounded selected safe⟩

theorem projected_count {mask : List Bool} {ranges cell : List Nat}
    (lengths : mask.length = ranges.length) (bounds : Idx.InBounds (W8A8ProjectionTrace.select mask ranges) cell)
    (left right : Idx → FBits .f32) :
    (pairs (W8A8IntegerTensor.coordinate (W8A8ProjectionTrace.select mask ranges) (W8A8ProjectionTrace.outputMap mask))
      left right (allPoints ranges) cell).length = (W8A8ProjectionTrace.select (mask.map not) ranges).prod := by
  have same : (allPoints ranges).filter (fun point => decide
      (cell = W8A8IntegerTensor.coordinate (W8A8ProjectionTrace.select mask ranges)
        (W8A8ProjectionTrace.outputMap mask) point)) =
      (allPoints ranges).filter (fun point => decide (W8A8ProjectionTrace.select mask point = cell)) := by
    apply List.filter_congr
    intro point member
    rw [W8A8ProjectionTrace.output_coordinate lengths ((mem_allPoints _ _).1 member)]
    simp [eq_comm]
  simpa only [pairs, same, List.length_map] using W8A8ProjectionTrace.selected_count lengths bounds

theorem projected_index {mask : List Bool} {ranges point : List Nat}
    (lengths : mask.length = ranges.length) (bounds : Idx.InBounds ranges point) :
    (evalMap (W8A8ProjectionTrace.outputMap mask) point >>= toIdx (W8A8ProjectionTrace.select mask ranges)) =
        some (W8A8IntegerTensor.coordinate (W8A8ProjectionTrace.select mask ranges)
          (W8A8ProjectionTrace.outputMap mask) point) ∧
      Idx.InBounds (W8A8ProjectionTrace.select mask ranges)
        (W8A8IntegerTensor.coordinate (W8A8ProjectionTrace.select mask ranges)
          (W8A8ProjectionTrace.outputMap mask) point) := by
  rw [W8A8ProjectionTrace.output_coordinate lengths bounds]
  have projected := W8A8ProjectionTrace.select_inBounds lengths bounds
  refine ⟨?_, projected⟩
  have pointLength : mask.length = point.length := lengths.trans bounds.1.symm
  simp only [evalMap, W8A8ProjectionTrace.outputMap, W8A8ProjectionTrace.eval_dimensions pointLength,
    Option.bind_eq_bind, Option.bind_some, toIdx_ofNat projected]

#print axioms decode_typed
#print axioms initialized_cell
#print axioms bounded_cell
#print axioms projected_count
#print axioms projected_index

end MerlinVerify.Rules.W8A8OrderedCells
