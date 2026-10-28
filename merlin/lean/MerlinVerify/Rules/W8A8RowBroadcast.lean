/- Coordinate and value correspondence for the actual trailing-unit broadcast.
The reshaped row maximum/denominator is read from the same source row. -/
import MerlinVerify.Rules.W8A8AttentionTemplate
import MerlinVerify.Rules.W8A8RowReshape
import MerlinVerify.Rules.W8A8TensorOperands

namespace MerlinVerify.Rules.W8A8RowBroadcast

open MerlinVerify Sem.Ops.Linalg

theorem leading_inBounds {shape point : List Nat} {width : Nat}
    (bounds : Idx.InBounds (shape ++ [width]) point) : Idx.InBounds shape (point.take shape.length) := by
  have lengths : (W8A8RowTrace.mask shape.length).length = (shape ++ [width]).length := by simp [W8A8RowTrace.mask]
  have selected := W8A8ProjectionTrace.select_inBounds lengths bounds
  simpa only [W8A8RowTrace.select_mask, List.take_left] using selected

theorem unit_point {shape point : List Nat} (bounds : Idx.InBounds (shape ++ [1]) point) :
    point = point.take shape.length ++ [0] := by
  have rowBounds := leading_inBounds bounds
  have selected : point ∈ (List.range 1).map (fun column => point.take shape.length ++ [column]) := by
    rw [← W8A8RowTrace.last_axis_points shape 1 _ rowBounds]
    exact List.mem_filter.mpr ⟨(mem_allPoints _ _).mpr bounds, by simp only [decide_true]⟩
  simpa using selected

theorem coordinate {shape point : List Nat} {width : Nat}
    (bounds : Idx.InBounds (shape ++ [width]) point) :
    W8A8TensorOperands.coordinate (shape ++ [1]) (W8A8AttentionTemplate.broadcastMap shape.length) point =
      point.take shape.length ++ [0] := by
  have pointLength : (W8A8RowTrace.mask shape.length).length = point.length := by
    simpa [W8A8RowTrace.mask] using bounds.1.symm
  have projected := W8A8ProjectionTrace.eval_dimensions pointLength
  rw [W8A8RowTrace.dimensions_mask, W8A8RowTrace.select_mask] at projected
  have rowBounds := leading_inBounds bounds
  have unitBounds : Idx.InBounds (shape ++ [1]) (point.take shape.length ++ [0]) :=
    W8A8RowReshape.append_inBounds rowBounds (by decide)
  have mapped : evalMap (W8A8AttentionTemplate.broadcastMap shape.length) point =
      some ((point.take shape.length ++ [0]).map Int.ofNat) := by
    simp only [evalMap, W8A8AttentionTemplate.broadcastMap, List.mapM_append, projected,
      List.mapM_cons, evalAffine, List.mapM_nil, Option.bind_some, Option.pure_def,
      List.map_append, List.map_cons, List.map_nil]
    rfl
  simp only [W8A8TensorOperands.coordinate, mapped, Option.bind_eq_bind, Option.bind_some,
    toIdx_ofNat unitBounds, Option.getD_some]

/-- No arbitrary association of row tensors is supplied: both reshapes start at
this tensor, and the actual broadcast map selects its matching row. -/
theorem reshaped_read (tensor : Tensor) (flat : List Nat) (point : Idx) (width : Nat)
    (size : shapeSize flat = shapeSize tensor.shape)
    (bounds : Idx.InBounds (tensor.shape ++ [width]) point) :
    (ReshapeIndex.reshape (ReshapeIndex.reshape tensor flat tensor.elt)
      (tensor.shape ++ [1]) tensor.elt).get
      (W8A8TensorOperands.coordinate (tensor.shape ++ [1])
        (W8A8AttentionTemplate.broadcastMap tensor.shape.length) point) =
      tensor.get (point.take tensor.shape.length) := by
  rw [coordinate bounds]
  exact W8A8RowReshape.collapsed_then_unit_get tensor flat _ size (leading_inBounds bounds)

end MerlinVerify.Rules.W8A8RowBroadcast
