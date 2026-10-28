/- Numerical row bounds for actual per-cell reduction traces. Cross-operation
associations and an actual finite row witness remain explicit obligations. -/
import MerlinVerify.Rules.W8A8ReductionOrder
import MerlinVerify.Rules.W8A8ReductionTensor

namespace MerlinVerify.Rules.W8A8ReductionRows

open MerlinVerify W8A8ReductionCells W8A8MaskedMaximum

theorem words_all (coordinate : Idx → Idx) (input : Idx → FBits .f32) (points : List Idx) (cell : Idx)
    (property : FBits .f32 → Prop) (members : ∀ point ∈ points, property (input point)) :
    ∀ word ∈ words coordinate input points cell, property word := by
  intro word member
  obtain ⟨point, selected, rfl⟩ := List.mem_map.mp member
  exact members point (List.mem_filter.mp selected).1

theorem words_witness {cap : Nat} (coordinate : Idx → Idx) (input : Idx → FBits .f32)
    (points : List Idx) (cell : Idx)
    (unmasked : ∃ point ∈ points, cell = coordinate point ∧ W8A8IEEEBounds.ExponentLE cap (input point)) :
    ∃ word ∈ words coordinate input points cell, W8A8IEEEBounds.ExponentLE cap word := by
  obtain ⟨point, member, same, finite⟩ := unmasked
  exact ⟨input point, List.mem_map.mpr ⟨point, List.mem_filter.mpr ⟨member, by simpa using same⟩, rfl⟩, finite⟩

/-- The exact runtime accumulator is an attained finite member of its actual row.
Neither the initializer nor a finite-looking shape can replace that row witness. -/
theorem maximum_cell {cap : Nat} (coordinate : Idx → Idx) (input : Idx → FBits .f32)
    (points : List Idx) (initial : Idx → Val) (cell : Idx)
    (initialized : initial cell = W8A8OrderedBody.value negativeInfinity)
    (members : ∀ point ∈ points, Admissible cap (input point))
    (unmasked : ∃ point ∈ points, cell = coordinate point ∧ W8A8IEEEBounds.ExponentLE cap (input point)) :
    ∃ bits ∈ words coordinate input points cell,
      W8A8TensorTrace.accumulate coordinate (step .maximum input) initial points cell = W8A8OrderedBody.value bits ∧
      W8A8IEEEBounds.ExponentLE cap bits ∧
      bits = W8A8SoftmaxRow.rowMaximum (words coordinate input points cell) := by
  have rowMembers := words_all coordinate input points cell (Admissible cap) members
  have rowWitness := words_witness coordinate input points cell unmasked
  have exactTrace := initialized_cell .maximum coordinate input points initial cell negativeInfinity initialized
  rw [W8A8ReductionOrder.maximum_fold _ _ (Or.inl rfl) rowMembers] at exactTrace
  obtain ⟨bits, present, same, bounded⟩ := row_maximum_attained _ rowMembers rowWitness
  exact ⟨bits, present, by simpa only [same] using exactTrace, bounded, same.symm⟩

/-- An actual sum trace whose read list is the corresponding shifted exponentials
has precisely the proved denominator. The association must come from checked SSA/maps. -/
theorem sum_exponentials_cell {cap : Nat} (coordinate : Idx → Idx) (input : Idx → FBits .f32)
    (points : List Idx) (initial : Idx → Val) (cell : Idx) (scores : List (FBits .f32))
    (initialized : initial cell = W8A8OrderedBody.value 0#32)
    (association : words coordinate input points cell = W8A8SoftmaxRow.exponentials scores)
    (safe : cap ≤ 252) (members : ∀ score ∈ scores, Admissible cap score)
    (unmasked : ∃ score ∈ scores, W8A8IEEEBounds.ExponentLE cap score) (count : scores.length ≤ 2^22) :
    W8A8TensorTrace.accumulate coordinate (step .sum input) initial points cell =
        W8A8OrderedBody.value (W8A8SoftmaxRow.denominator scores) ∧
      W8A8IEEEBounds.ExponentLE (W8A8OrderedMagnitude.sumCap 149 scores.length) (W8A8SoftmaxRow.denominator scores) ∧
      1 ≤ W8A8RoundedReal.value (W8A8SoftmaxRow.denominator scores) := by
  have exactTrace := initialized_cell .sum coordinate input points initial cell 0#32 initialized
  rw [association] at exactTrace
  have folded := W8A8ReductionOrder.sum_fold (W8A8SoftmaxRow.exponentials scores) (0 : FBits .f32)
  exact ⟨exactTrace.trans (congrArg W8A8OrderedBody.value folded),
    W8A8SoftmaxRow.denominator_finite_lower scores safe members unmasked count⟩

end MerlinVerify.Rules.W8A8ReductionRows
