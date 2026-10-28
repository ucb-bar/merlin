import MerlinVerify.Rules.QuantizeRuntime

namespace MerlinVerify.Rules.QuantizeGeometry

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg

/-- Structured access checks depend on runtime shapes, independently of element widths. -/
theorem retype {maps : List AffineMap} {n : Nat} {before after : List VTy} {ranges : List Nat}
    (geometry : StructuredOk maps n before ranges)
    (shapes : before.map VTy.shape = after.map VTy.shape) : StructuredOk maps n after ranges := by
  have lengths : before.length = after.length := by
    simpa only [List.length_map] using congrArg List.length shapes
  refine ⟨geometry.len.trans lengths, geometry.dims, ?_, ?_⟩
  · rw [← shapes]
    exact geometry.loops
  · have mapped : List.Forall₂ (fun m shape => mapInBounds ranges m shape = true)
        maps (before.map VTy.shape) := by
      simpa only [List.forall₂_map_right_iff] using geometry.inBounds
    rw [shapes] at mapped
    simpa only [List.forall₂_map_right_iff] using mapped

end MerlinVerify.Rules.QuantizeGeometry
