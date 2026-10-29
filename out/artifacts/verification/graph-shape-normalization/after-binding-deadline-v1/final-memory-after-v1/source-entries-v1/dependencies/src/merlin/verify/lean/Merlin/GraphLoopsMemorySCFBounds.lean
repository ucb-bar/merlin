/- Signed control and allocation-relative byte safety from the ACTUAL checked
boundary. These facts do not establish physical base/capacity or scalar fidelity. -/
import Merlin.GraphLoopsMemorySCFWitness

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.Model

theorem checked_outer_control {p : Projection} (accepted : check p = true) {boundary : Boundary}
    (member : boundary ∈ p.boundaries) {outer : Nat} (inside : outer < boundary.segment.bounds.upper) :
    GraphLoopsSegment.FitsSigned p.reuse.bits outer ∧
    GraphLoopsSegment.FitsSigned p.reuse.bits (outer + boundary.segment.bounds.step * boundary.segment.bounds.tile) ∧
    GraphLoopsSegment.FitsSigned p.reuse.bits (min (outer + boundary.segment.bounds.step * boundary.segment.bounds.tile) boundary.segment.bounds.upper) := by
  have valid := (check_sound accepted).1.2.2.2.1 boundary member
  have bounds := valid.2.2.2.1.1
  have bits := valid.2.2.2.2.2.1
  rw [← bits]
  exact GraphLoopsSegment.Bounds.outer_control bounds inside

theorem checked_inner_control {p : Projection} (accepted : check p = true) {boundary : Boundary}
    (member : boundary ∈ p.boundaries) {outer inner : Nat}
    (inside : inner < min (outer + boundary.segment.bounds.step * boundary.segment.bounds.tile) boundary.segment.bounds.upper) :
    GraphLoopsSegment.FitsSigned p.reuse.bits inner ∧
    GraphLoopsSegment.FitsSigned p.reuse.bits (inner + boundary.segment.bounds.step) := by
  have valid := (check_sound accepted).1.2.2.2.1 boundary member
  rw [← valid.2.2.2.2.2.1]
  exact GraphLoopsSegment.Bounds.inner_control valid.2.2.2.1.1 inside

/-- Every admitted initializer store coordinate has a fitting relative byte
offset. No actual allocator base address or host ABI fact is invented. -/
theorem checked_initializer_byte_offset {p : Projection} (accepted : check p = true) {boundary : Boundary}
    (member : boundary ∈ p.boundaries) {coordinates : List Nat}
    (inside : GraphLoopsReuse.Geometry.Coordinate boundary.config.program.shape coordinates) :
    GraphLoopsReuse.Geometry.flatten boundary.config.program.shape coordinates *
      ((GraphLoopsReuse.objectAt p.reuse.objects boundary.config.object).elementBits / 8) < 2 ^ (p.reuse.bits - 1) := by
  have configValid := ((check_sound accepted).1.2.2.2.1 boundary member).1
  have objectMember : GraphLoopsReuse.objectAt p.reuse.objects boundary.config.object ∈ p.reuse.objects := by
    unfold GraphLoopsReuse.objectAt
    rw [List.getElem?_eq_getElem configValid.1]
    exact List.mem_of_getElem rfl
  have objectValid := (check_sound accepted).2.1.2.1 _ objectMember
  rw [configValid.2.2.1] at inside ⊢
  exact GraphLoopsReuse.Model.ValidObject.byte_offset objectValid inside

end Merlin.GraphLoopsMemorySCF.Model
