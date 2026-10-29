/- Explicit reachable signed-index64 correspondence for the mathematical
index summand. Ordinary opaque data/operations have no arithmetic obligation. -/
import Merlin.GraphStatePipelineValueCandidate
import Merlin.GraphStatePipelineLoops

namespace Merlin.GraphState.Pipeline.ValueCarrierCandidate.Reachable
set_option autoImplicit false
open Scalar

theorem progression_member (lower step count value : Nat)
    (member : value ∈ progression lower step count) :
    ∃ ordinal, ordinal < count ∧ value = lower + step * ordinal := by
  induction count generalizing lower with
  | zero =>
    change value ∈ ([] : List Nat) at member
    cases member
  | succ count ih =>
    simp only [progression,List.mem_cons] at member
    cases member with
    | inl first =>
      refine ⟨0,by omega,?_⟩
      simpa only [Nat.mul_zero,Nat.add_zero] using first
    | inr later =>
      obtain ⟨ordinal,inside,equal⟩ := ih (lower+step) later
      refine ⟨ordinal+1,by omega,?_⟩
      calc
        value = (lower+step) + step*ordinal := equal
        _ = lower + step*(ordinal+1) := by
          simp only [Nat.mul_add,Nat.mul_one]
          omega

/-- Both an actual source IV and its SCF successor fit the declared index
width. Membership follows the checked counted range, not a finite trace. -/
theorem source_control {bounds : GraphLoopsUnroll.Bounds} (valid : bounds.Valid)
    {index : Int} (member : index ∈ indexIterations bounds.lower bounds.upper bounds.step) :
    ∃ natural, index = Int.ofNat natural ∧ natural < bounds.upper ∧
      signedRepresentative bounds.bits index ∧
      signedRepresentative bounds.bits (Int.ofNat (natural+bounds.step)) := by
  have counted := valid.2.2.2.2.1
  rw [indexIterations_count valid.2.2.1 counted] at member
  obtain ⟨natural,inside,equalIndex⟩ := List.mem_map.mp member
  obtain ⟨ordinal,ordinalInside,equalNatural⟩ :=
    progression_member bounds.lower bounds.step bounds.count natural inside
  have upper := counted.index_bound ordinalInside
  rw [← equalNatural] at upper
  have fits := GraphLoopsUnroll.Bounds.source_control valid upper
  refine ⟨natural,equalIndex.symm,upper,?_,fits.2⟩
  rw [← equalIndex]
  exact fits.1

/-- Generated lane addition is interpreted mathematically only at these
proved fitted index operands/results. This imposes no unbounded arithmetic law
on an ordinary retained prefix operation or on opaque Payload values. -/
theorem fitted_lane {Value World Error : Type} {registry : List Signature} {controls : Controls}
    (codec : Codec Value) (controlValid : ControlValid registry controls)
    (default : Value) (primitive : Prim → List Value → Value)
    (external : External Value World Error registry) {identity : Nat}
    (found : controlAt controls identity = some .add)
    {bounds : GraphLoopsUnroll.Bounds} (valid : bounds.Valid)
    {group lane : Nat} (groupInside : group < bounds.groups) (laneInside : lane < bounds.factor)
    (world : World) :
    opaqueStep (model codec controlValid default primitive external) identity
      [codec.encode (Int.ofNat (bounds.lower+(bounds.step*bounds.factor)*group)),
       codec.encode (Int.ofNat (bounds.step*lane))] world =
      .ok ([codec.encode (Int.ofNat (bounds.lower+bounds.step*(group*bounds.factor+lane)))],world) ∧
    signedRepresentative bounds.bits (Int.ofNat (bounds.step*lane)) ∧
    signedRepresentative bounds.bits (Int.ofNat (bounds.lower+bounds.step*(group*bounds.factor+lane))) := by
  have meaning := unroll_lane_meaning valid groupInside laneInside
  refine ⟨?_,meaning.2⟩
  have actual := control_on_indices codec controlValid default primitive external found
    [Int.ofNat (bounds.lower+(bounds.step*bounds.factor)*group),Int.ofNat (bounds.step*lane)] world
  simpa only [List.map_cons,List.map_nil,meaning.1] using actual

end Merlin.GraphState.Pipeline.ValueCarrierCandidate.Reachable
