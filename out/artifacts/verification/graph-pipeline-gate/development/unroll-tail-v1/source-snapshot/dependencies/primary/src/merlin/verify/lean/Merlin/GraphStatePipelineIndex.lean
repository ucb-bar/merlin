/- Fixed concrete INDEX meanings for the scalar/SCF graph domain.
Index values are mathematical signed representatives. Accepted W bounds prove
that generated operations have no fixed-width overflow and minui compares
nonnegative representatives. This is not a backend/runtime fidelity theorem. -/
import Merlin.GraphStatePipelineControl
import Merlin.GraphLoopsUnrollBounds

namespace Merlin.GraphState.Pipeline
set_option autoImplicit false
open Scalar GraphLoopsSegment

inductive IndexOp where
  | constant (value : Int)
  | add
  | minui
  deriving DecidableEq, BEq, ReflBEq, LawfulBEq, Repr

def IndexOp.inputs : IndexOp → List Ty
  | .constant _ => []
  | .add | .minui => [.index,.index]

def IndexOp.eval : IndexOp → List Int → Int
  | .constant value,_ => value
  | .add,values => values[0]?.getD 0 + values[1]?.getD 0
  | .minui,values => min (values[0]?.getD 0) (values[1]?.getD 0)

/-- A single common registry. Control classification must match the exact
supported operation/type descriptor projection; other calls remain partial. -/
abbrev Controls := List (Option IndexOp)

def controlAt (controls : Controls) (identity : Nat) : Option IndexOp :=
  (controls[identity]?).getD none

def ControlValid (registry : List Signature) (controls : Controls) : Prop :=
  controls.length = registry.length ∧ ∀ identity op, controlAt controls identity = some op →
    identity < registry.length ∧ (signature registry identity).inputs = op.inputs ∧
      (signature registry identity).outputs = [.index]

/-- Executable finite signature check; its soundness is kernel proved below. -/
def checkControl (registry : List Signature) (controls : Controls) : Bool :=
  decide (controls.length = registry.length) && (List.range controls.length).all (fun identity =>
    match controlAt controls identity with
    | none => true
    | some op => decide ((signature registry identity).inputs = op.inputs ∧
        (signature registry identity).outputs = [.index]))

theorem checkControl_sound {registry : List Signature} {controls : Controls}
    (accepted : checkControl registry controls = true) : ControlValid registry controls := by
  simp only [checkControl, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at accepted
  refine ⟨accepted.1, ?_⟩
  intro identity op found
  have inside : identity < controls.length := by
    by_cases inside : identity < controls.length
    · exact inside
    · have absent : controls[identity]? = none := List.getElem?_eq_none (by omega)
      simp only [controlAt, absent, Option.getD_none] at found
      cases found
  have checked := accepted.2 identity (List.mem_range.mpr inside)
  rw [found] at checked
  exact ⟨by omega, of_decide_eq_true checked⟩

/-- The only external premise is successful result arity for the retained
noncontrol call meaning. It does not assert success, purity, totality or no poison. -/
structure External (World Error : Type) (registry : List Signature) where
  run : Nat → List Int → World → Except (Error × World) (List Int × World)
  arity : ∀ identity arguments world values final,
    run identity arguments world = .ok (values,final) →
      values.length = (signature registry identity).outputs.length

def indexEffect {World Error : Type} {registry : List Signature} (controls : Controls)
    (external : External World Error registry) (identity : Nat) (arguments : List Int) (world : World) :
    Except (Error × World) (List Int × World) :=
  match controlAt controls identity with
  | some op => .ok ([op.eval arguments],world)
  | none => external.run identity arguments world

/-- The unchanged Scalar evaluator specialized to ONE concrete index/control
interpretation and an explicitly total pure common word-primitive function. -/
def indexModel {World Error : Type} {registry : List Signature} {controls : Controls}
    (valid : ControlValid registry controls) (primitive : Prim → List Int → Int)
    (external : External World Error registry) : Model Int World Error registry where
  default := 0
  primitive := primitive
  index := id
  iterations := indexIterations
  effect := indexEffect controls external
  arity := by
    intro identity arguments world values final success
    cases found : controlAt controls identity with
    | none => exact external.arity _ _ _ _ _ (by simpa only [indexEffect,found] using success)
    | some op =>
      have same : ([op.eval arguments],world) = (values,final) := by
        simpa only [indexEffect,found,Except.ok.injEq] using success
      cases same
      rw [(valid.2 identity op found).2.2]
      rfl

theorem control_effect {World Error : Type} {registry : List Signature} {controls : Controls}
    (valid : ControlValid registry controls) (primitive : Prim → List Int → Int)
    (external : External World Error registry) {identity : Nat} {op : IndexOp}
    (found : controlAt controls identity = some op) (arguments : List Int) (world : World) :
    opaqueStep (indexModel valid primitive external) identity arguments world = .ok ([op.eval arguments],world) := by
  simp only [opaqueStep,indexModel,indexEffect,found]

/-- Fixed-width representative relation used by the control bridge. -/
def signedRepresentative (bits : Nat) (value : Int) : Prop :=
  -((2^(bits-1) : Nat) : Int) ≤ value ∧ value < ((2^(bits-1) : Nat) : Int)

theorem signed_from_W {bits value : Nat} (fits : FitsSigned bits value) :
    signedRepresentative bits (Int.ofNat value) := fits

/-- The actual unsigned minimum agrees with the nonnegative natural minimum
on the proved control domain; negative inputs are not admitted by this bridge. -/
theorem minui_natural (left right : Nat) :
    IndexOp.eval .minui [Int.ofNat left,Int.ofNat right] = Int.ofNat (min left right) := by
  simp only [IndexOp.eval, List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
  by_cases ordered : left ≤ right
  · rw [Nat.min_eq_left ordered]
    exact Int.min_eq_left (Int.ofNat_le.mpr ordered)
  · rw [Nat.min_eq_right (by omega)]
    exact Int.min_eq_right (Int.ofNat_le.mpr (by omega))

theorem unroll_lane_meaning {b : GraphLoopsUnroll.Bounds} (valid : b.Valid)
    {group lane : Nat} (groupInside : group < b.groups) (laneInside : lane < b.factor) :
    IndexOp.eval .add [Int.ofNat (b.lower+(b.step*b.factor)*group),Int.ofNat (b.step*lane)] =
      Int.ofNat (b.lower+b.step*(group*b.factor+lane)) ∧
    signedRepresentative b.bits (Int.ofNat (b.step*lane)) ∧
    signedRepresentative b.bits (Int.ofNat (b.lower+b.step*(group*b.factor+lane))) := by
  have derived := GraphLoopsUnroll.Bounds.lane_control valid groupInside laneInside
  refine ⟨?_, derived.2.1, ?_⟩
  · simp only [IndexOp.eval, List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    exact congrArg Int.ofNat derived.1
  · rw [← derived.1]
    exact derived.2.2

theorem segment_control_meaning {b : GraphLoopsSegment.Bounds} (valid : b.Valid)
    {outer : Nat} (inside : outer < b.upper) :
    IndexOp.eval .add [Int.ofNat outer,Int.ofNat (b.step*b.tile)] = Int.ofNat (outer+b.step*b.tile) ∧
    IndexOp.eval .minui [Int.ofNat (outer+b.step*b.tile),Int.ofNat b.upper] =
      Int.ofNat (min (outer+b.step*b.tile) b.upper) ∧
    signedRepresentative b.bits (Int.ofNat (outer+b.step*b.tile)) ∧
    signedRepresentative b.bits (Int.ofNat (min (outer+b.step*b.tile) b.upper)) := by
  exact ⟨by simp [IndexOp.eval], minui_natural _ _,
    (GraphLoopsSegment.Bounds.outer_control valid inside).2.1,
    (GraphLoopsSegment.Bounds.outer_control valid inside).2.2⟩

end Merlin.GraphState.Pipeline
