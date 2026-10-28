/- Numerical facts describe actual value bits. They are retained only by rules proving their
preservation. In particular, initialized floating arithmetic need not preserve finiteness. -/
import MerlinVerify.Check.DefinedRefinements
import MerlinVerify.Check.PolicyW8A8FiniteDomain

namespace MerlinVerify.Check.FiniteFacts

open MerlinVerify MerlinVerify.Sem.Ops Rules

inductive Class where
  | any | finite | normal
  deriving DecidableEq, Repr

def pointCheck : Class → Val → Bool
  | .any, _ => true
  | .finite, value => PolicyW8A8FiniteDomain.pointCheck value
  | .normal, value => match Dequant.float32 value with
    | some bits => W8A8FiniteLaws.scaleCheck bits
    | none => false

def Holds : Class → RVal → Prop
  | .any, _ => True
  | cls, .scalar value => pointCheck cls value = true
  | cls, .tensor value => ∀ point, Idx.InBounds value.shape point → pointCheck cls (value.get point) = true

def model : Definedness.Refined.Model Class where
  holds := Holds
  empty := .any
  empty_sound _ := trivial

def classify (value : Val) : Class :=
  if pointCheck .normal value then .normal
  else if pointCheck .finite value then .finite else .any

theorem classify_sound (value : Val) : Holds (classify value) (.scalar value) := by
  unfold classify
  split
  · assumption
  · split
    · assumption
    · trivial

theorem point_finite {cls : Class} {value : Val} (allowed : cls = .finite ∨ cls = .normal)
    (holds : pointCheck cls value = true) : PolicyW8A8FiniteDomain.pointCheck value = true := by
  rcases allowed with rfl | rfl
  · exact holds
  · unfold pointCheck at holds
    cases decoded : Dequant.float32 value with
    | none => simp [decoded] at holds
    | some bits =>
      have admissible := W8A8FiniteLaws.scaleCheck_admissible (by simpa [decoded] using holds)
      simp only [W8A8SourceQuantize.scaleAdmissible, Bool.and_eq_true, decide_eq_true_eq] at admissible
      simp [PolicyW8A8FiniteDomain.pointCheck, decoded, W8A8FiniteLaws.Finite, admissible.2]

theorem tensor_finite {cls : Class} {input : Tensor} (allowed : cls = .finite ∨ cls = .normal)
    (holds : Holds cls (.tensor input)) :
    PolicyW8A8FiniteDomain.pointsCheck input.shape input = true := by
  apply List.all_eq_true.2
  intro point member
  have bounds := (mem_allPoints _ _).1 member
  have each : pointCheck cls (input.get point) = true := by
    rcases allowed with rfl | rfl <;> exact holds point bounds
  exact point_finite allowed each

theorem full_holds {cls : Class} {value : Val} (holds : Holds cls (.scalar value))
    (shape : List Nat) (elt : Ty) : Holds cls (.tensor (Tensor.full shape elt value)) := by
  cases cls with
  | any => trivial
  | finite =>
    intro point bounds
    change Idx.InBounds shape point at bounds
    simpa only [Holds, Tensor.full, Tensor.canon, bounds, ↓reduceIte] using holds
  | normal =>
    intro point bounds
    change Idx.InBounds shape point at bounds
    simpa only [Holds, Tensor.full, Tensor.canon, bounds, ↓reduceIte] using holds

/-- Finite f32 is an explicit function-entry requirement. Other types retain only the
existing initialization requirement; rank-zero f32 inputs are not assumed positive/normal. -/
def inputFact (ty : VTy) : Definedness.Refined.Fact Class :=
  ⟨⟨ty, true⟩, match ty with
    | .scalar (.float .f32) | .tensor _ (.float .f32) => .finite
    | _ => .any⟩

def inputFacts (types : List VTy) : List (Definedness.Refined.Fact Class) := types.map inputFact

#print axioms classify_sound
#print axioms point_finite
#print axioms tensor_finite
#print axioms full_holds

end MerlinVerify.Check.FiniteFacts
