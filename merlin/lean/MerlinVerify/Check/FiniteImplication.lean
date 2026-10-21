/- Checked weakening of numerical/initialization facts at function-call boundaries.
Types must agree and every required fact must follow from the caller's actual binding. -/
import MerlinVerify.Check.FiniteFacts

namespace MerlinVerify.Check.FiniteFacts.Implication

open MerlinVerify Definedness

def classes (source target : Class) : Bool :=
  match target with
  | .any => true
  | .finite => decide (source = .finite ∨ source = .normal)
  | .normal => source == .normal

theorem classes_sound {source target : Class} (checked : classes source target = true)
    {value : RVal} (holds : Holds source value) : Holds target value := by
  cases target with
  | any => trivial
  | finite =>
    have allowed : source = .finite ∨ source = .normal := of_decide_eq_true checked
    cases value with
    | scalar value =>
      have each : pointCheck source value = true := by
        rcases allowed with rfl | rfl <;> exact holds
      exact point_finite allowed each
    | tensor value =>
      intro point bounds
      have each : pointCheck source (value.get point) = true := by
        rcases allowed with rfl | rfl <;> exact holds point bounds
      exact point_finite allowed each
  | normal =>
    have same : source = .normal := eq_of_beq checked
    simpa only [same] using holds

def fact (source target : Refined.Fact Class) : Bool :=
  source.base.ty == target.base.ty && (!target.base.initialized || source.base.initialized) &&
    classes source.info target.info

theorem fact_sound {ib : Nat} {source target : Refined.Fact Class} (checked : fact source target = true)
    {value : RVal} (holds : source.Holds model ib value) : target.Holds model ib value := by
  simp only [fact, Bool.and_eq_true, beq_iff_eq] at checked
  obtain ⟨⟨types, initialized⟩, numerical⟩ := checked
  refine ⟨⟨by simpa only [types] using holds.1.1, ?_⟩, classes_sound numerical holds.2⟩
  intro required
  simp only [required, Bool.not_true, Bool.false_or] at initialized
  exact holds.1.2 initialized

def check : List (Refined.Fact Class) → List (Refined.Fact Class) → Bool
  | [], [] => true
  | source :: sources, target :: targets => fact source target && check sources targets
  | _, _ => false

theorem check_sound {ib : Nat} : ∀ {sources targets : List (Refined.Fact Class)},
    check sources targets = true → ∀ {values : List RVal}, Refined.FactsHold model ib values sources →
      Refined.FactsHold model ib values targets
  | [], [], _, _, holds => holds
  | [], _ :: _, checked, _, _ => by cases checked
  | _ :: _, [], checked, _, _ => by cases checked
  | source :: sources, target :: targets, checked, _, holds => by
    simp only [check, Bool.and_eq_true] at checked
    cases holds with
    | cons head tail => exact .cons (fact_sound checked.1 head) (check_sound checked.2 tail)

#print axioms classes_sound
#print axioms fact_sound
#print axioms check_sound

end MerlinVerify.Check.FiniteFacts.Implication
