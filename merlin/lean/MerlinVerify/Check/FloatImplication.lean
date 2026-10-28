/- Checked weakening of numerical classes and literal evidence at region/call boundaries. -/
import MerlinVerify.Check.FloatFacts

namespace MerlinVerify.Check.FloatFacts.Implication

open MerlinVerify Definedness

def literal (source target : Option (FBits .f32)) : Bool :=
  match target with
  | none => true
  | some bits => source == some bits

theorem literal_sound {source target : Option (FBits .f32)} (checked : literal source target = true)
    {value : RVal} (holds : Literal source value) : Literal target value := by
  cases target with
  | none => trivial
  | some bits =>
    have same : source = some bits := eq_of_beq checked
    exact same ▸ holds

def fact (source target : Refined.Fact Info) : Bool :=
  FiniteFacts.Implication.fact (Refined.mapFact Info.numeric source) (Refined.mapFact Info.numeric target) &&
    literal source.info.literal target.info.literal

theorem fact_sound {ib : Nat} {source target : Refined.Fact Info} (checked : fact source target = true)
    {value : RVal} (holds : source.Holds model ib value) : target.Holds model ib value := by
  simp only [fact, Bool.and_eq_true] at checked
  have numerical : (Refined.mapFact Info.numeric source).Holds FiniteFacts.model ib value :=
    ⟨holds.1, holds.2.1⟩
  have targetFacts := FiniteFacts.Implication.fact_sound checked.1 numerical
  exact ⟨targetFacts.1, targetFacts.2, literal_sound checked.2 holds.2.2⟩

def check : List (Refined.Fact Info) → List (Refined.Fact Info) → Bool
  | [], [] => true
  | source :: sources, target :: targets => fact source target && check sources targets
  | _, _ => false

theorem check_sound {ib : Nat} : ∀ {sources targets : List (Refined.Fact Info)},
    check sources targets = true → ∀ {values : List RVal}, Refined.FactsHold model ib values sources →
      Refined.FactsHold model ib values targets
  | [], [], _, _, holds => holds
  | [], _ :: _, checked, _, _ => by cases checked
  | _ :: _, [], checked, _, _ => by cases checked
  | _ :: _, _ :: _, checked, _, holds => by
    simp only [check, Bool.and_eq_true] at checked
    cases holds with
    | cons head tail => exact .cons (fact_sound checked.1 head) (check_sound checked.2 tail)

def widen (info : Info) : Info := ⟨FiniteFacts.Structured.widen info.numeric, none⟩

theorem widen_sound (info : Info) (value : RVal) (holds : model.holds info value) :
    model.holds (widen info) value :=
  ⟨FiniteFacts.Structured.widen_sound _ _ holds.1, trivial⟩

def invariants (ib : Nat) : Contextual.Structured.Invariants model ib where
  widen := widen
  widen_sound := widen_sound
  implies := check
  implies_sound _ _ checked _ holds := check_sound checked holds

#print axioms literal_sound
#print axioms fact_sound
#print axioms check_sound
#print axioms widen_sound
#print axioms invariants

end MerlinVerify.Check.FloatFacts.Implication
