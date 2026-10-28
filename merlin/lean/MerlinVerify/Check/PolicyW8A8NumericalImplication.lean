/- Checked weakening for the numerical candidate, including inherited finite/literal facts.
Region invariants retain non-NaN/sign evidence but never invent it from initialization. -/
import MerlinVerify.Check.PolicyW8A8NumericalFacts

namespace MerlinVerify.Check.PolicyW8A8NumericalFacts

open MerlinVerify Definedness Rules

def classes (source target : Class) : Bool :=
  match target with
  | .any => true
  | .finite => source == .finite || source == .nonzeroFinite
  | .nonNaN => source != .any
  | .nonnegative => source == .nonnegative
  | .nonzeroFinite => source == .nonzeroFinite

theorem classes_bits {source target : Class} {bits : FBits .f32}
    (checked : classes source target = true) (holds : bitsCheck source bits = true) :
    bitsCheck target bits = true := by
  cases target with
  | any => rfl
  | finite =>
    cases source <;> simp_all [classes, bitsCheck]
  | nonNaN =>
    cases source with
    | any => cases checked
    | nonNaN => exact holds
    | nonnegative =>
      have pair := holds
      simp only [bitsCheck, Bool.and_eq_true] at pair
      exact pair.1
    | finite =>
      have finite : W8A8IEEE32.Finite bits := of_decide_eq_true holds
      simpa [bitsCheck] using W8A8IEEE32.finite_notNaN finite
    | nonzeroFinite =>
      have pair := holds
      simp only [bitsCheck, Bool.and_eq_true, decide_eq_true_eq] at pair
      have finite : W8A8IEEE32.Finite bits := pair.1
      simpa [bitsCheck] using W8A8IEEE32.finite_notNaN finite
  | nonnegative => cases source <;> simp_all [classes]
  | nonzeroFinite => cases source <;> simp_all [classes]

theorem classes_point {source target : Class} {value : Val}
    (checked : classes source target = true) (holds : pointCheck source value = true) :
    pointCheck target value = true := by
  by_cases targetAny : target = .any
  · subst target; rfl
  have sourceAny : source ≠ .any := by
    intro equal
    subst source
    cases target <;> simp_all [classes]
  cases decoded : Sem.Ops.Dequant.float32 value with
  | none => cases source <;> simp_all [pointCheck]
  | some bits =>
    have actual : bitsCheck source bits = true := by
      cases source with
      | any => rfl
      | finite | nonNaN | nonnegative | nonzeroFinite =>
        simpa [pointCheck, decoded] using holds
    have result := classes_bits checked actual
    cases target <;> simp_all [pointCheck]

theorem classes_holds {source target : Class} {value : RVal}
    (checked : classes source target = true) (holds : Holds source value) : Holds target value := by
  by_cases targetAny : target = .any
  · subst target; trivial
  have sourceAny : source ≠ .any := by
    intro equal
    subst source
    cases target <;> simp_all [classes]
  cases value with
  | scalar value =>
    have actual : pointCheck source value = true := by cases source <;> simp_all [Holds]
    have result := classes_point checked actual
    cases target <;> simp_all [Holds]
  | tensor tensor =>
    have actual : ∀ point, Idx.InBounds tensor.shape point → pointCheck source (tensor.get point) = true := by
      cases source <;> simp_all [Holds]
    cases target with
    | any => exact False.elim (targetAny rfl)
    | finite | nonNaN | nonnegative | nonzeroFinite =>
      exact fun point bounds => classes_point checked (actual point bounds)

theorem inherited_finite_point {value : Val} (holds : FiniteFacts.pointCheck .finite value = true) :
    pointCheck .finite value = true := by
  cases decoded : Sem.Ops.Dequant.float32 value with
  | none => simp [FiniteFacts.pointCheck, PolicyW8A8FiniteDomain.pointCheck, decoded] at holds
  | some bits =>
    simpa [FiniteFacts.pointCheck, PolicyW8A8FiniteDomain.pointCheck, pointCheck, bitsCheck, decoded] using holds

theorem inherited_finite {value : RVal} (holds : FiniteFacts.Holds .finite value) : Holds .finite value := by
  cases value with
  | scalar value =>
    exact inherited_finite_point holds
  | tensor tensor =>
    intro point bounds
    exact inherited_finite_point (holds point bounds)

def accepts (info : Info) (target : Class) : Bool :=
  classes info.numeric target ||
    (FiniteFacts.Implication.classes info.prior.floating.numeric .finite && classes .finite target) ||
    match info.prior.floating.literal with | some bits => bitsCheck target bits | none => false

theorem accepts_sound {info : Info} {target : Class} {value : RVal}
    (checked : accepts info target = true) (holds : model.holds info value) : Holds target value := by
  simp only [accepts, Bool.or_eq_true, Bool.and_eq_true] at checked
  rcases checked with (current | previous) | literal
  · exact classes_holds current holds.2
  · have finite := FiniteFacts.Implication.classes_sound previous.1 holds.1.1.1
    exact classes_holds previous.2 (inherited_finite finite)
  · cases decoded : info.prior.floating.literal with
    | none => simp [decoded] at literal
    | some bits =>
      have actual : FloatFacts.Literal (some bits) value := by simpa [decoded] using holds.1.1.2
      exact literal_holds (by simpa [decoded] using literal) actual

def implies (source target : Refined.Fact Info) : Bool :=
  BoundedFloatFacts.implies (Refined.mapFact Info.prior source) (Refined.mapFact Info.prior target) &&
    accepts source.info target.info.numeric

theorem implies_sound {ib : Nat} {source target : Refined.Fact Info} {value : RVal}
    (checked : implies source target = true) (holds : source.Holds model ib value) :
    target.Holds model ib value := by
  simp only [implies, Bool.and_eq_true] at checked
  have previous := BoundedFloatFacts.implies_sound checked.1
    (show (Refined.mapFact Info.prior source).Holds BoundedFloatFacts.model ib value from ⟨holds.1, holds.2.1⟩)
  exact ⟨previous.1, previous.2, accepts_sound checked.2 holds.2⟩

def check : List (Refined.Fact Info) → List (Refined.Fact Info) → Bool
  | [], [] => true
  | source :: sources, target :: targets => implies source target && check sources targets
  | _, _ => false

theorem check_sound {ib : Nat} : ∀ {sources targets : List (Refined.Fact Info)}, check sources targets = true →
    ∀ {values : List RVal}, Refined.FactsHold model ib values sources → Refined.FactsHold model ib values targets
  | [], [], _, _, holds => holds
  | [], _ :: _, checked, _, _ => by cases checked
  | _ :: _, [], checked, _, _ => by cases checked
  | _ :: _, _ :: _, checked, _, holds => by
    simp only [check, Bool.and_eq_true] at checked
    cases holds with
    | cons head tail => exact .cons (implies_sound checked.1 head) (check_sound checked.2 tail)

def invariants (ib : Nat) : Contextual.Structured.Invariants model ib where
  -- A finite initializer does not make a floating sum finite. Retain only the explicit
  -- numerical candidate invariant; literals, inherited finiteness and intervals are forgotten.
  widen info := ⟨BoundedFloatFacts.model.empty, info.numeric⟩
  widen_sound _info value holds := ⟨BoundedFloatFacts.model.empty_sound value, holds.2⟩
  implies := check
  implies_sound _ _ checked _ holds := check_sound checked holds

#print axioms accepts_sound
#print axioms implies_sound
#print axioms invariants

end MerlinVerify.Check.PolicyW8A8NumericalFacts
