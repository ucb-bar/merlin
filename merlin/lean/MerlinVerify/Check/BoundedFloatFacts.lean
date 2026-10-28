/- Floating classifications and uniform integer intervals are distinct value properties.
Uniform intervals apply to every in-bounds tensor element and therefore survive a proved
structured read. They do not assert that an arbitrary integer input satisfies a token policy. -/
import MerlinVerify.Check.FloatImplication
import MerlinVerify.Check.IndexRanges
import MerlinVerify.Check.DefinedFactTransport

namespace MerlinVerify.Check.BoundedFloatFacts

open MerlinVerify Definedness

structure Info where
  floating : FloatFacts.Info := ⟨.any, none⟩
  indices : Option IndexRanges.Interval := none
  deriving DecidableEq

def Uniform : Option IndexRanges.Interval → RVal → Prop
  | none, _ => True
  | some range, .scalar value => range.Holds (.scalar value)
  | some range, .tensor tensor =>
    ∀ point, Idx.InBounds tensor.shape point → range.Holds (.scalar (tensor.get point))

def model : Refined.Model Info where
  holds info value := FloatFacts.model.holds info.floating value ∧ Uniform info.indices value
  empty := ⟨FloatFacts.model.empty, none⟩
  empty_sound value := ⟨FloatFacts.model.empty_sound value, trivial⟩

def fromFloat (info : FloatFacts.Info) : Info := ⟨info, none⟩

def floatMap : Refined.FactMap model FloatFacts.model where
  project := Info.floating
  embed := fromFloat
  project_sound _ _ holds := holds.1
  embed_sound _ _ holds := ⟨holds, trivial⟩

theorem uniform_elementwise (range : Option IndexRanges.Interval) (tensor : Tensor) :
    Uniform range (.tensor tensor) ↔
      ∀ point, Idx.InBounds tensor.shape point → Uniform range (.scalar (tensor.get point)) := by
  cases range with
  | none => exact ⟨fun _ _ _ => trivial, fun _ => trivial⟩
  | some _ => rfl

theorem elementwise : Refined.Elementwise model := by
  intro info tensor
  constructor
  · intro holds point bounds
    exact ⟨(FloatFacts.elementwise _ _).1 holds.1 point bounds,
      (uniform_elementwise _ _).1 holds.2 point bounds⟩
  · intro holds
    exact ⟨(FloatFacts.elementwise _ _).2 (fun point bounds => (holds point bounds).1),
      (uniform_elementwise _ _).2 (fun point bounds => (holds point bounds).2)⟩

theorem full_holds {info : Info} {value : Val} (holds : model.holds info (.scalar value))
    (shape : List Nat) (elt : Ty) : model.holds info (.tensor (Tensor.full shape elt value)) := by
  apply (elementwise _ _).2
  intro point bounds
  change Idx.InBounds shape point at bounds
  simpa only [Tensor.full, Tensor.canon, bounds, ↓reduceIte] using holds

/-- Old scalar-interval rules receive tensor facts with the interval erased. Passing a
uniform tensor interval as a scalar interval would claim evidence about the wrong value. -/
def scalarRange (fact : Refined.Fact Info) : IndexRanges.Fact :=
  ⟨fact.base, match fact.base.ty with
    | .scalar _ => fact.info.indices
    | .tensor _ _ => none⟩

def toScalarRanges (ib : Nat) : Refined.FactTransport model IndexRanges.model ib where
  map := scalarRange
  sound fact value holds := by
    refine ⟨holds.1, ?_⟩
    cases ty : fact.base.ty with
    | tensor _ _ => simp [scalarRange, ty, IndexRanges.model]
    | scalar _ =>
      cases value with
      | tensor _ => have typed := holds.1.1; simp [ty, RVal.HasTy] at typed
      | scalar value =>
        cases range : fact.info.indices with
        | none => simp [scalarRange, ty, range, IndexRanges.model]
        | some _ => simpa [scalarRange, ty, range, IndexRanges.model, Uniform] using holds.2.2

def fromScalarRanges (ib : Nat) : Refined.FactTransport IndexRanges.model model ib where
  map fact := ⟨fact.base, ⟨FloatFacts.model.empty, fact.info⟩⟩
  sound fact value holds := by
    refine ⟨holds.1, FloatFacts.model.empty_sound value, ?_⟩
    cases range : fact.info with
    | none => trivial
    | some interval =>
      cases value with
      | scalar _ => simpa [IndexRanges.model, range, Uniform] using holds.2
      | tensor tensor =>
        have impossible : interval.Holds (.tensor tensor) := by simpa [IndexRanges.model, range] using holds.2
        obtain ⟨_, decoded, _⟩ := impossible
        cases decoded

def rangeImplies (source target : Option IndexRanges.Interval) : Bool :=
  match source, target with
  | _, none => true
  | some source, some target => target.lower ≤ source.lower && source.upper ≤ target.upper
  | none, some _ => false

theorem rangeImplies_sound {source target : Option IndexRanges.Interval}
    (checked : rangeImplies source target = true) {value : RVal} (holds : Uniform source value) :
    Uniform target value := by
  cases target with
  | none => trivial
  | some target =>
    cases source with
    | none => cases checked
    | some source =>
      simp only [rangeImplies, Bool.and_eq_true, decide_eq_true_eq] at checked
      have weaken : ∀ value, source.Holds value → target.Holds value := by
        intro value holds
        obtain ⟨z, decoded, lower, upper⟩ := holds
        exact ⟨z, decoded, by omega, by omega⟩
      cases value with
      | scalar _ => exact weaken _ holds
      | tensor _ => exact fun point bounds => weaken _ (holds point bounds)

def implies (source target : Refined.Fact Info) : Bool :=
  FloatFacts.Implication.fact (Refined.mapFact Info.floating source) (Refined.mapFact Info.floating target) &&
    rangeImplies source.info.indices target.info.indices

theorem implies_sound {ib : Nat} {source target : Refined.Fact Info}
    (checked : implies source target = true) {value : RVal} (holds : source.Holds model ib value) :
    target.Holds model ib value := by
  simp only [implies, Bool.and_eq_true] at checked
  have floating := FloatFacts.Implication.fact_sound checked.1 (show
    (Refined.mapFact Info.floating source).Holds FloatFacts.model ib value from ⟨holds.1, holds.2.1⟩)
  exact ⟨floating.1, floating.2, rangeImplies_sound checked.2 holds.2.2⟩

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
  -- A reduction need not preserve its initializer's interval. Forget it before checking
  -- the body invariant; complete overwrites retain their independently inferred bounds.
  widen info := ⟨FloatFacts.Implication.widen info.floating, none⟩
  widen_sound info value holds := ⟨FloatFacts.Implication.widen_sound info.floating value holds.1, trivial⟩
  implies := check
  implies_sound _ _ checked _ holds := check_sound checked holds

#print axioms floatMap
#print axioms elementwise
#print axioms full_holds
#print axioms toScalarRanges
#print axioms fromScalarRanges
#print axioms rangeImplies_sound
#print axioms invariants

end MerlinVerify.Check.BoundedFloatFacts
