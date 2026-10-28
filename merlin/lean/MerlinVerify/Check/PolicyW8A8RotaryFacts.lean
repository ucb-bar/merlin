/- Additive exponent facts for the rotary/attention candidate. Previous numerical
facts and their input domain are unchanged. Bounds apply to every in-bounds cell. -/
import MerlinVerify.Check.PolicyW8A8NumericalScalar
import MerlinVerify.Rules.W8A8IEEEBounds

namespace MerlinVerify.Check.PolicyW8A8RotaryFacts

open MerlinVerify Rules Definedness

def pointCheck (upper : Option Nat) (value : Val) : Bool :=
  match upper with
  | none => true
  | some upper => match Sem.Ops.Dequant.float32 value with
    | none => false
    | some bits => decide (W8A8IEEEBounds.ExponentLE upper bits)

def Holds (upper : Option Nat) : RVal → Prop
  | .scalar value => pointCheck upper value = true
  | .tensor tensor => ∀ point, Idx.InBounds tensor.shape point → pointCheck upper (tensor.get point) = true

structure Info where
  prior : PolicyW8A8NumericalFacts.Info
  exponent : Option Nat := none
  deriving DecidableEq

def model : Refined.Model Info where
  holds info value := PolicyW8A8NumericalFacts.model.holds info.prior value ∧ Holds info.exponent value
  empty := ⟨PolicyW8A8NumericalFacts.model.empty, none⟩
  empty_sound value := by
    refine ⟨PolicyW8A8NumericalFacts.model.empty_sound value, ?_⟩
    cases value <;> simp [Holds, pointCheck]

def fromPrior (info : PolicyW8A8NumericalFacts.Info) : Info := ⟨info, none⟩

def priorMap : Refined.FactMap model PolicyW8A8NumericalFacts.model where
  project := Info.prior
  embed := fromPrior
  project_sound _ _ holds := holds.1
  embed_sound info value holds := by
    refine ⟨holds, ?_⟩
    cases value <;> simp [fromPrior, Holds, pointCheck]

theorem elementwise : Refined.Elementwise model := by
  intro info tensor
  constructor
  · intro holds point bounds
    exact ⟨(PolicyW8A8NumericalFacts.elementwise _ _).1 holds.1 point bounds, holds.2 point bounds⟩
  · intro holds
    exact ⟨(PolicyW8A8NumericalFacts.elementwise _ _).2 (fun point bounds => (holds point bounds).1),
      fun point bounds => (holds point bounds).2⟩

theorem literal_bits {info : Info} {bits expected : FBits .f32}
    (found : info.prior.prior.floating.literal = some expected)
    (holds : model.holds info (.scalar (.some (.flt .f32 bits)))) : bits = expected := by
  have literal := holds.1.1.1.2
  simp only [found, FloatFacts.Literal, Val.some.injEq, Scalar.flt.injEq] at literal
  exact eq_of_heq literal.2

def upper (info : Info) : Option Nat :=
  match info.exponent with
  | some cap => some cap
  | none => match info.prior.prior.floating.literal with
    | some bits => if decide (W8A8IEEE32.Finite bits) then some (W8A8ByteFloat.exponent bits) else none
    | none => if PolicyW8A8NumericalFacts.accepts info.prior .finite then some 254 else none

theorem upper_sound {ib cap : Nat} {info : Info} {bits : FBits .f32}
    (checked : upper info = some cap)
    (holds : (⟨⟨.scalar (.float .f32), true⟩, info⟩ : Refined.Fact Info).Holds model ib
      (.scalar (.some (.flt .f32 bits)))) : W8A8IEEEBounds.ExponentLE cap bits := by
  unfold upper at checked
  cases bounded : info.exponent with
  | some bound =>
    simp only [bounded, Option.some.injEq] at checked
    subst cap
    simpa [model, Holds, pointCheck, bounded, Sem.Ops.Dequant.float32] using holds.2.2
  | none =>
    simp only [bounded] at checked
    cases found : info.prior.prior.floating.literal with
    | some expected =>
      simp only [found] at checked
      split at checked
      · next finite =>
        cases checked
        rw [literal_bits found holds.2]
        exact ⟨of_decide_eq_true finite, Nat.le_refl _⟩
      · cases checked
    | none =>
      simp only [found] at checked
      split at checked
      · next finite =>
        cases checked
        have actual := PolicyW8A8NumericalScalar.requirement finite
          (show (⟨⟨.scalar (.float .f32), true⟩, info.prior⟩ : Refined.Fact PolicyW8A8NumericalFacts.Info).Holds
            PolicyW8A8NumericalFacts.model ib (.scalar (.some (.flt .f32 bits))) from ⟨holds.1, holds.2.1⟩)
        have finite : W8A8IEEE32.Finite bits := of_decide_eq_true actual
        exact ⟨finite, by omega⟩
      · cases checked

def exponentImplies : Option Nat → Option Nat → Bool
  | _, none => true
  | some left, some right => decide (left ≤ right)
  | none, some _ => false

theorem weaken_point {source target : Option Nat} {value : Val}
    (checked : exponentImplies source target = true)
    (holds : pointCheck source value = true) : pointCheck target value = true := by
  cases target with
  | none => rfl
  | some right =>
    cases source with
    | none => cases checked
    | some left =>
      have bound : left ≤ right := of_decide_eq_true checked
      cases decoded : Sem.Ops.Dequant.float32 value with
      | none => simp [pointCheck, decoded] at holds
      | some bits =>
        have actual : W8A8IEEEBounds.ExponentLE left bits := by simpa [pointCheck, decoded] using holds
        have weaker : W8A8IEEEBounds.ExponentLE right bits := ⟨actual.1, Nat.le_trans actual.2 bound⟩
        simpa [pointCheck, decoded] using weaker

theorem weaken {source target : Option Nat} {value : RVal}
    (checked : exponentImplies source target = true) (holds : Holds source value) : Holds target value := by
  cases value with
  | scalar _ => exact weaken_point checked holds
  | tensor _ => exact fun point bounds => weaken_point checked (holds point bounds)

def implies (source target : Refined.Fact Info) : Bool :=
  PolicyW8A8NumericalFacts.implies (Refined.mapFact Info.prior source) (Refined.mapFact Info.prior target) &&
    exponentImplies source.info.exponent target.info.exponent

theorem implies_sound {ib : Nat} {source target : Refined.Fact Info} {value : RVal}
    (checked : implies source target = true) (holds : source.Holds model ib value) : target.Holds model ib value := by
  simp only [implies, Bool.and_eq_true] at checked
  have previous := PolicyW8A8NumericalFacts.implies_sound checked.1
    (show (Refined.mapFact Info.prior source).Holds PolicyW8A8NumericalFacts.model ib value from
      ⟨holds.1, holds.2.1⟩)
  exact ⟨previous.1, previous.2, weaken checked.2 holds.2.2⟩

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

/-- Ordinary loop invariants forget exponent bounds. A counted accumulation or
single-write theorem must justify any stronger bound on a reduction result. -/
def invariants (ib : Nat) : Contextual.Structured.Invariants model ib where
  widen info := ⟨(PolicyW8A8NumericalFacts.invariants ib).widen info.prior, none⟩
  widen_sound info value holds := by
    refine ⟨(PolicyW8A8NumericalFacts.invariants ib).widen_sound _ _ holds.1, ?_⟩
    cases value <;> simp [Holds, pointCheck]
  implies := check
  implies_sound _ _ checked _ holds := check_sound checked holds

#print axioms priorMap
#print axioms elementwise
#print axioms implies_sound

end MerlinVerify.Check.PolicyW8A8RotaryFacts
