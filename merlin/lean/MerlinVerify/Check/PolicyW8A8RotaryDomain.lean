/- A separately named per-argument exponent domain. All bounds are explicit
contract data, validated against actual argument types. Acceptance also checks
an initialized, nonzero finite witness; no archive is claimed bound by this file. -/
import MerlinVerify.Check.PolicyW8A8RotaryFacts
import MerlinVerify.Check.PolicyW8A8NumericalInhabited

namespace MerlinVerify.Check.PolicyW8A8RotaryDomain

open MerlinVerify Rules Definedness PolicyW8A8RotaryFacts

def element : VTy → Ty
  | .scalar elt | .tensor _ elt => elt

def capFits (ty : VTy) : Option Nat → Bool
  | none => true
  | some upper => match ty with
    | .scalar (.float .f32) | .tensor _ (.float .f32) => upper ≤ 254
    | _ => false

def attach (fact : Refined.Fact PolicyW8A8NumericalFacts.Info) (cap : Option Nat) :
    Option (Refined.Fact Info) :=
  if capFits fact.base.ty cap then some ⟨fact.base, ⟨fact.info, cap⟩⟩ else none

def decorate : List (Refined.Fact PolicyW8A8NumericalFacts.Info) → List (Option Nat) →
    Option (List (Refined.Fact Info))
  | [], [] => some []
  | fact :: facts, cap :: caps => do
    let head ← attach fact cap
    let rest ← decorate facts caps
    return head :: rest
  | _, _ => none

def inputFacts (types : List VTy) (ranges : List (Option IndexRanges.Interval)) (caps : List (Option Nat)) :
    Option (List (Refined.Fact Info)) :=
  PolicyW8A8NumericalDefined.inputFacts types ranges >>= fun facts => decorate facts caps

def Domain (ib : Nat) (types : List VTy) (ranges : List (Option IndexRanges.Interval))
    (caps : List (Option Nat)) (args : List RVal) : Prop :=
  ∃ facts, inputFacts types ranges caps = some facts ∧ Refined.FactsHold model ib args facts

theorem decorate_prior : ∀ {facts caps bounded}, decorate facts caps = some bounded →
    bounded.map (Refined.mapFact Info.prior) = facts
  | [], [], _, checked => by cases checked; rfl
  | [], _ :: _, _, checked => by cases checked
  | _ :: _, [], _, checked => by cases checked
  | fact :: facts, cap :: caps, _, checked => by
    cases head : attach fact cap with
    | none => simp [decorate, head] at checked
    | some bounded =>
      cases rest : decorate facts caps with
      | none => simp [decorate, head, rest] at checked
      | some tail =>
        simp only [decorate, head, rest, bind, Option.bind, pure, Option.some.injEq] at checked
        subst checked
        have same : Refined.mapFact Info.prior bounded = fact := by
          unfold attach at head
          split at head
          · cases head; rfl
          · cases head
        simp only [List.map_cons, same, decorate_prior rest]

theorem domain_prior {ib : Nat} {types : List VTy} {ranges : List (Option IndexRanges.Interval)}
    {caps : List (Option Nat)} {args : List RVal} (domain : Domain ib types ranges caps args) :
    PolicyW8A8NumericalDefined.Domain ib types ranges args := by
  obtain ⟨facts, selected, holds⟩ := domain
  unfold inputFacts at selected
  cases previous : PolicyW8A8NumericalDefined.inputFacts types ranges with
  | none => simp [previous] at selected
  | some old =>
    simp only [previous, bind, Option.bind] at selected
    refine ⟨old, previous, ?_⟩
    have projected := holds.map Info.prior priorMap.project_sound
    rw [decorate_prior selected] at projected
    exact projected

def scalarCheck (ib : Nat) (elt : Ty) (info : Info) (value : Val) : Bool :=
  PolicyW8A8NumericalInhabited.scalarCheck ib elt info.prior value && pointCheck info.exponent value

theorem scalarCheck_sound {ib : Nat} {elt : Ty} {info : Info} {value : Val}
    (checked : scalarCheck ib elt info value = true) :
    (⟨⟨.scalar elt, true⟩, info⟩ : Refined.Fact Info).Holds model ib (.scalar value) := by
  simp only [scalarCheck, Bool.and_eq_true] at checked
  have prior := PolicyW8A8NumericalInhabited.scalarCheck_sound checked.1
  exact ⟨prior.1, prior.2, checked.2⟩

def seed (ib : Nat) (elt : Ty) (info : Info) : Option Val :=
  match elt, info.exponent with
  | .float .f32, some _ => some (.some (.flt .f32 1#32))
  | _, _ => PolicyW8A8NumericalInhabited.seed ib elt info.prior.prior.indices

def witness (ib : Nat) (fact : Refined.Fact Info) : Option RVal := do
  let elt := (element fact.base.ty)
  let value ← seed ib elt fact.info
  if fact.base.initialized && scalarCheck ib elt fact.info value then
    return match fact.base.ty with
      | .scalar _ => .scalar value
      | .tensor shape _ => .tensor (Tensor.full shape elt value)
  else none

theorem witness_sound {ib : Nat} {fact : Refined.Fact Info} {value : RVal}
    (checked : witness ib fact = some value) : fact.Holds model ib value := by
  unfold witness at checked
  cases selected : seed ib (element fact.base.ty) fact.info with
  | none => simp [selected] at checked
  | some scalar =>
    simp only [selected, bind, Option.bind] at checked
    split at checked
    · next accepted =>
      have pair : fact.base.initialized = true ∧ scalarCheck ib (element fact.base.ty) fact.info scalar = true := by
        simpa only [Bool.and_eq_true] using accepted
      have scalarHolds := scalarCheck_sound pair.2
      cases ty : fact.base.ty with
      | scalar elt =>
        simp only [ty, pure, Option.some.injEq] at checked
        subst value
        refine ⟨⟨?_, fun _ => scalarHolds.1.2 rfl⟩, scalarHolds.2⟩
        simpa only [ty, element] using scalarHolds.1.1
      | tensor shape elt =>
        simp only [ty, element, pure, Option.some.injEq] at checked
        subst value
        refine ⟨⟨?_, ?_⟩, ?_⟩
        · simp only [ty, RVal.HasTy, Tensor.full, Tensor.canon, true_and]
          intro point bounds
          simpa only [ty, element, bounds, ↓reduceIte, RVal.HasTy] using scalarHolds.1.1
        · intro _ point bounds
          change Idx.InBounds shape point at bounds
          simpa only [Tensor.full, Tensor.canon, bounds, ↓reduceIte, RVal.Defined] using scalarHolds.1.2 rfl
        · apply (elementwise _ _).2
          intro point bounds
          change Idx.InBounds shape point at bounds
          simpa only [Tensor.full, Tensor.canon, bounds, ↓reduceIte] using scalarHolds.2
    · cases checked

def witnesses (ib : Nat) : List (Refined.Fact Info) → Option (List RVal)
  | [] => some []
  | fact :: facts => do
    let value ← witness ib fact
    let rest ← witnesses ib facts
    return value :: rest

theorem witnesses_sound {ib : Nat} : ∀ {facts : List (Refined.Fact Info)} {values : List RVal},
    witnesses ib facts = some values → Refined.FactsHold model ib values facts
  | [], _, checked => by cases checked; exact .nil
  | fact :: facts, _, checked => by
    cases head : witness ib fact with
    | none => simp [witnesses, head] at checked
    | some value =>
      cases rest : witnesses ib facts with
      | none => simp [witnesses, head, rest] at checked
      | some tail =>
        simp only [witnesses, head, rest, bind, Option.bind, pure, Option.some.injEq] at checked
        subst checked
        exact .cons (witness_sound head) (witnesses_sound rest)

def check (ib : Nat) (types : List VTy) (ranges : List (Option IndexRanges.Interval))
    (caps : List (Option Nat)) : Bool :=
  (inputFacts types ranges caps >>= witnesses ib).isSome

theorem check_inhabited {ib : Nat} {types : List VTy} {ranges : List (Option IndexRanges.Interval)}
    {caps : List (Option Nat)} (checked : check ib types ranges caps = true) :
    ∃ args, Domain ib types ranges caps args := by
  obtain ⟨args, accepted⟩ := Option.isSome_iff_exists.1 checked
  cases selected : inputFacts types ranges caps with
  | none => simp [selected] at accepted
  | some facts =>
    simp only [selected, bind, Option.bind] at accepted
    exact ⟨args, facts, selected, witnesses_sound accepted⟩

#print axioms domain_prior
#print axioms check_inhabited

end MerlinVerify.Check.PolicyW8A8RotaryDomain
