/- Executable nonvacuity certificates for the separately named numerical input domain.
Witnesses use one checked scalar per input tensor, so even large shapes need no enumeration.
The certificate proves existence of inputs, not existence of a concrete numerical context. -/
import MerlinVerify.Check.PolicyW8A8NumericalDefined

namespace MerlinVerify.Check.PolicyW8A8NumericalInhabited

open MerlinVerify Rules Definedness PolicyW8A8NumericalFacts

def rangeCheck (range : Option IndexRanges.Interval) (value : Val) : Bool :=
  match range with
  | none => true
  | some range => match IndexRanges.value? (.scalar value) with
    | some z => decide ((range.lower : Int) ≤ z ∧ z < (range.upper : Int))
    | none => false

theorem rangeCheck_sound {range : Option IndexRanges.Interval} {value : Val}
    (checked : rangeCheck range value = true) : BoundedFloatFacts.Uniform range (.scalar value) := by
  cases range with
  | none => trivial
  | some range =>
    cases decoded : IndexRanges.value? (.scalar value) with
    | none => simp [rangeCheck, decoded] at checked
    | some z =>
      have bounds : (range.lower : Int) ≤ z ∧ z < (range.upper : Int) := by
        simpa only [rangeCheck, decoded, decide_eq_true_eq] using checked
      exact ⟨z, decoded, bounds⟩

def literalCheck (literal : Option (FBits .f32)) (value : Val) : Bool :=
  match literal with | none => true | some bits => value == .some (.flt .f32 bits)

theorem literalCheck_sound {literal : Option (FBits .f32)} {value : Val}
    (checked : literalCheck literal value = true) : FloatFacts.Literal literal (.scalar value) := by
  cases literal with
  | none => trivial
  | some bits => exact eq_of_beq checked

def scalarCheck (ib : Nat) (elt : Ty) (info : Info) (value : Val) : Bool :=
  match value with
  | .poison => false
  | .some scalar => scalar.conforms ib elt && FiniteFacts.pointCheck info.prior.floating.numeric value &&
      literalCheck info.prior.floating.literal value && rangeCheck info.prior.indices value &&
      pointCheck info.numeric value

theorem scalarCheck_sound {ib : Nat} {elt : Ty} {info : Info} {value : Val}
    (checked : scalarCheck ib elt info value = true) :
    (⟨⟨.scalar elt, true⟩, info⟩ : Refined.Fact Info).Holds model ib (.scalar value) := by
  cases value with
  | poison => cases checked
  | some scalar =>
    simp only [scalarCheck, Bool.and_eq_true] at checked
    refine ⟨⟨checked.1.1.1.1, by simp [RVal.Defined]⟩,
      ⟨⟨?_, literalCheck_sound checked.1.1.2⟩, rangeCheck_sound checked.1.2⟩, ?_⟩
    · cases cls : info.prior.floating.numeric with
      | any => trivial
      | finite | normal => simpa only [cls, FiniteFacts.model, FiniteFacts.Holds] using checked.1.1.1.2
    · cases cls : info.numeric with
      | any => trivial
      | finite | nonNaN | nonnegative | nonzeroFinite => simpa only [cls, Holds] using checked.2

def seed (ib : Nat) (elt : Ty) (range : Option IndexRanges.Interval) : Option Val :=
  match elt with
  | .float kind => some (.some (.flt kind (if kind = .f32 then BitVec.ofNat kind.width 0x3f800000 else 0)))
  | _ => (elt.bitWidth? ib).map fun width => .some (.bits width
      (BitVec.ofNat width ((range.map (·.lower)).getD 0)))

def witness (ib : Nat) (fact : Refined.Fact Info) : Option RVal := do
  let elt := match fact.base.ty with | .scalar elt | .tensor _ elt => elt
  let value ← seed ib elt fact.info.prior.indices
  if scalarCheck ib elt fact.info value then
    return match fact.base.ty with
      | .scalar _ => .scalar value
      | .tensor shape _ => .tensor (Tensor.full shape elt value)
  else none

theorem witness_sound {ib : Nat} {fact : Refined.Fact Info} {value : RVal}
    (checked : witness ib fact = some value) : fact.Holds model ib value := by
  unfold witness at checked
  cases ty : fact.base.ty with
  | scalar elt =>
    simp only [ty] at checked
    cases selected : seed ib elt fact.info.prior.indices with
    | none => simp [selected] at checked
    | some scalar =>
      simp only [selected, bind, Option.bind] at checked
      split at checked
      · next accepted =>
        cases checked
        have result := scalarCheck_sound accepted
        exact ⟨⟨by simpa only [ty] using result.1.1, fun _ => result.1.2 rfl⟩, result.2⟩
      · cases checked
  | tensor shape elt =>
    simp only [ty] at checked
    cases selected : seed ib elt fact.info.prior.indices with
    | none => simp [selected] at checked
    | some scalar =>
      simp only [selected, bind, Option.bind] at checked
      split at checked
      · next accepted =>
        cases checked
        have result := scalarCheck_sound accepted
        refine ⟨⟨?_, ?_⟩, full_holds result.2 shape elt⟩
        · simp only [ty, RVal.HasTy, Tensor.full, Tensor.canon, true_and]
          intro point bounds
          simpa only [bounds, ↓reduceIte] using (show Val.HasTy ib elt scalar from result.1.1)
        · intro _ point bounds
          change Idx.InBounds shape point at bounds
          simp only [Tensor.full, Tensor.canon, bounds, ↓reduceIte]
          exact result.1.2 rfl
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
  | fact :: facts, values, checked => by
    cases head : witness ib fact with
    | none => simp [witnesses, head] at checked
    | some value =>
      cases tail : witnesses ib facts with
      | none => simp [witnesses, head, tail] at checked
      | some rest =>
        simp only [witnesses, head, tail, bind, Option.bind] at checked
        change some (value :: rest) = some values at checked
        cases checked
        exact .cons (witness_sound head) (witnesses_sound tail)

def inputs (ib : Nat) (types : List VTy) (ranges : List (Option IndexRanges.Interval)) : Option (List RVal) :=
  PolicyW8A8NumericalDefined.inputFacts types ranges >>= witnesses ib

def check (ib : Nat) (types : List VTy) (ranges : List (Option IndexRanges.Interval)) : Bool :=
  (inputs ib types ranges).isSome

theorem check_sound {ib : Nat} {types : List VTy} {ranges : List (Option IndexRanges.Interval)}
    (checked : check ib types ranges = true) :
    ∃ args, PolicyW8A8NumericalDefined.Domain ib types ranges args := by
  obtain ⟨args, accepted⟩ := Option.isSome_iff_exists.1 checked
  unfold inputs at accepted
  cases selected : PolicyW8A8NumericalDefined.inputFacts types ranges with
  | none => simp [selected] at accepted
  | some facts =>
    simp only [selected, bind, Option.bind] at accepted
    exact ⟨args, facts, selected, witnesses_sound accepted⟩

#print axioms scalarCheck_sound
#print axioms witness_sound
#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8NumericalInhabited
