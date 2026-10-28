/- Numerical classes and exact f32 literal evidence over actual scalar/tensor elements.
Literals are learned from execution and respect the current SSA lookup, including shadowing. -/
import MerlinVerify.Check.DefinedFactMap
import MerlinVerify.Check.FiniteStructured
import MerlinVerify.Check.ConstantFacts

namespace MerlinVerify.Check.FloatFacts

open MerlinVerify Definedness

structure Info where
  numeric : FiniteFacts.Class := .any
  literal : Option (FBits .f32) := none
  deriving DecidableEq, Repr

def Literal : Option (FBits .f32) → RVal → Prop
  | none, _ => True
  | some bits, .scalar value => value = .some (.flt .f32 bits)
  | some bits, .tensor tensor =>
    ∀ point, Idx.InBounds tensor.shape point → tensor.get point = .some (.flt .f32 bits)

def model : Refined.Model Info where
  holds info value := FiniteFacts.model.holds info.numeric value ∧ Literal info.literal value
  empty := ⟨.any, none⟩
  empty_sound _ := ⟨trivial, trivial⟩

def fromClass (numeric : FiniteFacts.Class) : Info := ⟨numeric, none⟩

def finiteMap : Refined.FactMap model FiniteFacts.model where
  project := Info.numeric
  embed := fromClass
  project_sound _ _ holds := holds.1
  embed_sound _ _ holds := ⟨holds, trivial⟩

theorem literal_elementwise (literal : Option (FBits .f32)) (tensor : Tensor) :
    Literal literal (.tensor tensor) ↔
      ∀ point, Idx.InBounds tensor.shape point → Literal literal (.scalar (tensor.get point)) := by
  cases literal with
  | none => exact ⟨fun _ _ _ => trivial, fun _ => trivial⟩
  | some _ => rfl

theorem elementwise : Refined.Elementwise model := by
  intro info tensor
  constructor
  · intro holds point bounds
    exact ⟨(FiniteFacts.Structured.elementwise _ _).1 holds.1 point bounds,
      (literal_elementwise _ _).1 holds.2 point bounds⟩
  · intro holds
    exact ⟨(FiniteFacts.Structured.elementwise _ _).2 (fun point bounds => (holds point bounds).1),
      (literal_elementwise _ _).2 (fun point bounds => (holds point bounds).2)⟩

theorem full_holds {info : Info} {value : Val} (holds : model.holds info (.scalar value))
    (shape : List Nat) (elt : Ty) : model.holds info (.tensor (Tensor.full shape elt value)) := by
  apply (elementwise _ _).2
  intro point bounds
  change Idx.InBounds shape point at bounds
  simpa only [Tensor.full, Tensor.canon, bounds, ↓reduceIte] using holds

/-- Inspect only requested bindings, using the evaluator's first-binding lookup. A body's
operand ids bound this work by the body size, rather than enumerating all environment pairs. -/
def known (Γ : Refined.FEnv Info) (ids : List ValueId) : List ConstantFacts.Fact :=
  ids.filterMap fun id => do
    let fact ← Γ.lookup id
    if fact.base.ty = .scalar (.float .f32) then do
      let bits ← fact.info.literal
      pure ⟨id, .scalar (.some (.flt .f32 bits))⟩
    else none

theorem known_sound {ib : Nat} {Γ : Refined.FEnv Info} {env : Env} (ids : List ValueId)
    (valid : Refined.EnvHolds model ib Γ env) : BindingFacts.Valid ConstantFacts.model (known Γ ids) env := by
  intro literal member
  obtain ⟨id, _member, learned⟩ := List.mem_filterMap.1 member
  cases found : Γ.lookup id with
  | none => simp [found] at learned
  | some fact =>
    simp only [found, bind, Option.bind] at learned
    split at learned
    · next ty =>
      cases decoded : fact.info.literal with
      | none => simp [decoded] at learned
      | some bits =>
        simp only [decoded, pure, Option.some.injEq] at learned
        cases learned
        obtain ⟨value, bound, holds⟩ := valid id fact found
        cases value with
        | tensor _ =>
          have typed := holds.1.1
          simp only [ty, RVal.HasTy] at typed
        | scalar value =>
          have same : value = .some (.flt .f32 bits) := by
            simpa only [decoded, Literal] using holds.2.2
          simpa only [ConstantFacts.model, same] using bound
    · cases learned

#print axioms finiteMap
#print axioms elementwise
#print axioms full_holds
#print axioms known_sound

end MerlinVerify.Check.FloatFacts
