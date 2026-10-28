/- Pointwise insertion facts depend only on the surviving tensor and scalar bindings. Actual
closed-coordinate values are replayed at the checked index width before learning any relation. -/
import MerlinVerify.Rules.InsertExtractKernel
import MerlinVerify.Rules.IndexFold

namespace MerlinVerify.Check.InsertExtractFacts

open MerlinVerify MerlinVerify.Sem.Ops
open Rules.InsertExtract

structure Fact where
  tensor : ValueId
  scalar : ValueId
  coords : List Int
  shape : List Nat
  element : Ty
  deriving DecidableEq

def Holds (f : Fact) (env : Env) : Prop :=
  ∃ t value, env.get? f.tensor = some (.tensor t) ∧ env.get? f.scalar = some (.scalar value) ∧
    t.shape = f.shape ∧ t.elt = f.element ∧ f.coords.length = f.shape.length ∧
    (f.coords.zip f.shape).all (fun (x, size) => 0 ≤ x ∧ x < size) = true ∧
    t.get (f.coords.map Int.toNat) = value

def model : BindingFacts.Model Fact where
  dependencies f := [f.tensor, f.scalar]
  holds := Holds
  congr f env env' h hf := by
    obtain ⟨t, value, ht, hv, hs, he, hl, hb, hp⟩ := hf
    exact ⟨t, value, (h f.tensor (by simp)).symm.trans ht,
      (h f.scalar (by simp)).symm.trans hv, hs, he, hl, hb, hp⟩

def positions (facts : List ConstantFacts.Fact) (indices : List ValueId) : Option (List Int) := do
  coordinates (← Rules.IndexFold.arguments facts indices)

theorem positions_sound {facts : List ConstantFacts.Fact} {indices : List ValueId} {coords : List Int}
    (h : positions facts indices = some coords) {env : Env}
    (hf : BindingFacts.Valid ConstantFacts.model facts env) :
    ∃ args, env.getAll indices = .ok args ∧ coordinates args = some coords := by
  unfold positions at h
  cases hi : Rules.IndexFold.arguments facts indices with
  | none => simp [hi] at h
  | some args =>
    simp only [hi, bind, Option.bind] at h
    exact ⟨args, Rules.IndexFold.arguments_sound hf indices args hi, h⟩

structure Write where
  scalar : ValueId
  destination : ValueId
  indices : List ValueId
  result : Value
  shape : List Nat
  element : Ty
  coords : List Int

private def valueDecEq (a b : Value) : Decidable (a = b) :=
  decidable_of_iff (a.id = b.id ∧ a.ty = b.ty) (by cases a; cases b; simp)
local instance : DecidableEq Value := valueDecEq

def Legal (Γ : TEnv) (constants : List ConstantFacts.Fact) (s : Op) (p : Write) : Prop :=
  s.name = "tensor.insert" ∧ unmodelledAttr? s.blind = none ∧ s.regions = [] ∧ s.successors = [] ∧
  s.operands = p.scalar :: p.destination :: p.indices ∧ s.results = [p.result] ∧
  Γ.lookup p.scalar = some (.scalar p.element) ∧ Γ.lookup p.destination = some (.tensor p.shape p.element) ∧
  Sem.Ops.Tensor.resultTensorTy s = some (p.shape, p.element) ∧
  positions constants p.indices = some p.coords ∧ p.coords.length = p.shape.length ∧
  (p.coords.zip p.shape).all (fun (x, size) => 0 ≤ x ∧ x < size) = true

instance (Γ : TEnv) (facts : List ConstantFacts.Fact) (s : Op) (p : Write) :
    Decidable (Legal Γ facts s p) := by unfold Legal; infer_instance

def proposal (Γ : TEnv) (facts : List ConstantFacts.Fact) (s : Op) : Option Write := do
  if s.name == "tensor.insert" then
    let [result] := s.results | none
    let scalar :: destination :: indices := s.operands | none
    let some (.tensor shape element) := Γ.lookup destination | none
    let coords ← positions facts indices
    pure ⟨scalar, destination, indices, result, shape, element, coords⟩
  else none

def checked (Γ : TEnv) (facts : List ConstantFacts.Fact) (s : Op) : Option Write := do
  let p ← proposal Γ facts s
  if Legal Γ facts s p then some p else none

theorem checked_spec {Γ : TEnv} {facts : List ConstantFacts.Fact} {s : Op} {p : Write}
    (h : checked Γ facts s = some p) : Legal Γ facts s p := by
  unfold checked at h
  cases hp : proposal Γ facts s with
  | none => simp [hp] at h
  | some p' =>
    simp only [hp, bind, Option.bind] at h
    split at h
    · next hc => cases h; exact hc
    · cases h

/-- Decoded source facts and actual runtime typing make this particular insertion total.
Poison payloads are still inserted as poison; no initialized-input premise is imposed here. -/
theorem write_exec {ib : Nat} {Γ : TEnv} {facts : List ConstantFacts.Fact} {s : Op} {p : Write}
    (h : Legal Γ facts s p) {env : Env} (he : EnvTyped ib Γ env)
    (hf : BindingFacts.Valid ConstantFacts.model facts env) (ctx : Ctx) (call : CallSem) :
    ∃ (t : Tensor) (value : Val), env.get? p.scalar = some (.scalar value) ∧ t.shape = p.shape ∧ t.elt = p.element ∧
      stepOp opSem ctx call env s =
        .ok ((p.result.id, .tensor (t.set (p.coords.map Int.toNat) value).canonize) :: env) := by
  obtain ⟨hn, hm, hregs, _, ho, hr, hscalar, hdest, _, hcoords, hlen, hb⟩ := h
  obtain ⟨scalar, hs, hst⟩ := he _ _ hscalar
  cases scalar with
  | tensor _ => cases hst
  | scalar value =>
    obtain ⟨destination, hd, hdt⟩ := he _ _ hdest
    cases destination with
    | scalar _ => cases hdt
    | tensor t =>
      obtain ⟨hshape, helt, _⟩ := hdt
      obtain ⟨indices, hi, hic⟩ := positions_sound hcoords hf
      refine ⟨t, value, hs, hshape, helt, ?_⟩
      have ha : env.getAll s.operands = .ok (.scalar value :: .tensor t :: indices) := by
        simp only [ho, Env.getAll, Outcome.mapM, hs, hd, Outcome.ofOption, Outcome.bind_ok]
        change (((env.getAll p.indices).bind (fun rest => .ok (.tensor t :: rest))).bind
          (fun rest => .ok (.scalar value :: rest))) = _
        rw [hi]; rfl
      unfold stepOp
      rw [ha, hregs]
      simp only [Outcome.bind_ok, evalRegions,
        Rules.InsertExtract.insert_sem hn hm hic (hshape.symm ▸ hlen) (hshape.symm ▸ hb) ctx call value]
      simp [Env.bindAll, hr]

def learn (Γ : TEnv) (facts : List ConstantFacts.Fact) (s : Op) : Option Fact := do
  let p ← checked Γ facts s
  if p.scalar ≠ p.result.id then
    some ⟨p.result.id, p.scalar, p.coords, p.shape, p.element⟩
  else none

theorem learn_sound {ib : Nat} {Γ : TEnv} {constants : List ConstantFacts.Fact} {s : Op} {f : Fact}
    (h : learn Γ constants s = some f) {ctx : Ctx} {call : CallSem} {env env' : Env}
    (he : EnvTyped ib Γ env) (hf : BindingFacts.Valid ConstantFacts.model constants env)
    (hs : stepOp opSem ctx call env s = .ok env') : Holds f env' := by
  unfold learn at h
  cases hp : checked Γ constants s with
  | none => simp [hp] at h
  | some p =>
    simp only [hp, bind, Option.bind] at h
    split at h
    · next hne =>
      cases h
      have hc := checked_spec hp
      obtain ⟨t, value, hscalar, hshape, helt, hstep⟩ := write_exec hc he hf ctx call
      rw [hstep] at hs
      cases hs
      obtain ⟨_, _, _, _, _, _, _, _, _, _, hlen, hb⟩ := hc
      refine ⟨(t.set (p.coords.map Int.toNat) value).canonize, value, ?_, ?_, ?_, ?_, hlen, hb, ?_⟩
      · simp [Env.get?, List.lookup]
      · simpa [Env.get?, List.lookup, show (p.scalar == p.result.id) = false by simpa using hne] using hscalar
      · exact hshape
      · exact helt
      · exact Rules.InsertExtract.inserted_get t value _
          (coords_inBounds p.coords t.shape (hshape.symm ▸ hlen) (hshape.symm ▸ hb))
    · cases h

def advance (Γ : TEnv) (constants : List ConstantFacts.Fact) (facts : List Fact) (s : Op) : List Fact :=
  let retained := BindingFacts.forget model (s.results.map (·.id)) facts
  match learn Γ constants s with
  | some f => f :: retained
  | none => retained

theorem advance_sound {ib : Nat} {Γ : TEnv} {constants : List ConstantFacts.Fact} {facts : List Fact}
    {s : Op} {ctx : Ctx} {call : CallSem} {env env' : Env}
    (he : EnvTyped ib Γ env) (hc : BindingFacts.Valid ConstantFacts.model constants env)
    (hf : BindingFacts.Valid model facts env) (hs : stepOp opSem ctx call env s = .ok env') :
    BindingFacts.Valid model (advance Γ constants facts s) env' := by
  have retained := BindingFacts.forget_step model hf hs
  unfold advance
  cases hl : learn Γ constants s with
  | none => exact retained
  | some f =>
    intro g hg
    rcases List.mem_cons.1 hg with rfl | hg
    · exact learn_sound hl he hc hs
    · exact retained g hg

end MerlinVerify.Check.InsertExtractFacts
