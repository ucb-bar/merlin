/- Closed constants are evaluated by the actual semantics at the checked index width. Facts
describe only current runtime bindings, including exact dense tensor decoding and float bits. -/
import MerlinVerify.Check.BindingFacts
import MerlinVerify.Rules.IntegerFold

namespace MerlinVerify.Check.ConstantFacts

open MerlinVerify MerlinVerify.Sem.Ops

def reference (ib : Nat) : Ctx := { Rules.IntegerFold.reference with indexBits := ib }

def Eligible (s : Op) : Prop :=
  s.name = "arith.constant" ∧ unmodelledAttr? s.blind = none ∧
    s.operands = [] ∧ s.regions = [] ∧ s.successors = [] ∧ s.results.length = 1

instance (s : Op) : Decidable (Eligible s) := by unfold Eligible; infer_instance

def value? (ib : Nat) (s : Op) : Option RVal :=
  if Eligible s then
    match opSem (reference ib) (reference ib).ext s [] [] with
    | .ok [value] => some value
    | _ => none
  else none

theorem context_independent {ib : Nat} {s : Op} (h : Eligible s)
    {ctx : Ctx} (hi : ctx.indexBits = ib) (call : CallSem) :
    opSem ctx call s [] [] = opSem (reference ib) (reference ib).ext s [] [] := by
  have hd : dialectOf s.name = "arith" := by rw [h.1]; cbv
  rw [opSem_modelled h.2.1, opSem_modelled h.2.1]
  simp only [dispatch, Op.blind_name, hd]
  simp only [Arith.sem, Op.blind_name, h.1, hi, reference]

theorem value_spec {ib : Nat} {s : Op} {value : RVal} (h : value? ib s = some value) :
    Eligible s ∧ opSem (reference ib) (reference ib).ext s [] [] = .ok [value] := by
  unfold value? at h
  split at h
  · next he =>
    split at h
    · next v hv => cases h; exact ⟨he, hv⟩
    · cases h
  · cases h

/-- Every learned value is produced by a total step of this exact source operation. -/
theorem value_exec {ib : Nat} {s : Op} {value : RVal} (h : value? ib s = some value)
    {ctx : Ctx} (hi : ctx.indexBits = ib) (call : CallSem) (env : Env) :
    ∃ r, s.results = [r] ∧ stepOp opSem ctx call env s = .ok ((r.id, value) :: env) := by
  obtain ⟨he, hv⟩ := value_spec h
  obtain ⟨r, hr⟩ := List.length_eq_one_iff.1 he.2.2.2.2.2
  refine ⟨r, hr, ?_⟩
  unfold stepOp
  rw [he.2.2.1, he.2.2.2.1]
  simp only [Env.getAll, Outcome.mapM, Outcome.bind_ok, evalRegions,
    context_independent he hi call, hv]
  simp [Env.bindAll, hr]

structure Fact where
  id : ValueId
  value : RVal

def model : BindingFacts.Model Fact where
  dependencies f := [f.id]
  holds f env := env.get? f.id = some f.value
  congr f env env' h hf := (h f.id (by simp)).symm.trans hf

def learn (ib : Nat) (s : Op) : Option Fact := do
  let value ← value? ib s
  let [r] := s.results | none
  pure ⟨r.id, value⟩

theorem learn_sound {ib : Nat} {s : Op} {f : Fact} (h : learn ib s = some f)
    {ctx : Ctx} {call : CallSem} {env env' : Env} (hi : ctx.indexBits = ib)
    (hs : stepOp opSem ctx call env s = .ok env') : model.holds f env' := by
  unfold learn at h
  cases hv : value? ib s with
  | none => simp [hv] at h
  | some value =>
    simp only [hv, bind, Option.bind] at h
    split at h
    · next r hr =>
      cases h
      obtain ⟨r', hr', hstep⟩ := value_exec hv hi call env
      have : r' = r := by simpa [hr] using hr'.symm
      subst r'
      rw [hstep] at hs
      cases hs
      simp [model, Env.get?, List.lookup]
    · cases h

def advance (ib : Nat) (facts : List Fact) (s : Op) : List Fact :=
  let retained := BindingFacts.forget model (s.results.map (·.id)) facts
  match learn ib s with
  | some f => f :: retained
  | none => retained

theorem advance_sound {ib : Nat} {facts : List Fact} {s : Op}
    {ctx : Ctx} {call : CallSem} {env env' : Env}
    (hf : BindingFacts.Valid model facts env) (hi : ctx.indexBits = ib)
    (hs : stepOp opSem ctx call env s = .ok env') :
    BindingFacts.Valid model (advance ib facts s) env' := by
  have retained := BindingFacts.forget_step model hf hs
  unfold advance
  cases hl : learn ib s with
  | none => exact retained
  | some f =>
    intro g hg
    rcases List.mem_cons.1 hg with rfl | hg
    · exact learn_sound hl hi hs
    · exact retained g hg

end MerlinVerify.Check.ConstantFacts
