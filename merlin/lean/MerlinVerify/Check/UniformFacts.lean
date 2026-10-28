/- Learn uniform tensors from actual successful splat execution. These facts are a reusable
premise for the splat-input semantic kernel, not yet a whole-program rewrite checker. -/
import MerlinVerify.Check.BindingFacts
import MerlinVerify.Check.RulesTensor

namespace MerlinVerify.Check.UniformFacts

open MerlinVerify MerlinVerify.Sem.Ops

structure Fact where
  tensor : ValueId
  scalar : ValueId
  shape : List Nat
  element : Ty
  deriving DecidableEq

def Holds (f : Fact) (env : Env) : Prop :=
  ∃ value, env.get? f.scalar = some (.scalar value) ∧
    env.get? f.tensor = some (.tensor (Tensor.full f.shape f.element value))

def model : BindingFacts.Model Fact where
  dependencies f := [f.scalar, f.tensor]
  holds := Holds
  congr f env env' he h := by
    obtain ⟨value, hs, ht⟩ := h
    exact ⟨value, (he f.scalar (by simp)).symm.trans hs, (he f.tensor (by simp)).symm.trans ht⟩

/-- Source and target tensors share a payload only when their actual scalar bindings correspond.
Matching tensor types, shapes or producer names cannot establish this premise. -/
theorem coupled {source target : Fact} {envS envT : Env} {ρ : Rho}
    (hs : Holds source envS) (ht : Holds target envT)
    (hc : Corr ρ envS envT) (hm : ρ.find source.scalar = some target.scalar) :
    ∃ value, envS.get? source.scalar = some (.scalar value) ∧
      envT.get? target.scalar = some (.scalar value) ∧
      envS.get? source.tensor = some (.tensor (Tensor.full source.shape source.element value)) ∧
      envT.get? target.tensor = some (.tensor (Tensor.full target.shape target.element value)) := by
  obtain ⟨s, hscalarS, htensorS⟩ := hs
  obtain ⟨t, hscalarT, htensorT⟩ := ht
  have he := hc source.scalar target.scalar hm
  rw [hscalarS, hscalarT] at he
  have he' : s = t := by simpa using he
  subst t
  exact ⟨s, hscalarS, hscalarT, htensorS, htensorT⟩

/-- Only the actual scalar operand and static result of a supported splat can introduce a fact.
A self-shadowing result would overwrite the payload binding, so it is conservatively refused. -/
def learn (op : Op) : Option Fact :=
  if op.name = "tensor.splat" ∧ unmodelledAttr? op.blind = none then
    match op.operands, op.results, Sem.Ops.Tensor.resultTensorTy op with
    | [scalar], [result], some (shape, element) =>
      if scalar ≠ result.id then some ⟨result.id, scalar, shape, element⟩ else none
    | _, _, _ => none
  else none

/-- Constructing a splat from an actually bound scalar is total, including a poison scalar.
Freshness and correspondence of its result binding are separate rewrite obligations. -/
theorem splat_exec {op : Op} {scalar : ValueId} {result : Value} {shape : List Nat} {element : Ty}
    (hn : op.name = "tensor.splat") (hm : unmodelledAttr? op.blind = none)
    (ha : op.operands = [scalar]) (hr : op.results = [result])
    (ht : Sem.Ops.Tensor.resultTensorTy op = some (shape, element))
    (ctx : Ctx) (call : CallSem) (env : Env) {value : Val}
    (hv : env.get? scalar = some (.scalar value)) :
    stepOp opSem ctx call env op =
      .ok ((result.id, .tensor (Tensor.full shape element value)) :: env) := by
  unfold stepOp
  have hargs : env.getAll op.operands = .ok [.scalar value] := by
    simp [ha, Env.getAll, Outcome.mapM, hv, Outcome.ofOption]
  have hd : dialectOf op.name = "tensor" := by rw [hn]; cbv
  rw [hargs, Outcome.bind_ok, tensor_sem_of hm hd]
  simp [Sem.Ops.Tensor.sem, Op.blind_name, hn, resultTensorTy_blind, ht, Env.bindAll, hr]

theorem splat_step {op : Op} {scalar : ValueId} {result : Value} {shape : List Nat} {element : Ty}
    (hn : op.name = "tensor.splat") (hm : unmodelledAttr? op.blind = none)
    (ha : op.operands = [scalar]) (hr : op.results = [result])
    (ht : Sem.Ops.Tensor.resultTensorTy op = some (shape, element))
    {ctx : Ctx} {call : CallSem} {env env' : Env}
    (hs : stepOp opSem ctx call env op = .ok env') :
    ∃ value, env.get? scalar = some (.scalar value) ∧
      env' = (result.id, .tensor (Tensor.full shape element value)) :: env := by
  unfold stepOp at hs
  rw [ha] at hs
  cases hx : env.get? scalar with
  | none => simp [Env.getAll, Outcome.mapM, hx, Outcome.ofOption, Outcome.bind] at hs
  | some value =>
    have hargs : env.getAll [scalar] = .ok [value] := by
      simp [Env.getAll, Outcome.mapM, hx, Outcome.ofOption]
    have hd : dialectOf op.name = "tensor" := by rw [hn]; cbv
    rw [hargs, Outcome.bind_ok, tensor_sem_of hm hd] at hs
    simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn, resultTensorTy_blind, ht] at hs
    cases value with
    | tensor t => cases hs
    | scalar v =>
      simp [Outcome.bind, Env.bindAll, hr] at hs
      exact ⟨v, rfl, hs.symm⟩

/-- A learned fact relates the actual post-step tensor to the actual scalar payload, for every
float model, shape and initialization state. Successful execution is an explicit premise. -/
theorem learn_sound {op : Op} {fact : Fact} (hl : learn op = some fact)
    {ctx : Ctx} {call : CallSem} {env env' : Env}
    (hs : stepOp opSem ctx call env op = .ok env') : Holds fact env' := by
  unfold learn at hl
  split at hl
  · next h =>
    split at hl
    · next scalar result shape element ha hr ht =>
      split at hl
      · next hne =>
        cases hl
        obtain ⟨value, hx, rfl⟩ := splat_step h.1 h.2 ha hr ht hs
        refine ⟨value, ?_, ?_⟩
        · have hb : (scalar == result.id) = false := by simpa using hne
          simpa [Env.get?, List.lookup, hb] using hx
        · simp [Env.get?, List.lookup]
      · cases hl
    · cases hl
  · cases hl

def advance (facts : List Fact) (op : Op) : List Fact :=
  let retained := BindingFacts.forget model (op.results.map (·.id)) facts
  match learn op with
  | some fact => fact :: retained
  | none => retained

theorem advance_sound {facts : List Fact} {op : Op} {ctx : Ctx} {call : CallSem} {env env' : Env}
    (hf : BindingFacts.Valid model facts env) (hs : stepOp opSem ctx call env op = .ok env') :
    BindingFacts.Valid model (advance facts op) env' := by
  have retained := BindingFacts.forget_step model hf hs
  unfold advance
  cases hl : learn op with
  | none => exact retained
  | some fact =>
    intro f hf
    rcases List.mem_cons.1 hf with rfl | hf
    · exact learn_sound hl hs
    · exact retained f hf

def scan (facts : List Fact) (ops : List Op) : List Fact := ops.foldl advance facts

/-- Successful execution of a whole source prefix establishes every retained uniform fact.
No SSA uniqueness premise is needed: rebinding either dependency invalidates the old fact. -/
theorem scan_sound {ctx : Ctx} {call : CallSem} :
    ∀ (ops : List Op) (facts : List Fact) (env env' : Env),
      BindingFacts.Valid model facts env → execOps opSem ctx call env ops = .ok env' →
      BindingFacts.Valid model (scan facts ops) env'
  | [], _, _, _, hf, hs => by cases hs; exact hf
  | op :: ops, facts, env, env', hf, hs => by
    by_cases ht : op.isTerminator = true
    · simp [execOps, ht] at hs
    · have ht' : op.isTerminator = false := by simpa using ht
      rw [execOps_cons_step _ _ _ _ _ _ ht'] at hs
      cases he : stepOp opSem ctx call env op with
      | ub => simp [he] at hs
      | stuck r => simp [he] at hs
      | ok next =>
        rw [he, Outcome.bind_ok] at hs
        exact scan_sound ops (advance facts op) next env' (advance_sound hf he) hs

end MerlinVerify.Check.UniformFacts
