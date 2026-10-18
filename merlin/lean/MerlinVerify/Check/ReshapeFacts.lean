/- Exact reshape provenance learned from successful checked source execution. -/
import MerlinVerify.Check.BindingFacts
import MerlinVerify.Check.TypedRewrite
import MerlinVerify.Rules.ReshapeCompose

namespace MerlinVerify.Check.ReshapeFacts

open MerlinVerify MerlinVerify.Sem.Ops

structure Fact where
  result : ValueId
  root : ValueId
  original : List Nat
  shape : List Nat
  element : Ty
  deriving DecidableEq

def Holds (f : Fact) (env : Env) : Prop :=
  ∃ t, env.get? f.root = some (.tensor t) ∧ t.shape = f.original ∧ t.elt = f.element ∧
    env.get? f.result = some (.tensor (Rules.ReshapeIndex.reshape t f.shape f.element))

def model : BindingFacts.Model Fact where
  dependencies f := [f.root, f.result]
  holds := Holds
  congr f env env' h hf := by
    obtain ⟨t, hr, hs, he, hv⟩ := hf
    exact ⟨t, (h _ (by simp)).symm.trans hr, hs, he, (h _ (by simp)).symm.trans hv⟩

def learn (Γ : TEnv) (op : Op) : Option Fact :=
  if Rules.ReshapeCompose.Names op then
    match op.operands, op.results, Sem.Ops.Tensor.resultTensorTy op with
    | [root], [result], some (shape, element) =>
      match Γ.lookup root with
      | some (.tensor original e) =>
        if e = element ∧ root ≠ result.id then some ⟨result.id, root, original, shape, element⟩ else none
      | _ => none
    | _, _, _ => none
  else none

theorem learn_sound {ib : Nat} {Γ : TEnv} {op : Op} {fact : Fact}
    (hl : learn Γ op = some fact) {ctx : Ctx} {call : CallSem} {env env' : Env}
    (he : EnvTyped ib Γ env) (hs : stepOp opSem ctx call env op = .ok env') :
    Holds fact env' := by
  unfold learn at hl
  split at hl
  · next hn =>
    split at hl
    · next root result shape element ho hr ht =>
      split at hl
      · next original e hlookup =>
        split at hl
        · next hc =>
          obtain ⟨rfl, hne⟩ := hc
          cases hl
          obtain ⟨v, hv, hty⟩ := he _ _ hlookup
          cases v with
          | scalar _ => cases hty
          | tensor t =>
            obtain ⟨hts, hte, _⟩ := hty
            unfold stepOp at hs
            rw [ho, hn.2.2.2.1] at hs
            simp only [Env.getAll, Outcome.mapM, hv, Outcome.ofOption, Outcome.bind_ok, evalRegions] at hs
            cases hg : Sem.Ops.Tensor.reassociation? op with
            | none =>
              rw [tensor_sem_of hn.2.2.1 hn.1] at hs
              rcases hn.2.1 with hname | hname <;>
                simp [Sem.Ops.Tensor.sem, hname, resultTensorTy_blind, ht,
                  reassociation?_blind, hg] at hs
            | some groups =>
              cases hsem : opSem ctx call op [.tensor t] [] with
              | ub => simp [hsem] at hs
              | stuck reason => simp [hsem] at hs
              | ok vs =>
                obtain ⟨t', ht', _, _, rfl⟩ := Rules.ReshapeCompose.sem_success hn ht hg hsem
                cases ht'
                have hout : env' = (result.id, .tensor (Rules.ReshapeIndex.reshape t shape e)) :: env := by
                  simpa [hsem, Env.bindAll, hr] using hs.symm
                rw [hout]
                exact ⟨t, by simpa [Env.get?, List.lookup,
                    show (root == result.id) = false by simpa using hne] using hv,
                  hts, hte, by simp [Env.get?, List.lookup]⟩
        · cases hl
      · cases hl
    · cases hl
  · cases hl

def advance (Γ : TEnv) (facts : List Fact) (op : Op) : List Fact :=
  let retained := BindingFacts.forget model (op.results.map (·.id)) facts
  match learn Γ op with
  | some f => f :: retained
  | none => retained

theorem advance_sound {ib : Nat} {Γ : TEnv} {facts : List Fact} {op : Op}
    {ctx : Ctx} {call : CallSem} {env env' : Env}
    (he : EnvTyped ib Γ env) (hf : BindingFacts.Valid model facts env)
    (hs : stepOp opSem ctx call env op = .ok env') :
    BindingFacts.Valid model (advance Γ facts op) env' := by
  have hr := BindingFacts.forget_step model hf hs
  unfold advance
  cases hl : learn Γ op with
  | none => exact hr
  | some f =>
    intro g hg
    rcases List.mem_cons.1 hg with rfl | hg
    · exact learn_sound hl he hs
    · exact hr g hg

end MerlinVerify.Check.ReshapeFacts
