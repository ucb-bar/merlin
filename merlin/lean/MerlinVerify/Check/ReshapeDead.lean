/- Exact deletion of dead, total reshapes. Runtime input shape comes from the checked source
scope; continuation alpha checking includes nested captures and prevents escaping deletion. -/
import MerlinVerify.Check.ReshapeFacts
import MerlinVerify.Check.TypedDead

namespace MerlinVerify.Check.ReshapeDead

open MerlinVerify MerlinVerify.Sem.Ops

/-- A modeled reshape is total on its actual checked tensor input, including poison elements. -/
def Total (Γ : TEnv) (s : Op) : Prop :=
  Rules.ReshapeCompose.Names s ∧
  match s.operands, s.results, Sem.Ops.Tensor.resultTensorTy s, Sem.Ops.Tensor.reassociation? s with
  | [x], [_], some (dst, e), some groups =>
    match Γ.lookup x with
    | some (.tensor original elt) => elt = e ∧ reshapeValid s groups original dst = true
    | _ => False
  | _, _, _, _ => False

instance (Γ : TEnv) (s : Op) : Decidable (Total Γ s) := by
  unfold Total
  split <;> first | (split <;> infer_instance) | infer_instance

theorem total_exec {ib : Nat} {Γ : TEnv} {s : Op} (h : Total Γ s)
    {env : Env} (he : EnvTyped ib Γ env) (ctx : Ctx) (call : CallSem) :
    ∃ r value, s.results = [r] ∧ stepOp opSem ctx call env s = .ok ((r.id, value) :: env) := by
  obtain ⟨hn, h⟩ := h
  split at h
  · next x r dst e groups ho hr ht hg =>
    split at h
    · next original elt hlookup =>
      obtain ⟨rfl, hv⟩ := h
      obtain ⟨value, hvalue, hty⟩ := he _ _ hlookup
      cases value with
      | scalar _ => cases hty
      | tensor t =>
        obtain ⟨hshape, helt, _⟩ := hty
        have hsem := Rules.ReshapeCompose.sem_defined hn ht hg ctx call t
          (hshape ▸ hv) helt.symm
        refine ⟨r, .tensor (Rules.ReshapeIndex.reshape t dst elt), hr, ?_⟩
        unfold stepOp
        rw [ho, hn.2.2.2.1]
        simp [Env.getAll, Outcome.mapM, hvalue, Outcome.ofOption, evalRegions,
          hsem, Env.bindAll, hr]
    · contradiction
  · contradiction

/-- Freshness protects prior bindings. The whole normalized continuation must check without
any producer result, including uses captured by a nested region. -/
def drop? (Γ : TEnv) (s : Op) (tail : List Op) : Bool :=
  decide (Total Γ s) && TypedDead.unused Γ s tail

theorem drop_sound {ib : Nat} {Γ : TEnv} {s : Op} {tail : List Op}
    (hd : drop? Γ s tail = true) {env : Env} (he : EnvTyped ib Γ env)
    (ctx : Ctx) (call : CallSem) :
    ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env' tail = evalOps opSem ctx call env tail := by
  simp only [drop?, Bool.and_eq_true, decide_eq_true_eq] at hd
  exact TypedDead.drop_sound hd.2 (total_exec hd.1 he ctx call)

end MerlinVerify.Check.ReshapeDead
