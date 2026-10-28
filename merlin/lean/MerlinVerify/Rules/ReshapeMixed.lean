/- Mixed expand/collapse composition. A successful expansion determines its actual input shape
from the fixed expanded shape and reassociation; no source argument annotation is trusted. -/
import MerlinVerify.Rules.ReshapeCompose

namespace MerlinVerify.Rules.ReshapeMixed

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open ReshapeCompose

def inputShape (groups : List (List Nat)) (expanded : List Nat) : List Nat :=
  groups.map fun g => shapeSize (g.map fun i => expanded.getD i 0)

/-- Unlike collapse, expansion fixes every input extent as the product of its expanded group.
This premise comes from the actual successful source step, even for untyped runtime inputs. -/
theorem recover_input_shape {groups : List (List Nat)} {expanded original : List Nat}
    (h : Sem.Ops.Tensor.reassocValid groups expanded original = true) :
    original = inputShape groups expanded := by
  have hlen : groups.length = original.length := by
    have hv := h
    simp only [Sem.Ops.Tensor.reassocValid, Bool.and_eq_true, beq_iff_eq] at hv
    exact hv.1.1
  apply List.ext_getElem (by simp [inputShape, hlen])
  intro k hk hk'
  have hkg : k < groups.length := by simpa [inputShape] using hk'
  have hv := ReshapeGroups.valid_get h k hk
  simpa only [inputShape, List.getElem_map, List.getD, List.getElem?_eq_getElem hkg,
    List.getElem?_eq_getElem hk, Option.getD_some] using hv.symm

/-- Each premise is recomputed from the actual three operations. Target reassociation need
not be a particular spelling: any modeled reshape with the checked endpoints has this meaning. -/
def Checks (ρ : Rho) (a b t : Op) (ra rb rt : Value) (x y : ValueId)
    (mid dst : List Nat) (elt : Ty) (ga gb gt : List (List Nat)) : Prop :=
  Names a ∧ Names b ∧ Names t ∧ a.name = "tensor.expand_shape" ∧
  b.name = "tensor.collapse_shape" ∧
  a.operands = [x] ∧ b.operands = [ra.id] ∧ t.operands = [y] ∧
  ρ.find x = some y ∧ freshIds ρ [rt.id] = true ∧
  ρ.find ra.id = none ∧ ra.id ≠ x ∧ rb.id ≠ ra.id ∧
  Sem.Ops.Tensor.resultTensorTy a = some (mid, elt) ∧
  Sem.Ops.Tensor.resultTensorTy b = some (dst, elt) ∧
  Sem.Ops.Tensor.resultTensorTy t = some (dst, elt) ∧
  Sem.Ops.Tensor.reassociation? a = some ga ∧
  Sem.Ops.Tensor.reassociation? b = some gb ∧
  Sem.Ops.Tensor.reassociation? t = some gt ∧
  reshapeValid b gb mid dst = true ∧
  reshapeValid t gt (inputShape ga mid) dst = true

instance (ρ : Rho) (a b t : Op) (ra rb rt : Value) (x y : ValueId)
    (mid dst : List Nat) (elt : Ty) (ga gb gt : List (List Nat)) :
    Decidable (Checks ρ a b t ra rb rt x y mid dst elt ga gb gt) := by
  unfold Checks; infer_instance

theorem site_sound {ρ : Rho} {a b t : Op} {ra rb rt : Value} {x y : ValueId}
    {mid dst : List Nat} {elt : Ty} {ga gb gt : List (List Nat)}
    (hra : a.results = [ra]) (hrb : b.results = [rb]) (hrt : t.results = [rt])
    (h : Checks ρ a b t ra rb rt x y mid dst elt ga gb gt)
    (ctx : Ctx) (callS callT : CallSem) (envS envT : Env) (hc : Corr ρ envS envT) :
    SiteRel ((rb.id, rt.id) :: ρ)
      (execOps opSem ctx callT envT [t]) (execOps opSem ctx callS envS [a, b]) := by
  obtain ⟨hna, hnb, hnt, han, -, hoa, hob, hot, hxy, hfr, hhidden, -, -, hta, htb,
    htt, hga, hgb, hgt, hvb, hvt⟩ := h
  rw [execOps_cons_step _ _ _ _ _ _ (nonterm hna)]
  cases hsa : stepOp opSem ctx callS envS a with
  | ub => exact .inl rfl
  | stuck reason => exact .inr (.inl ⟨reason, rfl⟩)
  | ok envA =>
    obtain ⟨input, hinput, hva, helt, rfl⟩ := step_success hna hra hoa hta hga hsa
    have hshape : input.shape = inputShape ga mid := recover_input_shape (by
      simpa [reshapeValid, han] using hva)
    let midTensor := ReshapeIndex.reshape input mid elt
    let dstTensor := ReshapeIndex.reshape input dst elt
    have hcompose : ReshapeIndex.reshape midTensor dst elt = dstTensor :=
      ReshapeIndex.compose input mid dst elt (reshapeValid_size hvb)
    have hbs : stepOp opSem ctx callS ((ra.id, .tensor midTensor) :: envS) b =
        .ok ((rb.id, .tensor dstTensor) :: (ra.id, .tensor midTensor) :: envS) := by
      unfold stepOp
      rw [hob, hnb.2.2.2.1]
      simp only [Env.getAll, Outcome.mapM, Env.get?, List.lookup, beq_self_eq_true,
        Outcome.ofOption, Outcome.bind_ok, evalRegions]
      rw [sem_defined hnb htb hgb ctx callS midTensor hvb rfl, hcompose]
      simp [Env.bindAll, hrb]
    have htarget : envT.get? y = some (.tensor input) := (hc x y hxy).symm.trans hinput
    have hts : stepOp opSem ctx callT envT t = .ok ((rt.id, .tensor dstTensor) :: envT) := by
      unfold stepOp
      rw [hot, hnt.2.2.2.1]
      simp only [Env.getAll, Outcome.mapM, htarget, Outcome.ofOption, Outcome.bind_ok, evalRegions]
      rw [sem_defined hnt htt hgt ctx callT input (hshape ▸ hvt) helt]
      simp [Env.bindAll, hrt, dstTensor]
    simp only [Outcome.bind_ok]
    rw [execOps_cons_step _ _ _ _ _ _ (nonterm hnb), hbs, Outcome.bind_ok]
    refine .inr (.inr ⟨_, (rt.id, .tensor dstTensor) :: envT, rfl, ?_, ?_⟩)
    · rw [execOps_cons_step _ _ _ _ _ _ (nonterm hnt), hts]; rfl
    · have hf := (freshIds_sound hfr).2
      exact Corr.extend (hidden_source hc hhidden (.tensor midTensor)) [rb.id] [rt.id]
        [.tensor dstTensor] rfl rfl (by simp) (by simpa using hf)

def check (ρ : Rho) : List Op → List Op → Option Rho
  | [a, b], [t] =>
    match a.results, b.results, t.results, a.operands, t.operands,
      Sem.Ops.Tensor.resultTensorTy a, Sem.Ops.Tensor.resultTensorTy b,
      Sem.Ops.Tensor.reassociation? a, Sem.Ops.Tensor.reassociation? b,
      Sem.Ops.Tensor.reassociation? t with
    | [ra], [rb], [rt], [x], [y], some (mid, elt), some (dst, _), some ga, some gb, some gt =>
      if Checks ρ a b t ra rb rt x y mid dst elt ga gb gt then some ((rb.id, rt.id) :: ρ) else none
    | _, _, _, _, _, _, _, _, _, _ => none
  | _, _ => none

theorem check_sound : ∀ ρ ss ts ρ', check ρ ss ts = some ρ' →
    (∀ op ∈ ss, op.isTerminator = false) ∧ (∀ op ∈ ts, op.isTerminator = false) ∧
    ∀ ctx callS callT envS envT, Corr ρ envS envT →
      CallAgree callS callT (calleesOps ss ++ calleesOps ts) →
      SiteRel ρ' (execOps opSem ctx callT envT ts) (execOps opSem ctx callS envS ss)
  | ρ, [a, b], [t], ρ', h => by
    simp only [check] at h
    split at h
    · split at h
      · next hc =>
        cases h
        exact ⟨by simp only [List.mem_cons, List.not_mem_nil, or_false];
                  intro op hm; rcases hm with rfl | rfl
                  · exact nonterm hc.1
                  · exact nonterm hc.2.1,
          by simpa using nonterm hc.2.2.1,
          fun ctx callS callT envS envT hcorr _ => site_sound ‹_› ‹_› ‹_› hc ctx callS callT envS envT hcorr⟩
      · cases h
    · cases h
  | _, [], _, _, h => by simp [check] at h
  | _, [_], _, _, h => by simp [check] at h
  | _, _ :: _ :: _ :: _, _, _, h => by simp [check] at h
  | _, [_, _], [], _, h => by simp [check] at h
  | _, [_, _], _ :: _ :: _, _, h => by simp [check] at h

def rule : Rule := ⟨"reshape-mixed", check, check_sound⟩

end MerlinVerify.Rules.ReshapeMixed
