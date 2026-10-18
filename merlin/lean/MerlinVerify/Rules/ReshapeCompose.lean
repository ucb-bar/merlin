/- Universally checked composition of two static collapses or two static expansions.
The intermediate source result is deliberately not mapped: any escaping use then fails C1. -/
import MerlinVerify.Rules.ReshapeGroups
import MerlinVerify.Rules.Motion
import MerlinVerify.Check.CorrLemmas

namespace MerlinVerify.Rules.ReshapeCompose

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

def Names (op : Op) : Prop :=
  dialectOf op.name = "tensor" ∧
  (op.name = "tensor.collapse_shape" ∨ op.name = "tensor.expand_shape") ∧
  unmodelledAttr? op.blind = none ∧ op.regions = [] ∧ op.successors = []

instance (op : Op) : Decidable (Names op) := by unfold Names; infer_instance

theorem nonterm {op : Op} (h : Names op) : op.isTerminator = false := by
  rcases h.2.1 with hn | hn <;> simp [Op.isTerminator, terminatorNames, hn]

theorem sem_success {op : Op} {shape : List Nat} {elt : Ty} {groups : List (List Nat)}
    (hn : Names op) (hr : Sem.Ops.Tensor.resultTensorTy op = some (shape, elt))
    (hg : Sem.Ops.Tensor.reassociation? op = some groups)
    {ctx : Ctx} {call : CallSem} {args vs : List RVal}
    (hs : opSem ctx call op args [] = .ok vs) :
    ∃ t, args = [.tensor t] ∧ reshapeValid op groups t.shape shape = true ∧
      elt = t.elt ∧ vs = [.tensor (ReshapeIndex.reshape t shape elt)] := by
  rw [tensor_sem_of hn.2.2.1 hn.1] at hs
  rcases hn.2.1 with hname | hname
  all_goals
    cases args with
    | nil => simp [Sem.Ops.Tensor.sem, hname] at hs
    | cons arg rest =>
      cases arg with
      | scalar _ => simp [Sem.Ops.Tensor.sem, hname] at hs
      | tensor t =>
        cases rest with
        | cons _ _ => simp [Sem.Ops.Tensor.sem, hname] at hs
        | nil =>
          simp only [Sem.Ops.Tensor.sem, Op.blind_name, hname, resultTensorTy_blind, hr,
            reassociation?_blind, hg, beq_self_eq_true,
            show ("tensor.expand_shape" == "tensor.collapse_shape") = false by decide,
            Bool.false_eq_true, ↓reduceIte] at hs
          split at hs
          · next hv =>
            simp only [Bool.and_eq_true, beq_iff_eq] at hv
            cases hs
            exact ⟨t, rfl, by simpa [reshapeValid, hname] using hv.1, hv.2, rfl⟩
          · cases hs

theorem sem_defined {op : Op} {shape : List Nat} {elt : Ty} {groups : List (List Nat)}
    (hn : Names op) (hr : Sem.Ops.Tensor.resultTensorTy op = some (shape, elt))
    (hg : Sem.Ops.Tensor.reassociation? op = some groups)
    (ctx : Ctx) (call : CallSem) (t : Tensor)
    (hv : reshapeValid op groups t.shape shape = true) (he : elt = t.elt) :
    opSem ctx call op [.tensor t] [] = .ok [.tensor (ReshapeIndex.reshape t shape elt)] := by
  rw [tensor_sem_of hn.2.2.1 hn.1]
  rcases hn.2.1 with hname | hname
  all_goals
    simpa [Sem.Ops.Tensor.sem, hname, resultTensorTy_blind, hr, reassociation?_blind, hg,
      reshapeValid, hname, he, ReshapeIndex.reshape] using hv

/-- The target groups are recomputed from the two actual source reassociations. Result and
operand identities, semantic attributes, and source shadowing are checked independently. -/
def Checks (ρ : Rho) (a b t : Op) (ra rb rt : Value) (x y : ValueId)
    (mid dst : List Nat) (elt : Ty) (ga gb gt : List (List Nat)) : Prop :=
  Names a ∧ Names b ∧ Names t ∧ a.name = b.name ∧ b.name = t.name ∧
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
  gt = (if a.name = "tensor.collapse_shape" then ReshapeGroups.merge ga gb
    else ReshapeGroups.merge gb ga)

instance (ρ : Rho) (a b t : Op) (ra rb rt : Value) (x y : ValueId)
    (mid dst : List Nat) (elt : Ty) (ga gb gt : List (List Nat)) :
    Decidable (Checks ρ a b t ra rb rt x y mid dst elt ga gb gt) := by
  unfold Checks; infer_instance

theorem composed_valid {a b t : Op} {mid dst src : List Nat} {ga gb gt : List (List Nat)}
    (hn : a.name = b.name) (ht : b.name = t.name)
    (hv : reshapeValid a ga src mid = true) (hb : reshapeValid b gb mid dst = true)
    (hg : gt = (if a.name = "tensor.collapse_shape" then ReshapeGroups.merge ga gb
      else ReshapeGroups.merge gb ga)) : reshapeValid t gt src dst = true := by
  unfold reshapeValid at hv hb ⊢
  by_cases hc : a.name = "tensor.collapse_shape"
  · have hbc : b.name = "tensor.collapse_shape" := hn ▸ hc
    have htc : t.name = "tensor.collapse_shape" := ht ▸ hbc
    simp only [hc, hbc, htc, beq_iff_eq, ↓reduceIte] at hv hb hg ⊢
    rw [hg]
    exact ReshapeGroups.valid_merge hv hb
  · have hbc : b.name ≠ "tensor.collapse_shape" := hn ▸ hc
    have htc : t.name ≠ "tensor.collapse_shape" := ht ▸ hbc
    simp only [hc, hbc, htc, beq_iff_eq, ↓reduceIte] at hv hb hg ⊢
    rw [hg]
    exact ReshapeGroups.valid_merge hb hv

theorem step_success {op : Op} {r : Value} {x : ValueId} {shape : List Nat} {elt : Ty}
    {groups : List (List Nat)} (hn : Names op) (hr : op.results = [r]) (ho : op.operands = [x])
    (ht : Sem.Ops.Tensor.resultTensorTy op = some (shape, elt))
    (hg : Sem.Ops.Tensor.reassociation? op = some groups)
    {ctx : Ctx} {call : CallSem} {env out : Env} (hs : stepOp opSem ctx call env op = .ok out) :
    ∃ t, env.get? x = some (.tensor t) ∧ reshapeValid op groups t.shape shape = true ∧
      elt = t.elt ∧ out = (r.id, .tensor (ReshapeIndex.reshape t shape elt)) :: env := by
  unfold stepOp at hs
  rw [hn.2.2.2.1] at hs
  simp only [evalRegions] at hs
  cases hargs : env.getAll op.operands with
  | ub => simp [hargs] at hs
  | stuck reason => simp [hargs] at hs
  | ok args =>
    rw [hargs, Outcome.bind_ok] at hs
    cases hsem : opSem ctx call op args [] with
    | ub => simp [hsem] at hs
    | stuck reason => simp [hsem] at hs
    | ok vs =>
      obtain ⟨t, rfl, hv, he, rfl⟩ := sem_success hn ht hg hsem
      have hx : env.get? x = some (.tensor t) := by
        rw [ho] at hargs
        simp only [Env.getAll, Outcome.mapM, Outcome.ofOption] at hargs
        cases he : env.get? x <;> simp_all
      exact ⟨t, hx, hv, he, by simpa [hsem, Env.bindAll, hr] using hs.symm⟩

theorem hidden_source {ρ : Rho} {envS envT : Env} {id : ValueId}
    (hc : Corr ρ envS envT) (hf : ρ.find id = none) (v : RVal) :
    Corr ρ ((id, v) :: envS) envT := Corr.bindUnmapped hc hf v

theorem site_sound {ρ : Rho} {a b t : Op} {ra rb rt : Value} {x y : ValueId}
    {mid dst : List Nat} {elt : Ty} {ga gb gt : List (List Nat)}
    (hra : a.results = [ra]) (hrb : b.results = [rb]) (hrt : t.results = [rt])
    (h : Checks ρ a b t ra rb rt x y mid dst elt ga gb gt)
    (ctx : Ctx) (callS callT : CallSem) (envS envT : Env) (hc : Corr ρ envS envT) :
    SiteRel ((rb.id, rt.id) :: ρ)
      (execOps opSem ctx callT envT [t]) (execOps opSem ctx callS envS [a, b]) := by
  obtain ⟨hna, hnb, hnt, hab, hbt, hoa, hob, hot, hxy, hfr, hhidden, -, -, hta, htb,
    htt, hga, hgb, hgt, hvb, hg⟩ := h
  rw [execOps_cons_step _ _ _ _ _ _ (nonterm hna)]
  cases hsa : stepOp opSem ctx callS envS a with
  | ub => exact .inl rfl
  | stuck reason => exact .inr (.inl ⟨reason, rfl⟩)
  | ok envA =>
    obtain ⟨input, hinput, hva, helt, rfl⟩ := step_success hna hra hoa hta hga hsa
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
    have hvt := composed_valid hab hbt hva hvb hg
    have hts : stepOp opSem ctx callT envT t = .ok ((rt.id, .tensor dstTensor) :: envT) := by
      unfold stepOp
      rw [hot, hnt.2.2.2.1]
      simp only [Env.getAll, Outcome.mapM, htarget, Outcome.ofOption, Outcome.bind_ok, evalRegions]
      rw [sem_defined hnt htt hgt ctx callT input hvt helt]
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

def rule : Rule := ⟨"reshape-compose", check, check_sound⟩

end MerlinVerify.Rules.ReshapeCompose
